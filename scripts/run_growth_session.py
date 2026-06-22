"""Run a multi-symbol Bybit testnet growth session with a synthetic bankroll.

Goal: start from a small USDT stake (default 10) and grow it toward a profit
target (default 100 USDT), trading BTC / ETH / XRP on Bybit testnet.

Model (per the user's "10 USDT bankroll + leverage" choice):
  * A *synthetic bankroll* (starting at ``--bankroll-start`` USDT) is tracked
    inside the :class:`~kairon.live.orchestrator.TradingLoop`, NOT the live
    broker equity. Position size = ``bankroll * leverage * allocation / price``.
  * Leverage (default 10x) makes the notional large enough to clear Bybit's
    per-symbol minimum order quantities from a small stake.
  * The bankroll compounds with each closed trade's realized PnL; milestones
    (50, 100) are logged; the loop halts at ``--bankroll-stop`` (default 100).
  * The real testnet account balance serves as margin, so a synthetic-bankroll
    drawdown halts trading without liquidating the real account.

Everything is logged: per-tick decisions (full indicator snapshots + confidence
+ confluence + justification), orders, fills, heartbeats, a ``growth_ledger``
bankroll curve, and a final markdown + JSON report under ``reports/`` and
``logs/``.

Usage:
    uv run python scripts/run_growth_session.py [--duration 7200] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa

# pybit logs contain unicode arrows; force UTF-8 on Windows subprocesses.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from kairon.config import KaironSettings
from kairon.data.adapters.ccxt_adapter import CCXTAdapter
from kairon.data.symbols import CryptoVenue, Symbol, crypto_perp
from kairon.live.analytics import compute_session_report, format_report
from kairon.live.broker.bybit import BybitBroker
from kairon.live.config import BankrollConfig, LiveConfig
from kairon.live.cooldown import CooldownBrokerWrapper, CooledTradingLoop
from kairon.live.guardian import Guardian
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore
from kairon.live.strategy import ComprehensiveStrategy

# Compute once at import (sync) so async functions avoid pathlib method calls
# (ruff ASYNC240) when resolving paths.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SYMBOLS = ["BTC-USDT-PERP", "ETH-USDT-PERP", "XRP-USDT-PERP"]
DEFAULT_DURATION_SECONDS = 2 * 60 * 60  # 2 hours
DEFAULT_HISTORY_MINUTES = 90
DEFAULT_POLL_INTERVAL_SECONDS = 60.0


def _load_env(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


class MultiSymbolPollingFeed:
    """REST-polling candle feed for several symbols feeding one shared queue.

    Each symbol is polled independently; every newly-closed bar is pushed as a
    one-row OHLCV table with an extra ``symbol`` string column so the trading
    loop can route it to the right symbol. An initial history window is loaded
    per symbol so the strategy warms up immediately.
    """

    def __init__(
        self,
        symbols: list[Symbol],
        *,
        timeframe: str = "1m",
        history_minutes: int = DEFAULT_HISTORY_MINUTES,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.history_minutes = history_minutes
        self.poll_interval_seconds = poll_interval_seconds
        self.queue: asyncio.Queue = asyncio.Queue()
        self._adapter: CCXTAdapter | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        self._adapter = CCXTAdapter(venue=CryptoVenue.BYBIT, testnet=True)
        self._running = True
        logger = logging.getLogger(__name__)
        logger.info(
            "MultiSymbolPollingFeed starting for %d symbols (tf=%s, poll=%.1fs, history=%dm)",
            len(self.symbols), self.timeframe, self.poll_interval_seconds, self.history_minutes,
        )

        # Initial history per symbol so the strategy warms up immediately.
        for sym in self.symbols:
            try:
                end = datetime.now(UTC)
                start = end - timedelta(minutes=self.history_minutes)
                table = await self._adapter.afetch(sym, self.timeframe, start, end)
                if table.num_rows > 0:
                    logger.info(
                        "Feed history: %d bars loaded for %s", table.num_rows, sym.canonical
                    )
                    for i in range(table.num_rows):
                        await self.queue.put(_with_symbol(table.slice(i, 1), sym.canonical))
            except Exception as e:
                logger.warning("Feed history failed for %s: %s", sym.canonical, e)

        # One polling task per symbol.
        for sym in self.symbols:
            t = asyncio.create_task(self._poll_one(sym), name=f"poll-{sym.canonical}")
            self._tasks.append(t)

        # Run until stopped; tasks cancelled in aclose().
        with contextlib.suppress(asyncio.CancelledError):
            await self._stop.wait()

    async def _poll_one(self, sym: Symbol) -> None:
        logger = logging.getLogger(__name__)
        last_ts: object | None = None
        while self._running:
            try:
                end = datetime.now(UTC)
                start = end - timedelta(minutes=2)
                table = await self._adapter.afetch(sym, self.timeframe, start, end)
                if table.num_rows > 0:
                    latest = table.slice(table.num_rows - 1, 1)
                    ts_val = latest.column("ts")[0].as_py()
                    if ts_val != last_ts:
                        last_ts = ts_val
                        await self.queue.put(_with_symbol(latest, sym.canonical))
                        close_val = float(latest.column("close")[0].as_py())
                        logger.info(
                            "Feed: %s @ %.4f (%s)", sym.canonical, close_val, ts_val
                        )
            except Exception as e:
                logger.warning("Feed poll failed for %s: %s", sym.canonical, e)
            await asyncio.sleep(self.poll_interval_seconds)

    async def aclose(self) -> None:
        self._running = False
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            # CancelledError is a BaseException (not Exception) in Py 3.8+;
            # suppress it so cancelling poll tasks during shutdown is clean.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await t
        self._tasks = []
        if self._adapter is not None:
            await self._adapter.aclose()
            self._adapter = None


def _with_symbol(table: pa.Table, symbol_str: str) -> pa.Table:
    """Append a ``symbol`` string column to a one-row OHLCV table."""
    return table.append_column("symbol", pa.array([symbol_str], type=pa.string()))


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Synchronous JSON writer (kept out of async to avoid ASYNC230)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, indent=2, default=str))


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return logging.getLogger(f"kairon.growth.{log_file.stem}")


async def _preflight(broker: BybitBroker, symbols: list[str]) -> dict:
    health = await broker.check_health()
    if not health.get("ok"):
        raise RuntimeError(f"Health check failed: {health.get('errors')}")
    balances = await broker.get_balances()
    usdt = next((b for b in balances if b.currency == "USDT"), None)
    if usdt is None or usdt.total <= 0:
        raise RuntimeError("No USDT balance found on testnet account")
    return {"health": health, "usdt_total": usdt.total}


async def _flatten_all(broker: BybitBroker, symbols: list[str], logger: logging.Logger) -> None:
    """Flatten any open position on each symbol using reduce-only orders."""
    for sym in symbols:
        try:
            positions = await broker.get_positions(sym)
            open_qty = next((p.qty for p in positions if p.symbol == sym and p.qty > 1e-9), 0.0)
            if open_qty <= 1e-9:
                logger.info("No open position to flatten for %s", sym)
                continue
            logger.info("Flattening residual %s position qty=%.6f ...", sym, open_qty)
            order = await broker.close_position(sym)
            logger.info(
                "Close order for %s: %s qty=%.6f status=%s",
                sym, order.side.value, order.qty, order.status.value,
            )
        except Exception as e:
            logger.warning("Flatten failed for %s: %s", sym, e)


async def _run_session(args: argparse.Namespace, ts: str) -> dict:  # noqa: PLR0915
    repo_root = REPO_ROOT
    _load_env(repo_root)
    # Avoid pathlib `/` and mkdir inside the async function (ASYNC240); build
    # paths with os.path.join and let LiveStore create the parent dir.
    log_path = Path(os.path.join(str(repo_root), "logs", f"growth_{ts}.log"))
    logger = _setup_logging(log_path)
    logger.info("=" * 64)
    logger.info("Kairon GROWTH session @ %s", ts)
    logger.info("Symbols=%s  duration=%ds  bankroll_start=%.2f leverage=%.1f stop=%.2f",
                args.symbols, args.duration, args.bankroll_start, args.leverage, args.bankroll_stop)
    logger.info("=" * 64)

    settings = KaironSettings()
    db_path = Path(os.path.join(str(repo_root), "data", f"growth_{ts}.db"))
    store = LiveStore(db_path)
    store.unhalt()

    broker = BybitBroker(
        api_key=settings.bybit_api_key,
        api_secret=settings.bybit_api_secret,
        testnet=settings.bybit_testnet,
        tld=settings.bybit_tld,
    )

    start_ms = int(time.time() * 1000)
    initial_balance = 0.0
    try:
        pre = await _preflight(broker, args.symbols)
        initial_balance = pre["usdt_total"]
        logger.info("Preflight OK. Real testnet USDT balance: %.2f", initial_balance)
        store.write_heartbeat(mode="startup", equity=initial_balance, n_positions=0)
    except Exception as e:
        logger.exception("Preflight failed: %s", e)
        broker.close()
        store.close()
        raise

    # Flatten any residual positions before trading.
    await _flatten_all(broker, args.symbols, logger)

    # Set leverage per symbol.
    lev_int = int(args.leverage)
    for sym in args.symbols:
        try:
            await broker.set_leverage(sym, lev_int)
            logger.info("Leverage set to %dx for %s", lev_int, sym)
        except Exception as e:
            logger.warning("set_leverage failed for %s: %s", sym, e)

    # Symbol objects for the feed.
    sym_objs: list[Symbol] = []
    for s in args.symbols:
        base, quote = s.split("-")[0], s.split("-")[1]
        sym_objs.append(crypto_perp(base, quote, CryptoVenue.BYBIT))

    feed = MultiSymbolPollingFeed(sym_objs, timeframe="1m", history_minutes=DEFAULT_HISTORY_MINUTES)

    live_config = LiveConfig(
        symbols=tuple(args.symbols),
        timeframe="1m",
        cadence_seconds=60,
        max_daily_loss_pct=settings.live_max_daily_loss_pct,
        max_open_positions=len(args.symbols),
        warmup_bars=settings.live_warmup_bars,
        dry_run=False,
        bybit_testnet=settings.bybit_testnet,
        bybit_tld=settings.bybit_tld,
        strategy_name="comprehensive",
    )

    bankroll_cfg = BankrollConfig(
        start=args.bankroll_start,
        leverage=args.leverage,
        allocation=args.allocation,
        stop_at=args.bankroll_stop,
        milestones=(50.0, 100.0),
    )

    guardian = Guardian(
        max_position_equity_fraction=1.0,
        max_total_leverage=args.leverage,
        max_open_positions=len(args.symbols),
        max_daily_loss_pct=settings.live_max_daily_loss_pct,
        store=store,
    )
    wrapped_broker = CooldownBrokerWrapper(broker)
    reconciler = Reconciler(
        drift_tolerance_pct=0.05,
        grace_seconds=live_config.reconcile_grace_seconds,
        reconcile_interval_seconds=live_config.reconcile_interval_seconds,
        symbols=tuple(args.symbols),
        store=store,
        broker=wrapped_broker,
    )
    strategy = ComprehensiveStrategy()
    loop = CooledTradingLoop(
        config=live_config,
        broker=wrapped_broker,
        strategy=strategy,
        guardian=guardian,
        reconciler=reconciler,
        store=store,
        feed=feed,
        bankroll=bankroll_cfg,
    )

    shutdown_event = asyncio.Event()

    def _on_signal(sig: int) -> None:
        logger.info("Received signal %s, shutting down growth session...", sig)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, ValueError):
            asyncio.get_running_loop().add_signal_handler(sig, _on_signal, sig)

    background: set[asyncio.Task] = set()
    feed_task = asyncio.create_task(feed.run(), name="feed")
    background.add(feed_task)
    feed_task.add_done_callback(background.discard)
    loop_task = asyncio.create_task(loop.start(), name="loop")
    background.add(loop_task)
    loop_task.add_done_callback(background.discard)

    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=args.duration)
    except TimeoutError:
        logger.info("Session duration reached (%ds); shutting down.", args.duration)
    except asyncio.CancelledError:
        logger.info("Session cancelled.")
    finally:
        await loop.stop()
        try:
            await loop.finalize_open_positions()
        except Exception as e:
            logger.warning("finalize_open_positions failed: %s", e)
        await _flatten_all(broker, args.symbols, logger)
        await feed.aclose()
        await broker.aclose()

        try:
            balances = await broker.get_balances()
            usdt = next((b for b in balances if b.currency == "USDT"), None)
            final_balance = usdt.total if usdt else 0.0
            store.write_heartbeat(mode="shutdown", equity=final_balance, n_positions=0)
            logger.info("Final real USDT balance: %.2f", final_balance)
        except Exception as e:
            logger.error("Could not fetch final balance: %s", e)
            final_balance = 0.0

        closed_pnl: list[dict] = []
        try:
            end_ms = int(time.time() * 1000)
            for sym in args.symbols:
                closed_pnl.extend(
                    await broker.get_closed_pnl(
                        sym, start_time_ms=start_ms, end_time_ms=end_ms, limit=200
                    )
                )
            logger.info("Closed PnL entries (all symbols): %d", len(closed_pnl))
        except Exception as e:
            logger.error("Could not fetch closed PnL: %s", e)

        store.close()

    # Recompute analytics + growth summary from the DB.
    store2 = LiveStore(db_path)
    report = compute_session_report(store2, timeframe="1m", session_id=f"growth-{ts}")
    ledger = store2.get_ledger()
    store2.close()

    growth = _growth_summary(ledger, bankroll_cfg)
    report_data = {
        "session_id": f"growth-{ts}",
        "symbols": args.symbols,
        "duration_seconds": args.duration,
        "bankroll_start": bankroll_cfg.start,
        "bankroll_end": growth["bankroll_end"],
        "bankroll_peak": growth["bankroll_peak"],
        "milestones_hit": growth["milestones_hit"],
        "initial_balance_real": initial_balance,
        "final_balance_real": final_balance,
        "real_pnl": final_balance - initial_balance,
        "closed_pnl_count": len(closed_pnl),
        "closed_pnl_realized": sum(float(t.get("closedPnl", 0) or 0) for t in closed_pnl),
        "ledger": ledger,
        "report": format_report(report),
    }
    report_json = Path(os.path.join(str(repo_root), "logs", f"growth_{ts}.report.json"))
    _write_json_file(report_json, report_data)
    logger.info("Growth JSON report saved to %s", report_json)

    _write_markdown_report(
        repo_root, ts, args, report_data, report, growth, initial_balance, final_balance
    )
    print("\n" + "=" * 64)
    print("GROWTH SESSION SUMMARY")
    print("=" * 64)
    print(f"Bankroll: {bankroll_cfg.start:.2f} -> {growth['bankroll_end']:.2f} USDT "
          f"(peak {growth['bankroll_peak']:.2f})")
    print(f"Milestones hit: {growth['milestones_hit']}")
    print(f"Real account: {initial_balance:.2f} -> {final_balance:.2f} USDT")
    print(f"Closed PnL entries: {len(closed_pnl)}")
    print("=" * 64)
    return report_data


def _growth_summary(ledger: list[dict], cfg: BankrollConfig) -> dict:
    bankroll = cfg.start
    peak = cfg.start
    milestones_hit: list[float] = []
    for row in ledger:
        if row["kind"] == "close":
            bankroll = float(row["bankroll"])
        bankroll = max(bankroll, float(row["bankroll"]))
        peak = max(peak, float(row["bankroll"]))
        if row["kind"] == "milestone":
            milestones_hit.append(float(row["bankroll"]))
    # The final bankroll is the last ledger row's bankroll (most recent state).
    if ledger:
        bankroll = float(ledger[-1]["bankroll"])
    return {
        "bankroll_end": bankroll,
        "bankroll_peak": peak,
        "milestones_hit": milestones_hit,
    }


def _write_markdown_report(
    repo_root: Path,
    ts: str,
    args: argparse.Namespace,
    report_data: dict[str, Any],
    report: object,
    growth: dict[str, Any],
    initial_balance: float,
    final_balance: float,
) -> None:
    lines = [
        f"# Kairon Growth Session Report — {ts}",
        "",
        f"- Symbols: {', '.join(args.symbols)}",
        f"- Duration: {args.duration // 60} minutes ({args.duration}s)",
        f"- Bankroll model: synthetic, start={args.bankroll_start} USDT, "
        f"leverage={args.leverage}x, allocation={args.allocation}, stop_at={args.bankroll_stop}",
        "",
        "## Growth (synthetic bankroll)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Start bankroll | {args.bankroll_start:.2f} USDT |",
        f"| End bankroll | {growth['bankroll_end']:.2f} USDT |",
        f"| Peak bankroll | {growth['bankroll_peak']:.2f} USDT |",
        f"| Growth | {growth['bankroll_end'] - args.bankroll_start:+.2f} USDT "
        f"({(growth['bankroll_end'] / args.bankroll_start - 1) * 100:+.1f}%) |",
        f"| Milestones hit | {growth['milestones_hit']} |",
        "",
        "## Real testnet account",
        "",
        "| Initial | Final | Realized PnL | Closed PnL entries |",
        "|---|---|---|---|",
        f"| {initial_balance:.2f} | {final_balance:.2f} | {final_balance - initial_balance:+.2f} "
        f"| {report_data['closed_pnl_count']} |",
        "",
        "## Bankroll ledger (curve)",
        "",
        "| ts | kind | bankroll | delta | symbol | note |",
        "|---|---|---|---|---|---|",
    ]
    for row in report_data["ledger"]:
        lines.append(
            f"| {row['ts']} | {row['kind']} | {float(row['bankroll']):.2f} | "
            f"{float(row['delta']):+.4f} | {row['symbol'] or ''} | {row['note'] or ''} |"
        )
    lines.append("")
    lines.append("## Session analytics")
    lines.append("")
    lines.append("```")
    lines.append(format_report(report))
    lines.append("```")
    lines.append("")
    path = repo_root / "reports" / f"growth_{ts}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Growth markdown report written to: {path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_growth_session", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION_SECONDS, help="Seconds to run")
    p.add_argument("--bankroll-start", type=float, default=10.0)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--allocation", type=float, default=1.0)
    p.add_argument(
        "--bankroll-stop", type=float, default=100.0, help="Bankroll level that halts the loop"
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned config and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    print("Kairon GROWTH session plan")
    print(f"  symbols        : {args.symbols}")
    print(f"  duration       : {args.duration}s ({args.duration // 60} min)")
    print(f"  bankroll start : {args.bankroll_start} USDT")
    print(f"  leverage       : {args.leverage}x")
    print(f"  allocation     : {args.allocation}")
    print(f"  stop_at        : {args.bankroll_stop} USDT")
    print(f"  session id     : growth-{ts}")
    if args.dry_run:
        print("Dry run: no workers launched.")
        return 0
    try:
        asyncio.run(_run_session(args, ts))
    except Exception as e:
        print(f"ERROR: session failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
