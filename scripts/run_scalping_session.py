"""Run a multi-symbol Bybit testnet SCALPING session with a synthetic bankroll.

Goal: compound a small USDT stake (default 10) toward a profit target (default
100 USDT) *fast*, via high-frequency short-tilted scalping across several
symbols in parallel, while keeping risk bounded by best-practice controls.

Model (synthetic bankroll + leverage, per the approved growth model):
  * A *synthetic bankroll* (starting at ``--bankroll-start`` USDT) is tracked
    inside the :class:`~kairon.live.orchestrator.TradingLoop`, NOT the live
    broker equity. The real testnet balance (~10.5k USDT) serves as margin, so a
    synthetic-bankroll drawdown halts trading without liquidating the real
    account.
  * Leverage (default 10x) makes the notional large enough to clear Bybit's
    per-symbol minimum order quantities from a small stake.

Scalping risk management (the "minimal-to-medium risk, best practices" part):
  * **Fixed-fractional risk sizing**: each trade risks ``--risk-per-trade``
    (default 2.5%) of the bankroll over an ATR-based stop distance, capped by
    the leverage notional. A stop-out therefore loses a *known* fraction, not an
    arbitrary amount.
  * **ATR-based TP/SL attached natively** to each entry order (``attach_stops``)
    so Bybit manages exits server-side; TP respects ``--rr-ratio`` (default 1.3).
  * **Bankroll drawdown halt**: halts at ``--max-drawdown`` (default 30% from
    peak), plus the existing ``--bankroll-stop`` profit target and depletion halt.
  * **Daily-loss kill switch** on the real account equity (backstop).
  * **Per-symbol SL cooldown** (``--cooldown`` * 4) so a stopped-out symbol is
    not immediately re-entered; post-order cooldown is ``--cooldown`` (default 45s).
  * **Trend filter**: the strategy refuses to short into a strong uptrend (and
    refuses to long into a strong downtrend), preserving the repo's safety.

Everything is logged: per-tick decisions (full indicator snapshots + confidence
+ confluence + justification + sl/tp), orders, fills, heartbeats, a
``growth_ledger`` bankroll curve, and a final markdown + JSON report.

Usage:
    uv run python scripts/run_scalping_session.py [--duration 3600] [--dry-run]
    uv run python scripts/run_scalping_session.py --short-only --risk-per-trade 0.03
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sqlite3
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
from kairon.live.drift_killswitch import DriftKillSwitch, DriftKillSwitchConfig
from kairon.live.guardian import Guardian
from kairon.live.orderflow import compute_orderflow
from kairon.live.pure_fns import classify_symbol_risk_cap
from kairon.live.reconciler import Reconciler
from kairon.live.setup_matrix import LONG_ONLY, MEAN_REVERSION_ONLY, SetupMatrix
from kairon.live.store import LiveStore
from kairon.live.strategy import ScalpingStrategy

# Compute once at import (sync) so async functions avoid pathlib method calls
# (ruff ASYNC240) when resolving paths.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Symbol selection rationale (Bybit testnet, $10 synthetic bankroll):
#   * AVAX-USDT-PERP is delisted on testnet (ErrCode 110074 "not live").
#   * BTC-USDT-PERP ~$150k: the risk-sized qty (risk_amount / sl_distance) falls
#     below BTC's 0.001 min lot from a $10 stake, so it skips every tick. Add BTC
#     back once the bankroll compounds past ~$13 (or raise --risk-per-trade).
#   * ETH/SOL/XRP all clear both the ~$5 min-notional AND the min-lot from $10.
DEFAULT_SYMBOLS = ["ETH-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP"]
DEFAULT_DURATION_SECONDS = 60 * 60  # 1 hour
DEFAULT_HISTORY_MINUTES = 90
DEFAULT_POLL_INTERVAL_SECONDS = 15.0
DEFAULT_COOLDOWN_SECONDS = 45.0
DEFAULT_CADENCE_SECONDS = 10
DEFAULT_RISK_PER_TRADE = 0.025
# Default live timeframe. The setup-selection matrix was data-discovered and
# validated on 5m/15m bars (SOL 5m mr_long 62% win, SOL 15m 68%); 1m is noisier
# and outside the validated edge, so the runner defaults to 5m to align live
# execution with the in-sample regime. Override with --timeframe.
DEFAULT_TIMEFRAME = "5m"


def _tf_minutes(timeframe: str) -> int:
    """Parse a CCXT timeframe string ("1m","5m","15m","1h",...) to whole minutes."""
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"unsupported timeframe {timeframe!r}")


def _default_poll_for(timeframe: str) -> float:
    """Poll cadence that samples each bar ~4-5x without excess API churn.

    ~tf/4 clamped to >=15s (the 1m floor) so 1m stays at 15s, 5m -> 75s, 15m -> 225s.
    """
    return max(DEFAULT_POLL_INTERVAL_SECONDS, (_tf_minutes(timeframe) * 60) / 4.0)
DEFAULT_RR_RATIO = 1.3
DEFAULT_MAX_DRAWDOWN = 0.30
DEFAULT_RISK_CAP_TOL = 0.10

# Representative testnet prices + Bybit per-symbol min lots for the DRY-RUN
# risk-cap preflight (no broker connection in dry-run, so no live prices). The
# LIVE preflight at session start fetches the real last price and the broker's
# actual min_qty_for, so these static values only shape the plan preview.
_DRYRUN_PRICES: dict[str, float] = {
    "BTC-USDT-PERP": 150_000.0, "ETH-USDT-PERP": 4_000.0,
    "SOL-USDT-PERP": 150.0, "XRP-USDT-PERP": 2.5,
    "AVAX-USDT-PERP": 40.0, "LINK-USDT-PERP": 20.0,
}
_DRYRUN_MIN_QTY: dict[str, float] = {
    "BTC-USDT-PERP": 0.001, "ETH-USDT-PERP": 0.01,
    "SOL-USDT-PERP": 0.1, "XRP-USDT-PERP": 1.0,
    "AVAX-USDT-PERP": 0.1, "LINK-USDT-PERP": 0.1,
}


def _classify_symbols(
    symbols: list[str],
    *,
    bankroll: float,
    risk_per_trade: float,
    leverage: float,
    allocation: float,
    sl_distance_pct: float,
    prices: dict[str, float],
    min_qtys: dict[str, float],
) -> list[dict]:
    """Classify each symbol against the risk cap (shared by dry-run + live).

    Uses :func:`classify_symbol_risk_cap` from pure_fns so the preflight uses the
    EXACT same sizing math as the per-tick guard — no drift between the plan
    preview, the startup clearance, and the runtime skip decision.
    """
    out: list[dict] = []
    for sym in symbols:
        price = prices.get(sym)
        min_qty = min_qtys.get(sym)
        if price is None or min_qty is None:
            out.append({"symbol": sym, "verdict": "unknown",
                        "note": "no representative price/min-lot for dry-run"})
            continue
        out.append(classify_symbol_risk_cap(
            symbol=sym, bankroll=bankroll, risk_per_trade=risk_per_trade,
            leverage=leverage, allocation=allocation, min_qty=min_qty,
            price=price, sl_distance_pct=sl_distance_pct,
        ))
    return out


def _format_preflight(rows: list[dict]) -> str:
    """Render the risk-cap preflight as a printable table."""
    lines = [
        "Risk-cap preflight (sl_distance_pct = strategy max_sl_pct, the worst "
        "case for clearing the min lot):",
        "  symbol              verdict                 risk_qty   min_qty   "
        "min_bankroll_to_clear",
    ]
    for r in rows:
        if r["verdict"] == "unknown":
            lines.append(f"  {r['symbol']:<19} {r['verdict']:<23} {r['note']}")
            continue
        lines.append(
            f"  {r['symbol']:<19} {r['verdict']:<23} "
            f"{float(r['risk_qty']):>9.4f} {float(r['min_qty']):>9.4f}   "
            f"{float(r['min_bankroll_to_clear']):>18.2f}"
        )
    return "\n".join(lines)


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
    loop can route it to the right symbol. History pre-warming is NOT done here
    — it is seeded directly into the loop's buffer via ``TradingLoop.prewarm``
    before the loop starts, so stale historical bars never reach the order path
    (only freshly-closed live bars do, keeping attached TP/SL on the correct side
    of the current market).
    """

    def __init__(
        self,
        symbols: list[Symbol],
        *,
        timeframe: str = "1m",
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
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
            "MultiSymbolPollingFeed starting for %d symbols (tf=%s, poll=%.1fs)",
            len(self.symbols), self.timeframe, self.poll_interval_seconds,
        )

        # One polling task per symbol (live bars only; history is pre-seeded).
        for sym in self.symbols:
            t = asyncio.create_task(self._poll_one(sym), name=f"poll-{sym.canonical}")
            self._tasks.append(t)

        # Run until stopped; tasks cancelled in aclose().
        with contextlib.suppress(asyncio.CancelledError):
            await self._stop.wait()

    async def _poll_one(self, sym: Symbol) -> None:
        logger = logging.getLogger(__name__)
        last_ts: object | None = None
        tf_min = _tf_minutes(self.timeframe)
        while self._running:
            try:
                now = datetime.now(UTC)
                # Fetch a few bars so we always see at least the last closed bar
                # plus the in-progress one. The window scales with the timeframe
                # (a 1m-sized 2-min fetch would starve 5m/15m).
                start = now - timedelta(minutes=tf_min * 4)
                table = await self._adapter.afetch(sym, self.timeframe, start, now)
                # Emit only *closed* bars — drop the in-progress tail — so the
                # strategy acts on the same closed-bar semantics the in-sample
                # edge was validated on (out-of-sample alignment). Cost: up to
                # one poll interval of latency before a freshly-closed bar is
                # acted on.
                table = _drop_inprogress_bars(table, now, tf_min)
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


def _drop_inprogress_bars(table: pa.Table, now: datetime, tf_min: int) -> pa.Table:
    """Return ``table`` with the in-progress tail bar dropped.

    Exchanges return the in-progress (not-yet-closed) bar as the last row.
    Its close is just the current price part-way through the bar, so feeding it
    to the strategy would trade a partially-formed bar instead of the closed bar
    the in-sample edge was validated on. Dropping it keeps live execution
    aligned with the closed-bar backtest semantics (out-of-sample alignment).

    Only the last row can be in-progress (OHLCV is open-ts ascending, and ``now``
    falls inside the last bar's window); if it is already closed the table is
    returned unchanged.
    """
    if table.num_rows == 0:
        return table
    bar_delta = timedelta(minutes=tf_min)
    last_ts = table.column("ts")[table.num_rows - 1].as_py()
    if last_ts + bar_delta <= now:
        return table  # last bar already closed
    return table.slice(0, table.num_rows - 1)


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
    return logging.getLogger(f"kairon.scalping.{log_file.stem}")


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
            # Cancel any orphan attached TP/SL conditional orders left after the
            # flatten so they cannot fire on a future move with no position.
            with contextlib.suppress(Exception):
                await broker.cancel_all(sym)
        except Exception as e:
            logger.warning("Flatten failed for %s: %s", sym, e)


async def _risk_cap_preflight_live(
    broker: BybitBroker,
    symbols: list[str],
    bankroll_cfg: BankrollConfig,
    sl_distance_pct: float,
    logger: logging.Logger,
) -> list[dict]:
    """Live startup risk-cap preflight: classify each symbol with real prices.

    Fetches the real last price (``broker.get_last_price``) and the broker's
    actual ``min_qty_for`` per symbol, then runs the shared
    :func:`_classify_symbols` so the clearance verdict uses the exact same sizing
    math as the per-tick guard. Symbols that cannot clear their min lot at this
    bankroll (or whose min-lot overshoot would breach the cap) are logged loudly
    so the user knows they will skip every tick — and at what bankroll they unlock.
    """
    prices: dict[str, float] = {}
    min_qtys: dict[str, float] = {}
    for sym in symbols:
        try:
            lp = await broker.get_last_price(sym)
            if lp is not None and lp > 0:
                prices[sym] = float(lp)
        except Exception as e:
            logger.warning("get_last_price failed for %s: %s", sym, e)
        min_qtys[sym] = float(broker.min_qty_for(sym))
    rows = _classify_symbols(
        symbols,
        bankroll=bankroll_cfg.start,
        risk_per_trade=bankroll_cfg.risk_per_trade,
        leverage=bankroll_cfg.leverage,
        allocation=bankroll_cfg.allocation,
        sl_distance_pct=sl_distance_pct,
        prices=prices,
        min_qtys=min_qtys,
    )
    logger.info(_format_preflight(rows))
    n_tradeable = sum(1 for r in rows if r["verdict"] == "tradeable")
    logger.info(
        "Risk-cap preflight: %d/%d symbols tradeable at bankroll=%.2f "
        "(enforce_risk_cap=%s allow_min_lot_overshoot=%s tol=%.2f)",
        n_tradeable, len(rows), bankroll_cfg.start,
        bankroll_cfg.enforce_risk_cap, bankroll_cfg.allow_min_lot_overshoot,
        bankroll_cfg.risk_cap_tol,
    )
    return rows


async def _prewarm_buffers(
    loop: CooledTradingLoop, sym_objs: list[Symbol], logger: logging.Logger,
    *, timeframe: str = DEFAULT_TIMEFRAME,
) -> None:
    """Fetch recent history per symbol and seed the loop's bar buffer (no orders).

    History is seeded into the strategy's rolling buffer via
    :meth:`TradingLoop.prewarm` so the strategy is warm at start, but the seeded
    bars never reach the order path. Only freshly-closed live bars from the
    feed are traded, so attached TP/SL are relative to current market — not a
    stale historical close.
    """
    adapter = CCXTAdapter(venue=CryptoVenue.BYBIT, testnet=True)
    try:
        end = datetime.now(UTC)
        # Scale the lookback by timeframe so the strategy actually warms up: a
        # 30-bar warmup on 5m needs >=150 min of history (DEFAULT_HISTORY_MINUTES
        # of 90 would only yield ~18 5m bars — too few). Use enough wall-clock
        # minutes to cover ~2x the warmup window on this timeframe.
        tf_min = _tf_minutes(timeframe)
        warmup = max(getattr(loop, "_config", None).warmup_bars if loop else 30, 30)
        lookback = max(DEFAULT_HISTORY_MINUTES, warmup * tf_min * 2)
        start = end - timedelta(minutes=lookback)
        for sym in sym_objs:
            try:
                table = await adapter.afetch(sym, timeframe, start, end)
                # Seed only *closed* bars (drop the in-progress tail) so prewarm
                # and the live feed share closed-bar semantics with no overlap.
                table = _drop_inprogress_bars(table, end, tf_min)
                n = loop.prewarm(sym.canonical, table)
                logger.info("Prewarmed %s with %d history bars (%s)", sym.canonical, n, timeframe)
            except Exception as e:
                logger.warning("Prewarm failed for %s: %s", sym.canonical, e)
    finally:
        await adapter.aclose()


def _scalping_extras(db_path: Path) -> dict[str, Any]:
    """Query closed-trade outcomes + win/loss stats directly from the SQLite DB."""
    extras: dict[str, Any] = {
        "n_trades": 0, "n_tp": 0, "n_sl": 0, "n_manual": 0,
        "wins": 0, "losses": 0, "avg_win": 0.0, "avg_loss": 0.0, "realized_total": 0.0,
        "per_setup": [],
    }
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT symbol, realized_pnl FROM live_closed_trades ORDER BY id"
        ).fetchall()
        wins = [float(r["realized_pnl"]) for r in rows if float(r["realized_pnl"]) > 0]
        losses = [float(r["realized_pnl"]) for r in rows if float(r["realized_pnl"]) < 0]
        outcomes = conn.execute(
            "SELECT outcome, COUNT(*) c FROM live_decisions WHERE outcome IS NOT NULL "
            "GROUP BY outcome"
        ).fetchall()
        extras.update({
            "n_trades": len(rows),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "realized_total": sum(float(r["realized_pnl"]) for r in rows),
            "n_tp": next((int(r["c"]) for r in outcomes if r["outcome"] == "hit_tp"), 0),
            "n_sl": next((int(r["c"]) for r in outcomes if r["outcome"] == "hit_sl"), 0),
            "n_manual": next((int(r["c"]) for r in outcomes if r["outcome"] == "manual_close"), 0),
        })
        # Per-setup edge breakdown (Phase 2/3): bucket closed decisions by the
        # setup_id recorded in their indicator snapshot, so the live report
        # shows which setups actually paid on fresh bars vs the in-sample edge.
        setup_rows = conn.execute(
            "SELECT "
            " COALESCE(json_extract(indicators_json, '$.setup_id'), '(none)') AS sid, "
            " COUNT(*) AS n, "
            " SUM(CASE WHEN outcome_pnl > 0 THEN 1 ELSE 0 END) AS wins, "
            " SUM(CASE WHEN outcome = 'hit_tp' THEN 1 ELSE 0 END) AS tp, "
            " SUM(CASE WHEN outcome = 'hit_sl' THEN 1 ELSE 0 END) AS sl, "
            " COALESCE(SUM(outcome_pnl), 0.0) AS sum_pnl "
            "FROM live_decisions WHERE outcome IS NOT NULL "
            "GROUP BY sid ORDER BY sid"
        ).fetchall()
        extras["per_setup"] = [
            {
                "setup_id": r["sid"],
                "n": int(r["n"]),
                "wins": int(r["wins"]),
                "tp": int(r["tp"]),
                "sl": int(r["sl"]),
                "sum_pnl": float(r["sum_pnl"]),
                "win_rate": (int(r["wins"]) / int(r["n"])) if int(r["n"]) else 0.0,
            }
            for r in setup_rows
        ]
    except Exception as e:
        logging.getLogger(__name__).warning("scalping extras query failed: %s", e)
    finally:
        if conn is not None:
            conn.close()
    return extras


def _build_scalping_broker(settings: KaironSettings) -> BybitBroker:
    """Build the testnet-only broker for the scalping runner.

    Hard-enforces testnet: refuses to start when ``BYBIT_TESTNET=false`` so an
    env override cannot silently route the scalping strategy (with attached
    leverage) at the real mainnet account. The feed also hardcodes
    ``testnet=True``, so a venue mismatch here would trade mainnet orders
    against testnet prices — refusing up-front is the fail-closed guard.
    """
    if not settings.bybit_testnet:
        raise RuntimeError(
            "run_scalping_session is TESTNET-ONLY. Refusing to start with "
            "BYBIT_TESTNET=false. Use the comprehensive live orchestrator for "
            "promoted mainnet trading, not this scalping runner."
        )
    return BybitBroker(
        api_key=settings.bybit_api_key,
        api_secret=settings.bybit_api_secret,
        testnet=True,
        tld=settings.bybit_tld,
    )


class OrderFlowPoller:
    """Phase 4b: poll each symbol's order book into a cache for the strategy.

    Runs as an async background task, fetching ``broker.get_orderbook`` per
    symbol every ``interval`` seconds and stashing the latest
    :class:`~kairon.live.orderflow.OrderFlowSnapshot`. The trading loop reads the
    cache via the sync :meth:`snapshot` closure (passed as ``orderflow_provider``)
    so the hot loop never blocks on a network call. A fetch failure leaves the
    previous snapshot in place (stale book beats no book); a missing/empty book
    stores ``None`` so the strategy treats it as "no order-flow signal".
    """

    def __init__(
        self, broker: BybitBroker, symbols: list[str], interval: float, logger: logging.Logger,
    ) -> None:
        self._broker = broker
        self._symbols = symbols
        self._interval = interval
        self._logger = logger
        self._cache: dict[str, object] = dict.fromkeys(symbols)
        self._task: asyncio.Task | None = None

    def snapshot(self, symbol: str) -> object:
        """Sync cache read for the loop's ``orderflow_provider``."""
        return self._cache.get(symbol)

    async def _poll_once(self) -> None:
        for sym in self._symbols:
            try:
                ob = await self._broker.get_orderbook(sym, depth=5)
            except Exception as e:
                self._logger.debug("orderflow poll failed for %s: %s", sym, e)
                continue
            if ob is None:
                continue
            self._cache[sym] = compute_orderflow(ob["bids"], ob["asks"], depth=5)

    async def run(self) -> None:
        try:
            while True:
                await self._poll_once()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self.run(), name="orderflow")
        return self._task

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()


def _build_orderflow(
    args: argparse.Namespace, broker: BybitBroker, logger: logging.Logger,
) -> tuple[OrderFlowPoller | None, object | None]:
    """Build the Phase 4b order-flow poller + provider closure (opt-in).

    Returns ``(None, None)`` when ``--orderflow`` is off, so the legacy path is
    byte-for-byte unchanged. Extracted from ``_run_session`` to keep that
    function under the branch-count limit.
    """
    if not args.orderflow:
        return None, None
    poller = OrderFlowPoller(
        broker=broker, symbols=args.symbols, interval=float(args.orderflow_interval),
        logger=logger,
    )
    return poller, poller.snapshot


async def _shutdown_session(
    *,
    loop: CooledTradingLoop,
    orderflow_poller: OrderFlowPoller | None,
    broker: BybitBroker,
    feed: MultiSymbolPollingFeed,
    store: LiveStore,
    args: argparse.Namespace,
    logger: logging.Logger,
    start_ms: int,
) -> tuple[float, list[dict]]:
    """Tear down the loop, flatten, close resources, and collect final balances.

    Returns ``(final_balance, closed_pnl)``. Extracted from ``_run_session`` to
    keep that function under the branch-count limit; every step is fail-soft so
    one teardown failure does not lose the rest (final balance, closed PnL).
    """
    await loop.stop()
    if orderflow_poller is not None:
        orderflow_poller.cancel()
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
    return final_balance, closed_pnl


async def _run_session(args: argparse.Namespace, ts: str) -> dict:  # noqa: PLR0915
    repo_root = REPO_ROOT
    _load_env(repo_root)
    # Avoid pathlib `/` and mkdir inside the async function (ASYNC240); build
    # paths with os.path.join and let LiveStore create the parent dir.
    log_path = Path(os.path.join(str(repo_root), "logs", f"scalping_{ts}.log"))
    logger = _setup_logging(log_path)
    logger.info("=" * 64)
    logger.info("Kairon SCALPING session @ %s", ts)
    logger.info(
        "Symbols=%s  duration=%ds  bankroll_start=%.2f leverage=%.1f stop=%.2f "
        "risk/trade=%.3f rr=%.2f maxdd=%.2f cooldown=%.0fs poll=%.1fs cadence=%ds short_only=%s",
        args.symbols, args.duration, args.bankroll_start, args.leverage, args.bankroll_stop,
        args.risk_per_trade, args.rr_ratio, args.max_drawdown, args.cooldown,
        args.poll_interval, args.cadence, args.short_only,
    )
    logger.info("=" * 64)

    settings = KaironSettings()
    db_path = Path(os.path.join(str(repo_root), "data", f"scalping_{ts}.db"))
    store = LiveStore(db_path)
    store.unhalt()

    broker = _build_scalping_broker(settings)

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

    feed = MultiSymbolPollingFeed(
        sym_objs, timeframe=args.timeframe, poll_interval_seconds=args.poll_interval,
    )

    live_config = LiveConfig(
        symbols=tuple(args.symbols),
        timeframe=args.timeframe,
        cadence_seconds=args.cadence,
        max_daily_loss_pct=settings.live_max_daily_loss_pct,
        max_open_positions=len(args.symbols),
        warmup_bars=settings.live_warmup_bars,
        reconcile_interval_seconds=max(5, args.cadence),
        reconcile_grace_seconds=max(2 * args.cadence, 20),
        dry_run=False,
        bybit_testnet=settings.bybit_testnet,
        bybit_tld=settings.bybit_tld,
        strategy_name="scalping",
    )

    bankroll_cfg = BankrollConfig(
        start=args.bankroll_start,
        leverage=args.leverage,
        allocation=args.allocation,
        stop_at=args.bankroll_stop,
        milestones=(50.0, 100.0),
        risk_per_trade=args.risk_per_trade,
        rr_ratio=args.rr_ratio,
        max_drawdown=args.max_drawdown,
        enforce_risk_cap=not args.no_enforce_risk_cap,
        allow_min_lot_overshoot=args.allow_min_lot_overshoot,
        risk_cap_tol=args.risk_cap_tol,
    )

    guardian = Guardian(
        max_position_equity_fraction=1.0,
        max_total_leverage=args.leverage,
        max_open_positions=len(args.symbols),
        max_daily_loss_pct=settings.live_max_daily_loss_pct,
        # Per-symbol SL cooldown: shorter than the 4h default for scalping.
        cooldown_seconds=args.cooldown * 4,
        store=store,
    )
    wrapped_broker = CooldownBrokerWrapper(broker, cooldown_seconds=args.cooldown)
    reconciler = Reconciler(
        drift_tolerance_pct=0.05,
        grace_seconds=live_config.reconcile_grace_seconds,
        reconcile_interval_seconds=live_config.reconcile_interval_seconds,
        symbols=tuple(args.symbols),
        store=store,
        broker=wrapped_broker,
    )
    strategy = ScalpingStrategy(
        short_only=args.short_only,
        rr_ratio=args.rr_ratio,
        setup_matrix=_resolve_setup_matrix(args.setup_matrix),
        use_orderflow=args.orderflow,
    )
    # Live risk-cap preflight: classify each symbol with real testnet prices so
    # the user sees which symbols will trade vs skip-before-startup at this
    # bankroll, and at what stake each unlocks. Uses the strategy's max_sl_pct as
    # the worst-case stop distance (sl_distance <= close * max_sl_pct always).
    await _risk_cap_preflight_live(
        broker, args.symbols, bankroll_cfg,
        sl_distance_pct=float(strategy.max_sl_pct), logger=logger,
    )
    # Drift kill-switch (Phase 3): the setup matrix was data-discovered on the
    # 8-week testnet store, so overfitting is the dominant live risk. This
    # rolling win-rate/expectancy monitor halts the loop on fresh bars if the
    # edge evaporates. Disabled with --no-drift-killswitch.
    drift_killswitch = (
        DriftKillSwitch(DriftKillSwitchConfig()) if args.drift_killswitch else None
    )

    # Phase 4b: order-flow poller (opt-in via --orderflow). Polls each symbol's
    # book into a cache; the loop reads it via the sync ``snapshot`` closure so
    # the hot loop never blocks on a network call. Off by default -> provider
    # is None and the legacy path is byte-for-byte unchanged.
    orderflow_poller, orderflow_provider = _build_orderflow(args, broker, logger)

    loop = CooledTradingLoop(
        config=live_config,
        broker=wrapped_broker,
        strategy=strategy,
        guardian=guardian,
        reconciler=reconciler,
        store=store,
        feed=feed,
        bankroll=bankroll_cfg,
        attach_stops=True,
        drift_killswitch=drift_killswitch,
        orderflow_provider=orderflow_provider,
    )

    # Pre-seed each symbol's bar buffer with recent history so the strategy is
    # warm at start AND only acts on freshly-closed LIVE bars (history is seeded
    # into the buffer, not replayed through the order path). This keeps attached
    # TP/SL on the correct side of the current market — the stale-history replay
    # that previously made Bybit reject TP/SL on the wrong side.
    await _prewarm_buffers(loop, sym_objs, logger, timeframe=args.timeframe)

    shutdown_event = asyncio.Event()

    def _on_signal(sig: int) -> None:
        logger.info("Received signal %s, shutting down scalping session...", sig)
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
    if orderflow_poller is not None:
        of_task = orderflow_poller.start()
        background.add(of_task)
        of_task.add_done_callback(background.discard)
        logger.info("Order-flow poller started (interval=%.1fs)", float(args.orderflow_interval))

    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=args.duration)
    except TimeoutError:
        logger.info("Session duration reached (%ds); shutting down.", args.duration)
    except asyncio.CancelledError:
        logger.info("Session cancelled.")
    finally:
        final_balance, closed_pnl = await _shutdown_session(
            loop=loop, orderflow_poller=orderflow_poller, broker=broker, feed=feed,
            store=store, args=args, logger=logger, start_ms=start_ms,
        )

    # Recompute analytics + growth summary from the DB.
    store2 = LiveStore(db_path)
    report = compute_session_report(store2, timeframe=args.timeframe, session_id=f"scalping-{ts}")
    ledger = store2.get_ledger()
    store2.close()

    growth = _growth_summary(ledger, bankroll_cfg)
    extras = _scalping_extras(db_path)
    report_data = {
        "session_id": f"scalping-{ts}",
        "symbols": args.symbols,
        "duration_seconds": args.duration,
        "bankroll_start": bankroll_cfg.start,
        "bankroll_end": growth["bankroll_end"],
        "bankroll_peak": growth["bankroll_peak"],
        "bankroll_drawdown": growth["bankroll_drawdown"],
        "milestones_hit": growth["milestones_hit"],
        "risk_per_trade": args.risk_per_trade,
        "rr_ratio": args.rr_ratio,
        "max_drawdown": args.max_drawdown,
        "cooldown_seconds": args.cooldown,
        "poll_interval": args.poll_interval,
        "cadence_seconds": args.cadence,
        "short_only": args.short_only,
        "initial_balance_real": initial_balance,
        "final_balance_real": final_balance,
        "real_pnl": final_balance - initial_balance,
        "closed_pnl_count": len(closed_pnl),
        "closed_pnl_realized": sum(float(t.get("closedPnl", 0) or 0) for t in closed_pnl),
        "scalping_extras": extras,
        "ledger": ledger,
        "report": format_report(report),
    }
    report_json = Path(os.path.join(str(repo_root), "logs", f"scalping_{ts}.report.json"))
    _write_json_file(report_json, report_data)
    logger.info("Scalping JSON report saved to %s", report_json)

    _write_markdown_report(
        repo_root, ts, args, report_data, report, growth, initial_balance, final_balance, extras
    )
    print("\n" + "=" * 64)
    print("SCALPING SESSION SUMMARY")
    print("=" * 64)
    print(f"Bankroll: {bankroll_cfg.start:.2f} -> {growth['bankroll_end']:.2f} USDT "
          f"(peak {growth['bankroll_peak']:.2f}, max drawdown {growth['bankroll_drawdown']:.1%})")
    print(f"Trades: {extras['n_trades']}  (TP:{extras['n_tp']} SL:{extras['n_sl']} "
          f"manual:{extras['n_manual']})  wins:{extras['wins']} losses:{extras['losses']}")
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
        peak = max(peak, bankroll)
    drawdown = (peak - bankroll) / peak if peak > 0 else 0.0
    return {
        "bankroll_end": bankroll,
        "bankroll_peak": peak,
        "bankroll_drawdown": drawdown,
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
    extras: dict[str, Any],
) -> None:
    lines = [
        f"# Kairon Scalping Session Report — {ts}",
        "",
        f"- Symbols: {', '.join(args.symbols)}",
        f"- Timeframe: {args.timeframe}",
        f"- Duration: {args.duration // 60} minutes ({args.duration}s)",
        f"- Strategy: ScalpingStrategy (short_only={args.short_only})",
        f"- Setup matrix: {args.setup_matrix}",
        f"- Drift kill-switch: {'on' if args.drift_killswitch else 'off'}",
        f"- Order flow: {'on' if args.orderflow else 'off'}"
        + (f" (interval={args.orderflow_interval:.0f}s)" if args.orderflow else ""),
        f"- Bankroll model: synthetic, start={args.bankroll_start} USDT, "
        f"leverage={args.leverage}x, allocation={args.allocation}, stop_at={args.bankroll_stop}",
        f"- Risk: risk_per_trade={args.risk_per_trade}, rr_ratio={args.rr_ratio}, "
        f"max_drawdown={args.max_drawdown}, cooldown={args.cooldown}s, "
        f"poll={args.poll_interval}s, cadence={args.cadence}s",
        "",
        "## Growth (synthetic bankroll)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Start bankroll | {args.bankroll_start:.2f} USDT |",
        f"| End bankroll | {growth['bankroll_end']:.2f} USDT |",
        f"| Peak bankroll | {growth['bankroll_peak']:.2f} USDT |",
        f"| Max drawdown | {growth['bankroll_drawdown']:.1%} |",
        f"| Growth | {growth['bankroll_end'] - args.bankroll_start:+.2f} USDT "
        f"({(growth['bankroll_end'] / args.bankroll_start - 1) * 100:+.1f}%) |",
        f"| Milestones hit | {growth['milestones_hit']} |",
        "",
        "## Scalping execution",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Round-trip trades | {extras['n_trades']} |",
        f"| TP hits (hit_tp) | {extras['n_tp']} |",
        f"| SL hits (hit_sl) | {extras['n_sl']} |",
        f"| Manual closes | {extras['n_manual']} |",
        f"| Wins / Losses | {extras['wins']} / {extras['losses']} |",
        f"| Avg win | {extras['avg_win']:+.4f} USDT |",
        f"| Avg loss | {extras['avg_loss']:+.4f} USDT |",
        f"| Realized total (closed trades) | {extras['realized_total']:+.4f} USDT |",
        "",
        "## Per-setup edge (live, out-of-sample)",
        "",
        "| setup | n | win% | TP | SL | sumPnL |",
        "|---|---|---|---|---|---|",
    ]
    if extras.get("per_setup"):
        for row in extras["per_setup"]:
            lines.append(
                f"| {row['setup_id']} | {row['n']} | {row['win_rate']*100:.0f} | "
                f"{row['tp']} | {row['sl']} | {row['sum_pnl']:+.4f} |"
            )
    else:
        lines.append("| _no closed setups_ | - | - | - | - | - |")
    lines += [
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
    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "- **Stop-level exit approximation:** when the exchange's attached SL/TP "
        "closes a position but the WS close fill does not drain (a testnet race), "
        "the orchestrator reconciles the local mirror to flat at the *stop level* "
        "(the price where the attached SL/TP sat), not the crossing price. This "
        "matches the exchange fill and avoids overstating the realized loss. "
        "Tagged `software_stop_reconcile` in `live_events`."
    )
    lines.append(
        "- **Orphan stop cleanup:** every close path (software stop, reconcile, "
        "TP/SL fill, manual flatten) calls `cancel_all(symbol)` in attach_stops "
        "mode so a stale attached TP/SL conditional cannot fire on a future move "
        "with no position to close."
    )
    lines.append(
        "- **Risk-cap guard:** after any quantity rounding, the sizer recomputes "
        "the implied risk and skips a trade whose implied risk would exceed "
        "`risk_per_trade * (1 + risk_cap_tol)`; sub-min-lot signals skip unless "
        "`--allow-min-lot-overshoot` is set and the overshoot stays within tol."
    )
    lines.append("")
    lines.append("## Session analytics")
    lines.append("")
    lines.append("```")
    lines.append(format_report(report))
    lines.append("```")
    lines.append("")
    path = repo_root / "reports" / f"scalping_{ts}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Scalping markdown report written to: {path}")


def _resolve_setup_matrix(name: str) -> SetupMatrix | None:
    """Map the ``--setup-matrix`` choice to a :class:`SetupMatrix` (or None).

    ``mean-reversion`` -> ``MEAN_REVERSION_ONLY`` (data-discovered: keep
    mr_short/mr_long, kill momentum + breakout/breakdown, gate to ranges,
    exhaustion + MTF + confidence calibration). ``long-only`` -> ``LONG_ONLY``
    (Phase 4: mr_long only — kills the universal mr_short loser found in the
    universe backtest; the honest win-rate lever via selectivity, not floor
    tightening). ``legacy`` -> ``None`` (original ungated behaviour). ``all`` ->
    default ``SetupMatrix()`` (every setup enabled, no regime gate).
    """
    if name == "mean-reversion":
        return MEAN_REVERSION_ONLY
    if name == "long-only":
        return LONG_ONLY
    if name == "all":
        return SetupMatrix()
    return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_scalping_session", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION_SECONDS, help="Seconds to run")
    p.add_argument("--bankroll-start", type=float, default=10.0)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--allocation", type=float, default=1.0)
    p.add_argument(
        "--bankroll-stop", type=float, default=100.0, help="Bankroll level that halts the loop"
    )
    p.add_argument("--risk-per-trade", type=float, default=DEFAULT_RISK_PER_TRADE,
                   help="Fraction of bankroll risked per trade (fixed-fractional sizing)")
    p.add_argument("--rr-ratio", type=float, default=DEFAULT_RR_RATIO,
                   help="Reward:risk ratio for take-profit")
    p.add_argument("--max-drawdown", type=float, default=DEFAULT_MAX_DRAWDOWN,
                   help="Bankroll peak-to-trough drawdown fraction that halts the loop")
    p.add_argument("--allow-min-lot-overshoot", action="store_true",
                   help="Bump a sub-min-lot risk-sized qty UP to the min lot when the "
                        "implied risk still respects risk_per_trade*(1+tol). Default OFF: "
                        "sub-min-lot signals skip (risk cap exactly bounded).")
    p.add_argument("--risk-cap-tol", type=float, default=DEFAULT_RISK_CAP_TOL,
                   help="Tolerance on the risk cap for lot rounding (skip when "
                        "implied_risk > risk_per_trade*(1+tol)). 0 = hard cap.")
    p.add_argument("--no-enforce-risk-cap", action="store_true",
                   help="Disable the post-rounding risk-cap guard (NOT recommended; "
                        "the cap can then be breached by min-lot overshoot / confidence "
                        "scaling). Default: enforce.")
    p.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_SECONDS,
                   help="Post-order cooldown seconds per symbol")
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME,
                   help="Live bar timeframe (CCXT string, e.g. 1m/5m/15m). Default 5m "
                        "— the setup-selection matrix was validated on 5m/15m; 1m is "
                        "noisier and outside the validated edge.")
    p.add_argument("--poll-interval", type=float, default=None,
                   help="Feed poll interval seconds. Default: timeframe-aware (~tf/4, "
                        "floored at 15s) so each bar is sampled a few times without excess "
                        "API churn.")
    p.add_argument("--cadence", type=int, default=DEFAULT_CADENCE_SECONDS,
                   help="Loop cadence seconds (>=10)")
    p.add_argument("--short-only", action="store_true",
                   help="Suppress long setups (pure short scalping; risky in uptrends)")
    p.add_argument("--setup-matrix", default="mean-reversion",
                   choices=["mean-reversion", "long-only", "legacy", "all"],
                   help="Setup-selection matrix: 'mean-reversion' (default, data-discovered "
                        "MR-only + regime gate + exhaustion + MTF + calibration), 'long-only' "
                        "(Phase 4: mr_long only — kills the universal mr_short loser, the "
                        "win-rate lever via selectivity), 'legacy' (no matrix, original "
                        "behaviour), 'all' (all setups enabled, no gate)")
    p.add_argument("--drift-killswitch", dest="drift_killswitch",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Halt the loop when live win-rate/expectancy drifts below the "
                        "research edge (out-of-sample guardrail). Use --no-drift-killswitch "
                        "to disable.")
    p.add_argument("--orderflow", action="store_true",
                   help="Phase 4b: poll the order book per symbol and nudge confidence "
                        "toward the side the book leans (bid-heavy supports a long bounce, "
                        "ask-heavy supports a short fade). Entry-timing axis (cannot be "
                        "backtested on the OHLCV store — no historical L2), so it ships "
                        "opt-in/off-by-default; the drift kill-switch is the guardrail.")
    p.add_argument("--orderflow-interval", type=float, default=15.0,
                   help="Order-book poll interval seconds (default 15s). Only used with "
                        "--orderflow.")
    p.add_argument("--dry-run", action="store_true", help="Print planned config and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Timeframe-aware default poll interval (~tf/4, floored at 15s) so each bar
    # is sampled a few times without excess API churn. Explicit --poll-interval
    # overrides.
    if args.poll_interval is None:
        args.poll_interval = _default_poll_for(args.timeframe)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    print("Kairon SCALPING session plan")
    print(f"  symbols        : {args.symbols}")
    print(f"  timeframe      : {args.timeframe}")
    print(f"  duration       : {args.duration}s ({args.duration // 60} min)")
    print(f"  bankroll start : {args.bankroll_start} USDT")
    print(f"  leverage       : {args.leverage}x")
    print(f"  allocation     : {args.allocation}")
    print(f"  stop_at        : {args.bankroll_stop} USDT")
    risk_pct = args.risk_per_trade * 100
    print(f"  risk per trade : {args.risk_per_trade} ({risk_pct:.1f}% of bankroll)")
    print(f"  rr ratio       : {args.rr_ratio}")
    print(f"  max drawdown   : {args.max_drawdown} ({args.max_drawdown * 100:.0f}%)")
    print(f"  cooldown       : {args.cooldown}s (SL cooldown {args.cooldown * 4:.0f}s)")
    print(f"  poll interval  : {args.poll_interval}s")
    print(f"  cadence        : {args.cadence}s")
    print(f"  short only     : {args.short_only}")
    print(f"  setup matrix   : {args.setup_matrix}")
    print(f"  drift killswitch: {args.drift_killswitch}")
    print(f"  order flow     : {args.orderflow} (interval={args.orderflow_interval}s)")
    print("  strategy       : ScalpingStrategy")
    print("  attach stops   : True (native ATR-based TP/SL)")
    print(f"  enforce risk cap: {not args.no_enforce_risk_cap} "
          f"(tol={args.risk_cap_tol}, allow_min_lot_overshoot={args.allow_min_lot_overshoot})")
    print(f"  session id     : scalping-{ts}")
    # Dry-run risk-cap preflight with representative testnet prices (the live
    # preflight at session start uses real prices + the broker's actual min lot).
    dry_rows = _classify_symbols(
        args.symbols,
        bankroll=args.bankroll_start,
        risk_per_trade=args.risk_per_trade,
        leverage=args.leverage,
        allocation=args.allocation,
        sl_distance_pct=ScalpingStrategy().max_sl_pct,
        prices=_DRYRUN_PRICES,
        min_qtys=_DRYRUN_MIN_QTY,
    )
    print()
    print(_format_preflight(dry_rows))
    if args.dry_run:
        print("\nDry run: no session launched.")
        return 0
    try:
        asyncio.run(_run_session(args, ts))
    except Exception as e:
        print(f"ERROR: session failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
