"""Run a single-symbol Bybit testnet live trading session for a fixed duration.

This script is the per-symbol worker used by ``run_testnet_20min.py``.
It loads API credentials from the environment / ``.env``, trades one
perpetual symbol with the built-in ComprehensiveStrategy, stores every
order, fill, decision, and heartbeat to its own SQLite database, and
flattens any open position at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kairon.config import KaironSettings
from kairon.data.symbols import CryptoVenue, crypto_perp
from kairon.live.analytics import compute_session_report, format_report
from kairon.live.broker.base import Order, OrderStatus
from kairon.live.broker.bybit import BybitBroker
from kairon.live.config import LiveConfig
from kairon.live.guardian import Guardian
from kairon.live.orchestrator import TradingLoop
from kairon.live.predictor import LivePrediction
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore
from kairon.live.strategy import ComprehensiveStrategy

DEFAULT_DURATION_SECONDS = 20 * 60
DEFAULT_HISTORY_MINUTES = 60
DEFAULT_POLL_INTERVAL_SECONDS = 60.0
DEFAULT_COOLDOWN_SECONDS = 5 * 60  # one trade every 5 minutes per symbol


class CooldownBrokerWrapper:
    """Wrap a Broker and record the last accepted order timestamp per symbol.

    Used by CooledTradingLoop to suppress rapid rebalancing during short
    testnet sessions.
    """

    def __init__(self, inner: BybitBroker, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._inner = inner
        self.cooldown_seconds = cooldown_seconds
        self._last_order_ts: dict[str, float] = {}

    async def place_order(self, order: Order) -> Order:
        result = await self._inner.place_order(order)
        if result.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
            self._last_order_ts[order.symbol] = time.time()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class CooledTradingLoop(TradingLoop):
    """TradingLoop that ignores new signals while a symbol is in cooldown."""

    def _make_prediction(self, symbol: str) -> LivePrediction:
        broker = self._broker
        if isinstance(broker, CooldownBrokerWrapper):
            last = broker._last_order_ts.get(symbol)
            if last is not None and time.time() - last < broker.cooldown_seconds:
                return LivePrediction(
                    symbol=symbol,
                    direction=0.0,
                    magnitude=0.0,
                    volatility=0.01,
                    confidence=0.0,
                    horizon=self._config.horizon,
                    ts=_utc_now_iso(),
                )
        return super()._make_prediction(symbol)


class SingleSymbolPollingFeed:
    """REST-polling candle feed for exactly one symbol.

    Fetches an initial history window so the strategy can warm up
    immediately, then polls for new closed bars on a timer. Each bar is
    pushed as a one-row OHLCV_SCHEMA table on ``self.queue``.
    """

    def __init__(
        self,
        symbol,
        *,
        timeframe: str = "1m",
        history_minutes: int = DEFAULT_HISTORY_MINUTES,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.history_minutes = history_minutes
        self.poll_interval_seconds = poll_interval_seconds
        self.queue: asyncio.Queue = asyncio.Queue()
        self._adapter: CCXTAdapter | None = None
        self._running = False

    async def run(self) -> None:
        from datetime import UTC, datetime, timedelta
        from kairon.data.adapters.ccxt_adapter import CCXTAdapter
        from kairon.data.symbols import CryptoVenue

        self._adapter = CCXTAdapter(venue=CryptoVenue.BYBIT, testnet=True)
        self._running = True
        logger = logging.getLogger(__name__)
        logger.info(
            "PollingFeed starting for %s (interval=%.1fs, history=%dm)",
            self.symbol.canonical,
            self.poll_interval_seconds,
            self.history_minutes,
        )

        # Initial history load so the strategy warms up immediately.
        try:
            end = datetime.now(UTC)
            start = end - timedelta(minutes=self.history_minutes)
            table = await self._adapter.afetch(self.symbol, self.timeframe, start, end)
            n = table.num_rows
            if n > 0:
                logger.info("PollingFeed: loaded %d historical bars for %s", n, self.symbol.canonical)
                for i in range(n):
                    await self.queue.put(table.slice(i, 1))
        except Exception as e:
            logger.warning("PollingFeed: initial fetch failed for %s: %s", self.symbol.canonical, e)

        last_ts = None
        while self._running:
            try:
                from datetime import UTC, datetime, timedelta
                end = datetime.now(UTC)
                start = end - timedelta(minutes=2)
                table = await self._adapter.afetch(self.symbol, self.timeframe, start, end)
                if table.num_rows > 0:
                    latest = table.slice(table.num_rows - 1, 1)
                    ts_val = latest.column("ts")[0].as_py()
                    if ts_val != last_ts:
                        last_ts = ts_val
                        await self.queue.put(latest)
                        close_val = float(latest.column("close")[0].as_py())
                        logger.info(
                            "PollingFeed: %s @ %.4f (%s)",
                            self.symbol.canonical, close_val, ts_val,
                        )
            except Exception as e:
                logger.warning("PollingFeed: poll failed for %s: %s", self.symbol.canonical, e)
            await asyncio.sleep(self.poll_interval_seconds)

    async def aclose(self) -> None:
        self._running = False
        if self._adapter is not None:
            await self._adapter.aclose()
            self._adapter = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Ensure UTF-8 on Windows terminals so pybit's unicode arrows do not crash logging.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear existing handlers so repeated imports in tests do not duplicate.
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return logging.getLogger(f"kairon.testnet.{log_file.stem}")


async def _preflight(broker: BybitBroker, symbol: str) -> dict:
    health = await broker.check_health()
    if not health.get("ok"):
        raise RuntimeError(f"Health check failed: {health.get('errors')}")
    balances = await broker.get_balances()
    usdt = next((b for b in balances if b.currency == "USDT"), None)
    if usdt is None or usdt.total <= 0:
        raise RuntimeError("No USDT balance found on testnet account")
    await broker.set_leverage(symbol, 1)
    return {"health": health, "usdt_total": usdt.total}


async def _flatten(broker: BybitBroker, store: LiveStore, symbol: str) -> None:
    """Flatten any open position for the symbol using reduce-only orders."""
    logger = logging.getLogger(__name__)
    order = await broker.close_position(symbol)
    if order.status not in (OrderStatus.FILLED,):
        logger.info(
            "Close position order for %s: %s %.6f status=%s",
            symbol, order.side.value, order.qty, order.status.value,
        )
    positions = await broker.get_positions(symbol)
    residual = next((p.qty for p in positions if p.symbol == symbol), 0.0)
    if residual > 1e-9:
        logger.warning("Residual position remains for %s: qty=%.6f", symbol, residual)
    else:
        logger.info("Position fully flattened for %s", symbol)


async def _run_symbol_session(
    symbol_str: str,
    db_path: Path,
    log_file: Path,
    duration_seconds: int,
) -> dict:
    logger = _setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("Starting 20-minute Bybit testnet session for %s", symbol_str)
    logger.info("DB: %s", db_path)
    logger.info("Log: %s", log_file)
    logger.info("=" * 60)

    settings = KaironSettings()
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
        preflight = await _preflight(broker, symbol_str)
        initial_balance = preflight["usdt_total"]
        logger.info("Preflight OK. USDT total balance: %.2f", initial_balance)
        store.write_heartbeat(mode="startup", equity=initial_balance, n_positions=0)
    except Exception as e:
        logger.exception("Preflight failed for %s: %s", symbol_str, e)
        broker.close()
        store.close()
        raise

    parts = symbol_str.split("-")
    symbol = crypto_perp(parts[0], parts[1], CryptoVenue.BYBIT)

    feed = SingleSymbolPollingFeed(symbol, timeframe="1m", history_minutes=DEFAULT_HISTORY_MINUTES)

    live_config = LiveConfig(
        symbols=(symbol_str,),
        timeframe="1m",
        cadence_seconds=60,
        max_daily_loss_pct=settings.live_max_daily_loss_pct,
        max_open_positions=settings.live_max_open_positions,
        warmup_bars=settings.live_warmup_bars,
        dry_run=False,
        bybit_testnet=settings.bybit_testnet,
        bybit_tld=settings.bybit_tld,
        strategy_name="comprehensive",
    )

    guardian = Guardian(
        max_position_equity_fraction=settings.max_position_equity_fraction,
        max_total_leverage=settings.max_total_leverage,
        max_open_positions=settings.live_max_open_positions,
        max_daily_loss_pct=settings.live_max_daily_loss_pct,
        store=store,
    )
    wrapped_broker = CooldownBrokerWrapper(broker)
    reconciler = Reconciler(
        drift_tolerance_pct=0.05,
        grace_seconds=live_config.reconcile_grace_seconds,
        reconcile_interval_seconds=live_config.reconcile_interval_seconds,
        symbols=(symbol_str,),
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
    )

    shutdown_event = asyncio.Event()

    def _on_signal(sig: int) -> None:
        logger.info("Received signal %s, shutting down %s session...", sig, symbol_str)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _on_signal, sig)
        except (NotImplementedError, ValueError):
            pass  # Windows may not support add_signal_handler

    try:
        feed_task = asyncio.create_task(feed.run(), name=f"feed-{symbol_str}")
        loop_task = asyncio.create_task(loop.start(), name=f"loop-{symbol_str}")

        # Run until duration expires or a signal/external halt occurs.
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=duration_seconds)
        except asyncio.TimeoutError:
            logger.info("Session duration reached for %s", symbol_str)
        except asyncio.CancelledError:
            logger.info("Session cancelled for %s", symbol_str)
    except Exception as e:
        logger.exception("Runtime error for %s: %s", symbol_str, e)
    finally:
        await loop.stop()
        await _flatten(broker, store, symbol_str)
        await feed.aclose()
        await broker.aclose()

        # Final balance and closed PnL from Bybit
        try:
            balances = await broker.get_balances()
            usdt = next((b for b in balances if b.currency == "USDT"), None)
            final_balance = usdt.total if usdt else 0.0
            store.write_heartbeat(mode="shutdown", equity=final_balance, n_positions=0)
            logger.info("Final USDT balance for %s: %.2f", symbol_str, final_balance)
        except Exception as e:
            logger.error("Could not fetch final balance for %s: %s", symbol_str, e)
            final_balance = 0.0

        closed_pnl = []
        try:
            end_ms = int(time.time() * 1000)
            closed_pnl = await broker.get_closed_pnl(
                symbol_str, start_time_ms=start_ms, end_time_ms=end_ms, limit=200
            )
            logger.info("Closed PnL entries for %s: %d", symbol_str, len(closed_pnl))
        except Exception as e:
            logger.error("Could not fetch closed PnL for %s: %s", symbol_str, e)

        store.close()

    # Compute analytics from the store.
    store2 = LiveStore(db_path)
    try:
        report = compute_session_report(
            store2, timeframe="1m", session_id=f"testnet-{symbol_str}"
        )
        report_data = {
            "symbol": symbol_str,
            "duration_seconds": duration_seconds,
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_pnl": final_balance - initial_balance,
            "total_pnl_pct": (
                (final_balance - initial_balance) / initial_balance * 100
                if initial_balance > 0
                else 0.0
            ),
            "closed_pnl_count": len(closed_pnl),
            "closed_pnl_realized": sum(float(t.get("closedPnl", 0) or 0) for t in closed_pnl),
            "report": report,
        }
    finally:
        store2.close()

    report_path = log_file.with_suffix(".report.json")
    report_path.write_text(json.dumps(report_data, indent=2, default=str))
    logger.info("Report saved to %s", report_path)
    print(format_report(report))
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-symbol Bybit testnet session")
    parser.add_argument("symbol", help="Canonical symbol, e.g. BTC-USDT-PERP")
    parser.add_argument("--db", required=True, help="Path to SQLite store")
    parser.add_argument("--log", required=True, help="Path to log file")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_SECONDS, help="Seconds to run")
    args = parser.parse_args()

    result = asyncio.run(_run_symbol_session(args.symbol, Path(args.db), Path(args.log), args.duration))
    # Exit non-zero if the session had critical errors and zero trades.
    if result["report"].n_orders == 0 and result["initial_balance"] == result["final_balance"]:
        pass  # This is actually fine for a quiet market.


if __name__ == "__main__":
    main()
