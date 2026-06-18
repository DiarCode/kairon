"""Run a 1-hour dry-run trading session with REST-polling candle feed.

This script uses the CCXTAdapter's REST afetch method to poll for
candles on a timer, rather than relying on WebSocket streaming which
requires ccxt.pro. This is more reliable for a demo session.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from kairon.config import KaironSettings
from kairon.data.symbols import crypto_perp, CryptoVenue
from kairon.data.adapters.ccxt_adapter import CCXTAdapter
from kairon.live.broker.paper import PaperBroker
from kairon.live.config import LiveConfig
from kairon.live.guardian import Guardian
from kairon.live.orchestrator import TradingLoop
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore
from kairon.live.strategy import MACrossoverStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("kairon.session")


class PollingFeed:
    """Simple polling feed that fetches candles via REST at a fixed interval."""

    def __init__(self, symbols, timeframe: str = "1m", interval: float = 60.0):
        self.symbols = symbols
        self.timeframe = timeframe
        self.interval = interval
        self.queue: asyncio.Queue = asyncio.Queue()
        self._adapter: CCXTAdapter | None = None
        self._running = False

    async def run(self) -> None:
        """Poll for new candles and push them to the queue."""
        self._adapter = CCXTAdapter(venue=CryptoVenue.BYBIT)
        self._running = True
        logger.info("PollingFeed: starting (interval=%.1fs, symbols=%s)",
                     self.interval, [s.canonical for s in self.symbols])

        # Fetch initial history (warmup bars)
        for symbol in self.symbols:
            try:
                end = datetime.now(UTC)
                start = end - timedelta(minutes=30)  # 30 min of history
                table = await self._adapter.afetch(symbol, self.timeframe, start, end)
                if table.num_rows > 0:
                    logger.info("PollingFeed: loaded %d historical bars for %s",
                                table.num_rows, symbol.canonical)
                    # Push individual bars
                    for i in range(table.num_rows):
                        row = table.slice(i, 1)
                        await self.queue.put(row)
            except Exception as e:
                logger.warning("PollingFeed: initial fetch failed for %s: %s",
                               symbol.canonical, e)

        # Poll for new candles
        last_ts: dict = {}
        while self._running:
            for symbol in self.symbols:
                try:
                    end = datetime.now(UTC)
                    start = end - timedelta(minutes=2)
                    table = await self._adapter.afetch(symbol, self.timeframe, start, end)
                    if table.num_rows > 0:
                        # Only push the most recent bar if it's new
                        latest = table.slice(table.num_rows - 1, 1)
                        ts_val = latest.column("ts")[0].as_py()
                        if ts_val != last_ts.get(symbol.canonical):
                            last_ts[symbol.canonical] = ts_val
                            await self.queue.put(latest)
                            close_val = float(latest.column("close")[0].as_py())
                            logger.info("PollingFeed: %s @ %.2f (%s)",
                                        symbol.canonical, close_val, ts_val)
                except Exception as e:
                    logger.warning("PollingFeed: poll failed: %s", e)

            await asyncio.sleep(self.interval)

    async def aclose(self) -> None:
        """Stop polling and close the adapter."""
        self._running = False
        if self._adapter:
            await self._adapter.aclose()
            self._adapter = None


async def main():
    settings = KaironSettings()
    live_config = LiveConfig.from_settings(settings)

    # Setup store
    db_path = Path("data/runs.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = LiveStore(db_path)
    store.unhalt()  # Clear any previous halt

    # Paper broker with 10k USDT
    broker = PaperBroker(initial_balance=10_000.0)
    logger.info("Using PaperBroker (dry-run) with $10,000 initial equity")

    # Guardian
    guardian = Guardian(
        max_position_equity_fraction=0.20,
        max_open_positions=5,
        max_daily_loss_pct=0.03,
        store=store,
    )

    # Reconciler
    reconciler = Reconciler(
        drift_tolerance_pct=0.05,
        grace_seconds=live_config.reconcile_grace_seconds,
        reconcile_interval_seconds=live_config.reconcile_interval_seconds,
        store=store,
        broker=broker,
    )

    # Strategy
    strategy = MACrossoverStrategy()
    logger.info("MACrossoverStrategy: fast=%d, slow=%d, warmup=%d",
                strategy.fast_period, strategy.slow_period, strategy.warmup_bars)

    # Symbols
    symbols = [crypto_perp("BTC", "USDT", CryptoVenue.BYBIT)]

    # Polling feed (REST-based, more reliable than WebSocket)
    feed = PollingFeed(symbols, timeframe="1m", interval=60.0)

    # Trading loop
    loop = TradingLoop(
        config=live_config,
        broker=broker,
        strategy=strategy,
        guardian=guardian,
        reconciler=reconciler,
        store=store,
        feed=feed,
    )

    session_start = datetime.now(UTC)
    session_duration = timedelta(hours=1)
    session_end = session_start + session_duration

    logger.info("=" * 60)
    logger.info("Kairon 1-Hour Dry-Run Trading Session")
    logger.info("=" * 60)
    logger.info("Symbols:      %s", [s.canonical for s in symbols])
    logger.info("Timeframe:    %s", live_config.timeframe)
    logger.info("Warmup:       %d bars", live_config.warmup_bars)
    logger.info("Initial eq:   $10,000.00")
    logger.info("Strategy:     MACrossoverStrategy")
    logger.info("Session end:  %s", session_end.isoformat())
    logger.info("=" * 60)

    # Run feed and loop together
    feed_task = asyncio.create_task(feed.run(), name="polling-feed")
    loop_task = asyncio.create_task(loop.start(), name="trading-loop")

    # Periodic status reporter
    async def status_reporter():
        while True:
            await asyncio.sleep(120)  # Report every 2 minutes
            hb = store.get_recent_heartbeat()
            if hb:
                logger.info("STATUS: ticks=%d, equity=$%.2f, positions=%d, mode=%s",
                             loop.tick_count, hb.get("equity", 0),
                             hb.get("n_positions", 0), hb.get("mode", "?"))

    reporter_task = asyncio.create_task(status_reporter(), name="status-reporter")

    # Run for 1 hour
    try:
        done, pending = await asyncio.wait(
            [feed_task, loop_task, reporter_task],
            timeout=3600,  # 1 hour
        )
        logger.info("Session completed or timed out")
    except asyncio.CancelledError:
        logger.info("Session cancelled")
    finally:
        logger.info("Shutting down...")
        await loop.stop()
        await feed.aclose()
        reporter_task.cancel()
        store.close()

        logger.info("=" * 60)
        logger.info("SESSION COMPLETE")
        logger.info("Total ticks:     %d", loop.tick_count)
        logger.info("Session started: %s", session_start.isoformat())
        logger.info("Session ended:   %s", datetime.now(UTC).isoformat())

        # Run analytics
        store2 = LiveStore(db_path)
        from kairon.live.analytics import compute_session_report, format_report
        report = compute_session_report(store2, timeframe="1m", session_id="dry-run-1h")
        logger.info("\n" + format_report(report))
        store2.close()


if __name__ == "__main__":
    asyncio.run(main())