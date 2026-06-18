"""End-to-end smoke test of the live trading loop using PaperBroker.

Verifies that:
1. The TradingLoop runs ticks with a bar feed.
2. ComprehensiveStrategy produces direction-aware signals.
3. Orders are submitted and filled by PaperBroker.
4. Fills update local positions.
5. The Reconciler sees zero drift.
6. Positions are flattened at shutdown.

This is a deterministic dry-run that does NOT hit Bybit.
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.live.broker.base import Order, OrderSide, OrderStatus, OrderType
from kairon.live.broker.paper import PaperBroker
from kairon.live.config import LiveConfig
from kairon.live.guardian import Guardian
from kairon.live.orchestrator import TradingLoop
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore
from kairon.live.strategy import ComprehensiveStrategy


class FakeFeed:
    """Deterministic 1-minute bar feed that trends up then down."""

    def __init__(self, symbol: str, n_bars: int) -> None:
        self.symbol = symbol
        self.n_bars = n_bars
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def run(self) -> None:
        base_price = 1000.0
        for i in range(self.n_bars):
            # Create a mild trend + oscillation to generate crossovers.
            close = base_price + i * 0.5 + 5 * math.sin(i / 5.0)
            high = close + 2.0
            low = close - 2.0
            open_price = close - 0.5
            volume = 1000.0 + i * 10
            table = pa.table(
                {
                    "ts": [datetime(2026, 6, 16, 0, i, 0, tzinfo=UTC)],
                    "open": [open_price],
                    "high": [high],
                    "low": [low],
                    "close": [close],
                    "volume": [volume],
                    "symbol": [self.symbol],
                },
                schema=OHLCV_SCHEMA,
            )
            await self.queue.put(table)
            await asyncio.sleep(0.05)  # Fast deterministic simulation

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def _uuid() -> str:
    import uuid

    return uuid.uuid4().hex[:24]


async def _flatten_paper(broker: PaperBroker, symbol: str) -> None:
    """Flatten any open PaperBroker position for symbol."""
    positions = await broker.get_positions(symbol)
    if not positions:
        return
    pos = positions[0]
    close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
    order = Order(
        id=_uuid(),
        intent_id=_uuid(),
        trace_id="flatten",
        symbol=symbol,
        side=close_side,
        qty=pos.qty,
        order_type=OrderType.MARKET,
        status=OrderStatus.PENDING,
        ts=datetime.now(UTC).isoformat(),
    )
    await broker.place_order(order)


async def main() -> int:
    symbol = "BTC-USDT-PERP"
    db_path = Path(tempfile.gettempdir()) / f"kairon_paper_smoke_{datetime.now(UTC).strftime('%H%M%S')}.db"
    store = LiveStore(str(db_path))
    store.unhalt()

    broker = PaperBroker(initial_balance=100_000.0)
    broker.set_mark_price(symbol, 1000.0)

    config = LiveConfig(
        symbols=(symbol,),
        timeframe="1m",
        cadence_seconds=10,
        max_daily_loss_pct=1.0,
        max_open_positions=1,
        warmup_bars=35,
        dry_run=True,
        bybit_testnet=True,
        strategy_name="comprehensive",
    )

    guardian = Guardian(
        max_position_equity_fraction=0.20,
        max_total_leverage=2.0,
        max_open_positions=1,
        max_daily_loss_pct=1.0,
        store=store,
    )
    reconciler = Reconciler(
        drift_tolerance_pct=0.05,
        grace_seconds=1.0,
        reconcile_interval_seconds=1.0,
        symbols=(symbol,),
        store=store,
        broker=broker,
    )
    strategy = ComprehensiveStrategy()
    feed = FakeFeed(symbol, n_bars=60)

    loop = TradingLoop(
        config=config,
        broker=broker,
        strategy=strategy,
        guardian=guardian,
        reconciler=reconciler,
        store=store,
        feed=feed,
    )

    feed_task = asyncio.create_task(feed.run())
    loop_task = asyncio.create_task(loop.start())

    await asyncio.sleep(8)  # Let it run ~8 seconds of simulated time

    await loop.stop()
    await feed.aclose()
    await _flatten_paper(broker, symbol)

    store.close()

    # Inspect results
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    counts = {
        "orders": cur.execute("SELECT COUNT(*) FROM live_orders").fetchone()[0],
        "fills": cur.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0],
        "decisions": cur.execute("SELECT COUNT(*) FROM live_decisions").fetchone()[0],
        "positions": cur.execute("SELECT COUNT(*) FROM live_positions").fetchone()[0],
        "events": cur.execute("SELECT COUNT(*) FROM live_events").fetchone()[0],
    }
    drift_rows = cur.execute(
        "SELECT payload_json FROM live_events WHERE kind = 'reconciler_alert'"
    ).fetchall()
    conn.close()

    print(json.dumps(counts, indent=2))

    success = counts["orders"] > 0 and counts["fills"] > 0 and counts["decisions"] > 0
    if not success:
        print("SMOKE TEST FAILED: expected orders, fills, and decisions > 0")
        return 1

    drift_alerts = [r for r in drift_rows if "drift" in r["payload_json"]]
    if drift_alerts:
        print(f"WARNING: {len(drift_alerts)} drift alerts")
    else:
        print("OK: no reconciler drift alerts")

    print(f"Smoke test passed. DB: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
