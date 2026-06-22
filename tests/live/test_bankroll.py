"""Tests for the synthetic-bankroll growth mode in :class:`TradingLoop`.

Covers the opt-in ``bankroll`` config: compounding on realized PnL, the
``growth_ledger`` audit rows, milestone logging, halt at ``stop_at``, depletion
clamp at zero, the default-off no-op, and that bankroll-mode sizing produces a
tradeable quantity for a small symbol (XRP) from a 10 USDT bankroll.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pytest

from kairon.live.broker.base import Balance, Order, OrderStatus
from kairon.live.config import BankrollConfig, LiveConfig
from kairon.live.guardian import Guardian
from kairon.live.orchestrator import TradingLoop
from kairon.live.predictor import LivePrediction
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore


class _Broker:
    """Minimal broker for bankroll tests: records orders, no fill streaming."""

    def __init__(self, equity: float = 10_000.0) -> None:
        self._equity = equity
        self.orders: list[Order] = []

    async def place_order(self, order: Order) -> Order:
        self.orders.append(order)
        return order.model_copy(
            update={"status": OrderStatus.FILLED, "broker_id": f"mock-{order.id[:8]}"}
        )

    async def get_balances(self) -> list[Balance]:
        return [Balance(currency="USDT", available=self._equity, total=self._equity,
                        ts="2026-06-18T12:00:00+00:00")]

    async def get_positions(self, symbol: str | None = None) -> list:
        return []

    def min_qty_for(self, symbol: str) -> float:
        # XRP min order qty on Bybit linear is 0.1.
        return {"XRP-USDT-PERP": 0.1, "BTC-USDT-PERP": 0.001, "ETH-USDT-PERP": 0.01}.get(symbol, 1e-9)


class _Strategy:
    """Mock strategy that always returns a long signal."""

    warmup_bars = 1

    def predict(self, table: object, symbol: str) -> LivePrediction:
        return LivePrediction(
            symbol=symbol,
            direction=1.0,
            magnitude=0.1,
            volatility=0.01,
            confidence=0.6,
            horizon="day",
            ts="2026-06-18T12:00:00+00:00",
        )


def _xrp_bar(close: float = 0.5, minute: int = 0) -> pa.Table:
    ts = datetime(2026, 6, 18, 12, minute, 0, tzinfo=UTC)
    return pa.table(
        {
            "ts": pa.array([ts], type=pa.timestamp("us", tz="UTC")),
            "open": pa.array([close], type=pa.float64()),
            "high": pa.array([close], type=pa.float64()),
            "low": pa.array([close], type=pa.float64()),
            "close": pa.array([close], type=pa.float64()),
            "volume": pa.array([100.0], type=pa.float64()),
            "symbol": pa.array(["XRP-USDT-PERP"], type=pa.string()),
        }
    )


def _config(**overrides: object) -> LiveConfig:
    defaults: dict[str, object] = {
        "symbols": ("XRP-USDT-PERP",),
        "timeframe": "1m",
        "cadence_seconds": 60,
        "max_daily_loss_pct": 0.03,
        "max_open_positions": 3,
        "warmup_bars": 1,
        "reconcile_interval_seconds": 300,
        "reconcile_grace_seconds": 120,
        "dry_run": True,
    }
    defaults.update(overrides)
    return LiveConfig(**defaults)  # type: ignore[arg-type]


class TestBankrollCompounding:
    @pytest.mark.asyncio
    async def test_close_compounds_and_halts_at_stop(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=15.0, milestones=(15.0,))
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "g.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_Strategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg,
            )
            loop._running = True
            loop._apply_bankroll_close("XRP-USDT-PERP", 5.0)

            assert loop.bankroll == 15.0
            assert loop._running is False
            assert store.is_halted()

            kinds = [row["kind"] for row in store.get_ledger()]
            assert "start" in kinds
            assert "close" in kinds
            assert "milestone" in kinds
            assert "halt" in kinds
            store.close()

    @pytest.mark.asyncio
    async def test_close_clamps_at_zero_and_halts(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=())
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "g.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_Strategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg,
            )
            loop._running = True
            loop._apply_bankroll_close("XRP-USDT-PERP", -20.0)

            assert loop.bankroll == 0.0
            assert loop._running is False
            assert store.is_halted()
            store.close()

    @pytest.mark.asyncio
    async def test_default_off_is_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "g.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_Strategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(),  # no bankroll
            )
            assert loop.bankroll == 0.0
            assert loop.bankroll_config is None
            loop._apply_bankroll_close("XRP-USDT-PERP", 5.0)  # no-op
            assert loop.bankroll == 0.0
            assert store.get_ledger() == []
            store.close()

    @pytest.mark.asyncio
    async def test_bankroll_sizing_produces_tradable_qty(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(50.0, 100.0))
        broker = _Broker()
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "g.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=_Strategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg,
            )
            # bankroll=10, leverage=10, allocation=1 -> notional=100 USDT.
            # XRP @ 0.5 -> target qty = 200, well above the 0.1 min.
            # First tick warms up (warmup_bars=1); the second tick trades.
            await loop._process_tick(_xrp_bar(0.5, minute=0))
            await loop._process_tick(_xrp_bar(0.5, minute=1))

            assert len(broker.orders) == 1
            assert broker.orders[0].qty >= 0.1
            # Position opened (new), no close yet -> bankroll unchanged.
            assert loop.bankroll == 10.0
            store.close()
