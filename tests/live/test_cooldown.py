"""Tests for the order-cooldown wrapper and CooledTradingLoop suppression."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

from kairon.live.broker.base import Order, OrderSide, OrderStatus, OrderType
from kairon.live.config import LiveConfig
from kairon.live.cooldown import (
    DEFAULT_COOLDOWN_SECONDS,
    CooldownBrokerWrapper,
    CooledTradingLoop,
)
from kairon.live.guardian import Guardian
from kairon.live.predictor import LivePrediction
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore

# ---------------------------------------------------------------------------
# Minimal broker stub
# ---------------------------------------------------------------------------


class _StubBroker:
    """Records placed orders; returns them FILLED."""

    def __init__(self) -> None:
        self.placed: list[Order] = []

    async def place_order(self, order: Order) -> Order:
        self.placed.append(order)
        return order.model_copy(
            update={"status": OrderStatus.FILLED, "broker_id": "stub-1"}
        )


def _make_order(symbol: str = "BTC-USDT-PERP") -> Order:
    return Order(
        id="ord-1",
        intent_id="int-1",
        trace_id="tr-1",
        symbol=symbol,
        side=OrderSide.BUY,
        qty=0.01,
        order_type=OrderType.MARKET,
        status=OrderStatus.SUBMITTED,
        ts="2026-06-18T00:00:00+00:00",
    )


def _make_config(**overrides: object) -> LiveConfig:
    defaults: dict[str, object] = {
        "symbols": ("BTC-USDT-PERP",),
        "timeframe": "1m",
        "cadence_seconds": 10,
        "max_daily_loss_pct": 0.03,
        "max_open_positions": 5,
        "warmup_bars": 2,
        "reconcile_interval_seconds": 300,
        "reconcile_grace_seconds": 120,
        "dry_run": True,
    }
    defaults.update(overrides)
    return LiveConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CooldownBrokerWrapper
# ---------------------------------------------------------------------------


class TestCooldownBrokerWrapper:
    async def test_no_cooldown_before_order(self) -> None:
        broker = CooldownBrokerWrapper(_StubBroker(), cooldown_seconds=60.0)
        assert not broker.is_cooling_down("BTC-USDT-PERP")

    async def test_cooldown_after_order_then_expires(self) -> None:
        broker = CooldownBrokerWrapper(_StubBroker(), cooldown_seconds=60.0)
        await broker.place_order(_make_order())
        # Within the window (use an explicit now to avoid wall-clock flakiness).
        last = broker._last_order_ts["BTC-USDT-PERP"]
        assert broker.is_cooling_down("BTC-USDT-PERP", now=last + 1.0)
        # After the window expires.
        assert not broker.is_cooling_down("BTC-USDT-PERP", now=last + 61.0)

    async def test_rejected_order_does_not_start_cooldown(self) -> None:
        class _RejectBroker(_StubBroker):
            async def place_order(self, order: Order) -> Order:
                self.placed.append(order)
                return order.model_copy(update={"status": OrderStatus.REJECTED})

        broker = CooldownBrokerWrapper(_RejectBroker(), cooldown_seconds=60.0)
        await broker.place_order(_make_order())
        assert not broker.is_cooling_down("BTC-USDT-PERP")

    def test_default_cooldown_constant(self) -> None:
        assert DEFAULT_COOLDOWN_SECONDS == 5 * 60


# ---------------------------------------------------------------------------
# CooledTradingLoop._make_prediction suppression
# ---------------------------------------------------------------------------


class _FakeTable:
    """Stand-in bar table with enough rows to exceed warmup_bars."""

    num_rows = 100


class _LongStrategy:
    """Returns a LONG prediction when the strategy path is reached."""

    warmup_bars = 2

    def predict(self, bar_table: object, symbol: str) -> LivePrediction:
        return LivePrediction(
            symbol=symbol,
            direction=1.0,
            magnitude=0.5,
            volatility=0.01,
            confidence=0.8,
            horizon="day",
            ts="2026-06-18T00:00:00+00:00",
        )


class TestCooledTradingLoopSuppression:
    async def test_neutral_during_cooldown(self) -> None:
        config = _make_config()
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            try:
                wrapped = CooldownBrokerWrapper(_StubBroker(), cooldown_seconds=60.0)
                loop = CooledTradingLoop(
                    config=config,
                    broker=wrapped,
                    strategy=_LongStrategy(),
                    guardian=Guardian(),
                    reconciler=Reconciler(),
                    store=store,
                    feed=object(),
                )
                # Force the buffer to look warmed up so super() would otherwise
                # call the strategy — but the cooldown branch must short-circuit.
                loop._buffer_to_table = lambda symbol: _FakeTable()  # type: ignore[assignment]
                wrapped._last_order_ts["BTC-USDT-PERP"] = time.time()
                pred = loop._make_prediction("BTC-USDT-PERP")
                assert pred.direction == 0.0
                assert pred.confidence == 0.0
            finally:
                store.close()

    async def test_delegates_to_strategy_when_not_cooling(self) -> None:
        config = _make_config()
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            try:
                wrapped = CooldownBrokerWrapper(_StubBroker(), cooldown_seconds=60.0)
                loop = CooledTradingLoop(
                    config=config,
                    broker=wrapped,
                    strategy=_LongStrategy(),
                    guardian=Guardian(),
                    reconciler=Reconciler(),
                    store=store,
                    feed=object(),
                )
                loop._buffer_to_table = lambda symbol: _FakeTable()  # type: ignore[assignment]
                pred = loop._make_prediction("BTC-USDT-PERP")
                # Not cooling down → super() runs the strategy → LONG signal.
                assert pred.direction == 1.0
                assert pred.confidence == 0.8
            finally:
                store.close()
