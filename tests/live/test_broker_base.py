"""Tests for the Broker protocol, dataclasses, and FakeBroker."""

from __future__ import annotations

import pytest

from kairon.live.broker.base import (
    Balance,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.broker.fake import FakeBroker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(**overrides: object) -> Order:
    """Create a minimal Order with sensible defaults."""
    defaults = {
        "id": "ord-001",
        "intent_id": "intent-001",
        "trace_id": "trace-001",
        "symbol": "BTC-USDT-PERP",
        "side": OrderSide.BUY,
        "qty": 0.001,
        "order_type": OrderType.MARKET,
        "status": OrderStatus.PENDING,
        "ts": "2026-06-13T00:00:00+00:00",
    }
    defaults.update(overrides)
    return Order(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Order / Fill / Position / Balance dataclass tests
# ---------------------------------------------------------------------------


class TestBrokerDataclasses:
    """Test pydantic v2 frozen/strict dataclasses."""

    def test_order_creation(self) -> None:
        order = _make_order()
        assert order.symbol == "BTC-USDT-PERP"
        assert order.side == OrderSide.BUY
        assert order.qty == 0.001
        assert order.status == OrderStatus.PENDING

    def test_order_frozen(self) -> None:
        order = _make_order()
        with pytest.raises(Exception):
            order.symbol = "ETH-USDT-PERP"  # type: ignore[misc]

    def test_order_with_sl_tp(self) -> None:
        order = _make_order(sl=49500.0, tp=51000.0)
        assert order.sl == 49500.0
        assert order.tp == 51000.0

    def test_fill_creation(self) -> None:
        fill = Fill(
            id="fill-001",
            order_id="ord-001",
            symbol="BTC-USDT-PERP",
            side=OrderSide.BUY,
            qty=0.001,
            price=50000.0,
            fee=0.005,
            fee_ccy="USDT",
            ts="2026-06-13T00:00:01+00:00",
        )
        assert fill.qty == 0.001
        assert fill.fee_ccy == "USDT"

    def test_position_creation(self) -> None:
        pos = Position(
            symbol="BTC-USDT-PERP",
            side=OrderSide.BUY,
            qty=0.001,
            avg_entry=50000.0,
            unrealized_pnl=0.0,
            ts="2026-06-13T00:00:01+00:00",
        )
        assert pos.qty == 0.001

    def test_balance_creation(self) -> None:
        bal = Balance(
            currency="USDT",
            available=9500.0,
            total=10000.0,
            ts="2026-06-13T00:00:01+00:00",
        )
        assert bal.available == 9500.0


# ---------------------------------------------------------------------------
# FakeBroker tests
# ---------------------------------------------------------------------------


class TestFakeBroker:
    """Test the in-memory FakeBroker for unit tests."""

    @pytest.mark.asyncio
    async def test_place_market_order_fills_instantly(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        order = _make_order()
        filled = await broker.place_order(order)
        assert filled.status == OrderStatus.FILLED
        assert filled.broker_id is not None
        assert filled.broker_id.startswith("fake-")

    @pytest.mark.asyncio
    async def test_place_order_idempotent(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        order = _make_order()
        first = await broker.place_order(order)
        second = await broker.place_order(order)
        assert first.id == second.id
        assert first.broker_id == second.broker_id

    @pytest.mark.asyncio
    async def test_place_order_creates_fill(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        order = _make_order()
        await broker.place_order(order)
        fills = broker._fills
        assert len(fills) == 1
        assert fills[0].order_id == order.id
        assert fills[0].price == 50000.0

    @pytest.mark.asyncio
    async def test_place_order_updates_position(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        order = _make_order(qty=0.01)
        await broker.place_order(order)
        positions = await broker.get_positions(symbol="BTC-USDT-PERP")
        assert len(positions) == 1
        assert positions[0].qty == 0.01

    @pytest.mark.asyncio
    async def test_cancel_order(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        # Place a conditional order (stays PENDING)
        cond = await broker.place_conditional(
            symbol="BTC-USDT-PERP",
            side=OrderSide.BUY,
            qty=0.001,
            trigger_price=49000.0,
        )
        assert cond.status == OrderStatus.PENDING
        cancelled = await broker.cancel_order("BTC-USDT-PERP", cond.id)
        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_all(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        await broker.place_conditional("BTC-USDT-PERP", OrderSide.BUY, 0.001, 49000.0)
        await broker.place_conditional("BTC-USDT-PERP", OrderSide.SELL, 0.001, 51000.0)
        cancelled = await broker.cancel_all("BTC-USDT-PERP")
        assert len(cancelled) == 2

    @pytest.mark.asyncio
    async def test_get_balances(self) -> None:
        broker = FakeBroker(initial_balance=10000.0)
        balances = await broker.get_balances()
        assert len(balances) == 1
        assert balances[0].currency == "USDT"
        assert balances[0].total == 10000.0

    @pytest.mark.asyncio
    async def test_get_open_orders(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        await broker.place_conditional("BTC-USDT-PERP", OrderSide.BUY, 0.001, 49000.0)
        open_orders = await broker.get_open_orders(symbol="BTC-USDT-PERP")
        assert len(open_orders) == 1
        assert open_orders[0].status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_set_leverage(self) -> None:
        broker = FakeBroker()
        await broker.set_leverage("BTC-USDT-PERP", 5)
        assert broker._leverage["BTC-USDT-PERP"] == 5

    @pytest.mark.asyncio
    async def test_place_conditional(self) -> None:
        broker = FakeBroker(mark_price=50000.0)
        order = await broker.place_conditional(
            symbol="BTC-USDT-PERP",
            side=OrderSide.BUY,
            qty=0.001,
            trigger_price=49000.0,
            sl=48500.0,
            tp=51000.0,
        )
        assert order.status == OrderStatus.PENDING
        assert order.sl == 48500.0
        assert order.tp == 51000.0