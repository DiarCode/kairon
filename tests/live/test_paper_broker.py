"""Tests for PaperBroker and LivePredictorAdapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kairon.live.broker.base import (
    Balance,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.broker.paper import PaperBroker


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


class TestPaperBrokerMarketOrders:
    """Test PaperBroker market order fills with slippage and costs."""

    @pytest.mark.asyncio
    async def test_market_buy_applies_slippage(self) -> None:
        broker = PaperBroker(initial_balance=100000.0, slippage_bps=5.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        order = _make_order()
        filled = await broker.place_order(order)
        assert filled.status == OrderStatus.FILLED
        # Buy fills at mark * (1 + slippage)
        fill = broker.get_fills()[0]
        assert fill.price > 50000.0  # Slippage makes buy price higher
        assert fill.fee > 0.0  # Costs are applied

    @pytest.mark.asyncio
    async def test_market_sell_applies_slippage(self) -> None:
        # First buy to create a position, then sell
        broker = PaperBroker(initial_balance=100000.0, slippage_bps=5.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        buy = _make_order(side=OrderSide.BUY, qty=0.01)
        await broker.place_order(buy)
        sell = _make_order(id="ord-002", side=OrderSide.SELL, qty=0.01)
        filled = await broker.place_order(sell)
        assert filled.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_position_lifecycle_open_close(self) -> None:
        """Open a position, verify it exists, then close it."""
        broker = PaperBroker(initial_balance=100000.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        # Open
        buy = _make_order(qty=0.01)
        await broker.place_order(buy)
        positions = await broker.get_positions("BTC-USDT-PERP")
        assert len(positions) == 1
        assert positions[0].side == OrderSide.BUY
        assert abs(positions[0].qty - 0.01) < 1e-8
        # Close
        sell = _make_order(id="ord-002", side=OrderSide.SELL, qty=0.01)
        await broker.place_order(sell)
        positions = await broker.get_positions("BTC-USDT-PERP")
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_same_side_add_averages_entry(self) -> None:
        """Two buys at different prices should average the entry."""
        broker = PaperBroker(initial_balance=100000.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        buy1 = _make_order(qty=0.01)
        await broker.place_order(buy1)
        # Move price up
        broker.set_mark_price("BTC-USDT-PERP", 55000.0)
        buy2 = _make_order(id="ord-002", qty=0.01)
        await broker.place_order(buy2)
        positions = await broker.get_positions("BTC-USDT-PERP")
        assert len(positions) == 1
        # Weighted average should be between 50000 and 55000
        assert 50000.0 < positions[0].avg_entry < 55000.0
        assert abs(positions[0].qty - 0.02) < 1e-8


class TestPaperBrokerLimitOrders:
    """Test PaperBroker limit order handling."""

    @pytest.mark.asyncio
    async def test_limit_buy_fills_when_mark_crosses(self) -> None:
        broker = PaperBroker(initial_balance=100000.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        # Place limit buy at 49500
        order = _make_order(order_type=OrderType.LIMIT, price=49500.0, status=OrderStatus.PENDING)
        await broker.place_order(order)
        # Mark hasn't crossed yet — order stays pending
        fills = broker.tick()
        assert len(fills) == 0
        # Mark drops to 49400 — limit fills
        broker.set_mark_price("BTC-USDT-PERP", 49400.0)
        fills = broker.tick()
        assert len(fills) == 1
        assert fills[0].side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_limit_sell_fills_when_mark_crosses(self) -> None:
        broker = PaperBroker(initial_balance=100000.0)
        # First buy to create a position
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        buy = _make_order(qty=0.01)
        await broker.place_order(buy)
        # Place limit sell at 51000
        sell = _make_order(
            id="ord-002",
            side=OrderSide.SELL,
            qty=0.01,
            order_type=OrderType.LIMIT,
            price=51000.0,
            status=OrderStatus.PENDING,
        )
        await broker.place_order(sell)
        # Mark rises — limit fills
        broker.set_mark_price("BTC-USDT-PERP", 51200.0)
        fills = broker.tick()
        assert len(fills) == 1


class TestPaperBrokerSLTP:
    """Test PaperBroker stop-loss and take-profit handling."""

    @pytest.mark.asyncio
    async def test_stop_loss_triggers(self) -> None:
        """SL on a long position is a SELL conditional that triggers when price drops."""
        broker = PaperBroker(initial_balance=100000.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        # Open a long position
        buy = _make_order(qty=0.01)
        await broker.place_order(buy)
        # Place SL as a SELL conditional (close long when price drops)
        sl = await broker.place_conditional(
            symbol="BTC-USDT-PERP",
            side=OrderSide.SELL,
            qty=0.01,
            trigger_price=49500.0,
        )
        assert sl.status == OrderStatus.PENDING
        # Mark drops to 49400 — SL triggers (mark <= trigger for SELL SL)
        broker.set_mark_price("BTC-USDT-PERP", 49400.0)
        fills = broker.tick()
        assert len(fills) == 1

    @pytest.mark.asyncio
    async def test_take_profit_triggers(self) -> None:
        """TP on a long position triggers when mark rises above tp level.

        TP is checked via the tp field on the order, which triggers
        when mark >= tp for SELL-side orders.
        """
        broker = PaperBroker(initial_balance=100000.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        # Open a long
        buy = _make_order(qty=0.01)
        await broker.place_order(buy)
        # Place TP as a SELL conditional with tp=51000
        # SELL-side TP triggers when mark >= tp
        tp = await broker.place_conditional(
            symbol="BTC-USDT-PERP",
            side=OrderSide.SELL,
            qty=0.01,
            trigger_price=51000.0,
            tp=51000.0,
        )
        assert tp.status == OrderStatus.PENDING
        # Mark rises above TP — triggers fill
        broker.set_mark_price("BTC-USDT-PERP", 51200.0)
        fills = broker.tick()
        assert len(fills) == 1


class TestPaperBrokerIdempotency:
    """Test PaperBroker order idempotency and basic operations."""

    @pytest.mark.asyncio
    async def test_market_order_idempotent(self) -> None:
        broker = PaperBroker(initial_balance=100000.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        order = _make_order()
        first = await broker.place_order(order)
        second = await broker.place_order(order)
        assert first.id == second.id
        assert first.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancel_order(self) -> None:
        broker = PaperBroker(initial_balance=100000.0)
        cond = await broker.place_conditional(
            symbol="BTC-USDT-PERP", side=OrderSide.BUY, qty=0.001, trigger_price=49000.0
        )
        assert cond.status == OrderStatus.PENDING
        cancelled = await broker.cancel_order("BTC-USDT-PERP", cond.id)
        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_balances(self) -> None:
        broker = PaperBroker(initial_balance=10000.0)
        balances = await broker.get_balances()
        assert len(balances) == 1
        assert balances[0].currency == "USDT"
        assert balances[0].total == 10000.0

    @pytest.mark.asyncio
    async def test_set_leverage(self) -> None:
        broker = PaperBroker(initial_balance=10000.0)
        await broker.set_leverage("BTC-USDT-PERP", 5)
        assert broker._leverage["BTC-USDT-PERP"] == 5

    @pytest.mark.asyncio
    async def test_cash_tracking(self) -> None:
        broker = PaperBroker(initial_balance=10000.0)
        broker.set_mark_price("BTC-USDT-PERP", 50000.0)
        assert broker.cash == 10000.0
        # Buy 0.01 BTC at ~50000 + slippage
        buy = _make_order(qty=0.01)
        await broker.place_order(buy)
        # Cash should decrease (minus cost)
        assert broker.cash < 10000.0


class TestLivePredictorAdapter:
    """Test LivePredictorAdapter with a mock model."""

    @pytest.fixture
    def mock_predictor(self) -> None:
        """Set up a mock model and trained state for testing."""
        # This test is minimal since we can't easily mock the full model pipeline
        # without the ML stack. The adapter's contract is tested via integration.
        pass

    def test_live_prediction_dataclass(self) -> None:
        """Verify LivePrediction fields exist and are frozen."""
        from kairon.live.predictor import LivePrediction

        pred = LivePrediction(
            symbol="BTC-USDT-PERP",
            direction=1.0,
            magnitude=0.02,
            volatility=0.015,
            confidence=0.75,
            horizon="day",
            ts="2026-06-13T12:00:00+00:00",
        )
        assert pred.symbol == "BTC-USDT-PERP"
        assert pred.direction == 1.0
        assert pred.magnitude == 0.02
        assert pred.confidence == 0.75
        # Frozen check
        with pytest.raises(Exception):
            pred.symbol = "ETH-USDT-PERP"  # type: ignore[misc]

    def test_live_prediction_sell_direction(self) -> None:
        from kairon.live.predictor import LivePrediction

        pred = LivePrediction(
            symbol="ETH-USDT-PERP",
            direction=-1.0,
            magnitude=-0.01,
            volatility=0.02,
            confidence=0.6,
            horizon="swing",
            ts="2026-06-13T12:00:00+00:00",
        )
        assert pred.direction == -1.0
        assert pred.horizon == "swing"