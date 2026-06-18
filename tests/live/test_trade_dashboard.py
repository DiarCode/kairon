"""Tests for the trade dashboard API endpoints and LiveStore dashboard queries."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kairon.live.broker.base import Order, OrderSide, OrderStatus, OrderType, Position
from kairon.live.store import LiveStore


# ---------------------------------------------------------------------------
# LiveStore dashboard query tests
# ---------------------------------------------------------------------------


class TestLiveStoreDashboardQueries:
    """Test the new dashboard query methods on LiveStore."""

    def test_get_recent_heartbeat_empty(self) -> None:
        """Returns None when no heartbeats exist."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            result = store.get_recent_heartbeat()
            assert result is None
            store.close()

    def test_get_recent_heartbeat_returns_latest(self) -> None:
        """Returns the most recent heartbeat."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            store.write_heartbeat(mode="dry_run", equity=10000.0, n_positions=2)
            store.write_heartbeat(mode="live", equity=10500.0, n_positions=3)

            result = store.get_recent_heartbeat()
            assert result is not None
            assert result["mode"] == "live"
            assert result["equity"] == 10500.0
            assert result["n_positions"] == 3
            store.close()

    def test_get_recent_orders_empty(self) -> None:
        """Returns empty list when no orders exist."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            orders = store.get_recent_orders()
            assert orders == []
            store.close()

    def test_get_recent_orders_returns_latest(self) -> None:
        """Returns recent orders ordered by timestamp descending."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            order1 = Order(
                id="ord-001", intent_id="int-001", trace_id="tr-001",
                symbol="BTC-USDT-PERP", side=OrderSide.BUY, qty=0.01,
                order_type=OrderType.MARKET, status=OrderStatus.FILLED,
                ts="2026-06-13T10:00:00+00:00",
            )
            order2 = Order(
                id="ord-002", intent_id="int-002", trace_id="tr-002",
                symbol="ETH-USDT-PERP", side=OrderSide.SELL, qty=0.1,
                order_type=OrderType.LIMIT, price=3000.0,
                status=OrderStatus.SUBMITTED,
                ts="2026-06-13T11:00:00+00:00",
            )
            store.write_order(order1)
            store.write_order(order2)

            orders = store.get_recent_orders()
            assert len(orders) == 2
            # Most recent first
            assert orders[0]["id"] == "ord-002"
            assert orders[1]["id"] == "ord-001"
            store.close()

    def test_get_recent_orders_respects_limit(self) -> None:
        """Returns at most `limit` orders."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            for i in range(10):
                order = Order(
                    id=f"ord-{i:03d}", intent_id=f"int-{i:03d}", trace_id=f"tr-{i:03d}",
                    symbol="BTC-USDT-PERP", side=OrderSide.BUY, qty=0.001,
                    order_type=OrderType.MARKET, status=OrderStatus.FILLED,
                    ts=f"2026-06-13T{i%24:02d}:00:00+00:00",
                )
                store.write_order(order)

            orders = store.get_recent_orders(limit=5)
            assert len(orders) == 5
            store.close()

    def test_get_recent_events_empty(self) -> None:
        """Returns empty list when no events exist."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            events = store.get_recent_events()
            assert events == []
            store.close()

    def test_get_recent_events_returns_latest(self) -> None:
        """Returns recent events ordered by ID descending."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            store.write_event(kind="start", severity="info", payload_json='{"msg": "started"}')
            store.write_event(kind="halt", severity="critical", payload_json='{"reason": "daily_loss"}')

            events = store.get_recent_events()
            assert len(events) == 2
            assert events[0]["kind"] == "halt"
            assert events[1]["kind"] == "start"
            store.close()


# ---------------------------------------------------------------------------
# Trade screen import tests
# ---------------------------------------------------------------------------


class TestTradeScreenImports:
    """Verify all trade screen handlers are importable."""

    def test_trade_screen_importable(self) -> None:
        from kairon.ui.web.screens import trade_screen
        assert trade_screen is not None

    def test_trade_api_importable(self) -> None:
        from kairon.ui.web.screens import (
            trade_events,
            trade_halt,
            trade_orders,
            trade_positions,
            trade_status,
            trade_unhalt,
        )
        assert trade_status is not None
        assert trade_positions is not None
        assert trade_orders is not None
        assert trade_events is not None
        assert trade_halt is not None
        assert trade_unhalt is not None