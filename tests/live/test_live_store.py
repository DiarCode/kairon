"""Tests for LiveStore (co-located with RunStore in the same sqlite file)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kairon.live.broker.base import Order, OrderSide, OrderStatus, OrderType, Position
from kairon.live.store import LiveStore

# RunStore requires the full ML stack (pyarrow, models, etc.).
# Import conditionally so basic LiveStore tests work without it.
try:
    from kairon.store.runs import RunStore

    HAS_RUNSTORE = True
except ImportError:
    HAS_RUNSTORE = False


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


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """Return a path to a temporary database file."""
    return tmp_path / "test_live.db"


@pytest.fixture
def store(store_path: Path) -> LiveStore:
    """Return a LiveStore connected to the temporary database."""
    s = LiveStore(store_path)
    yield s
    s.close()


class TestLiveStoreCRUD:
    """Test basic CRUD operations on LiveStore tables."""

    def test_write_and_get_order(self, store: LiveStore) -> None:
        order = _make_order()
        store.write_order(order)
        fetched = store.get_order(order.id)
        assert fetched is not None
        assert fetched.symbol == "BTC-USDT-PERP"
        assert fetched.side == OrderSide.BUY

    def test_update_order_status(self, store: LiveStore) -> None:
        order = _make_order()
        store.write_order(order)
        store.update_order_status(order.id, OrderStatus.FILLED, broker_id="bybit-123")
        fetched = store.get_order(order.id)
        assert fetched is not None
        assert fetched.status == OrderStatus.FILLED
        assert fetched.broker_id == "bybit-123"

    def test_get_nonexistent_order(self, store: LiveStore) -> None:
        assert store.get_order("nonexistent") is None

    def test_write_and_get_positions(self, store: LiveStore) -> None:
        pos = Position(
            symbol="BTC-USDT-PERP",
            side=OrderSide.BUY,
            qty=0.01,
            avg_entry=50000.0,
            unrealized_pnl=5.0,
            ts="2026-06-13T00:00:01+00:00",
        )
        store.write_position(pos)
        positions = store.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC-USDT-PERP"

    def test_delete_position(self, store: LiveStore) -> None:
        pos = Position(
            symbol="BTC-USDT-PERP",
            side=OrderSide.BUY,
            qty=0.01,
            avg_entry=50000.0,
            unrealized_pnl=0.0,
            ts="2026-06-13T00:00:01+00:00",
        )
        store.write_position(pos)
        store.delete_position("BTC-USDT-PERP")
        assert len(store.get_positions()) == 0


class TestLiveStoreHaltSwitch:
    """Test the kill switch (halt/unhalt/is_halted)."""

    def test_halt_sets_flag(self, store: LiveStore) -> None:
        assert store.is_halted() is False
        store.halt("daily_loss_limit")
        assert store.is_halted() is True

    def test_unhalt_clears_flag(self, store: LiveStore) -> None:
        store.halt("test")
        assert store.is_halted() is True
        store.unhalt()
        assert store.is_halted() is False

    def test_halt_reason_stored(self, store: LiveStore) -> None:
        store.halt("user_pressed_button")
        reason = store.get_runtime_state("halted")
        assert reason == "user_pressed_button"


@pytest.mark.skipif(not HAS_RUNSTORE, reason="RunStore requires full ML stack")
class TestLiveStoreCoexistence:
    """Test that LiveStore tables coexist with RunStore in the same db file."""

    def test_tables_coexist_with_runs_store(self, tmp_path: Path) -> None:
        db_path = tmp_path / "runs.db"
        run_store = RunStore(db_path)
        live_store = LiveStore(db_path)

        try:
            # Write to RunStore
            from datetime import datetime, timezone

            from kairon.analysis.contracts import ModelTile, ProvenanceBlock, RunResult

            run = RunResult(
                run_id="run-001",
                asset="BTC-USDT",
                horizon="day",
                created_at_utc=datetime.now(timezone.utc),
                models=(
                    ModelTile(
                        name="trend",
                        chart_png_path="/tmp/chart.png",
                        predicted_pct=0.02,
                        stop_loss=49500.0,
                        ideal_entry=50000.0,
                        ideal_exit=51000.0,
                        confidence=0.8,
                    ),
                ),
                provenance=ProvenanceBlock(
                    config_hash="abc123",
                    data_hash="def456",
                    model_version="1.0",
                    seed=42,
                ),
                base_price=50000.0,
            )
            run_store.create(run, Path("/tmp/input.csv"))

            # Write to LiveStore
            order = _make_order()
            live_store.write_order(order)

            # Verify both can read
            fetched_run = run_store.get("run-001")
            assert fetched_run is not None
            assert fetched_run.asset == "BTC-USDT"

            fetched_order = live_store.get_order(order.id)
            assert fetched_order is not None
            assert fetched_order.symbol == "BTC-USDT-PERP"
        finally:
            run_store.close()
            live_store.close()


class TestLiveStoreHeartbeat:
    """Test heartbeat writing."""

    def test_write_heartbeat(self, store: LiveStore) -> None:
        store.write_heartbeat(mode="dry_run", equity=10000.0, n_positions=2)
        # No error = success


class TestLiveStoreEvents:
    """Test audit event writing."""

    def test_write_event(self, store: LiveStore) -> None:
        store.write_event(
            kind="guardian_trip", severity="critical", payload_json='{"reason": "daily_loss"}'
        )


class TestLiveStoreClose:
    """Test that LiveStore.close() is idempotent."""

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        store = LiveStore(tmp_path / "test_close.db")
        store.close()
        store.close()  # Second close should not raise