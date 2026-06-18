"""Tests for the Reconciler: position drift detection and orphan recovery."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kairon.live.alerts import AlertEngine, InMemoryChannel, Severity
from kairon.live.broker.base import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.reconciler import DriftFact, OrphanFact, Reconciler, _compute_drift
from kairon.live.store import LiveStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pos(symbol: str = "BTC-USDT-PERP", qty: float = 0.01, avg_entry: float = 50000.0) -> Position:
    return Position(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=qty,
        avg_entry=avg_entry,
        unrealized_pnl=0.0,
        ts="2026-06-13T12:00:00+00:00",
    )


class _FakeBroker:
    """Minimal broker stub for Reconciler tests."""

    def __init__(self, positions: list[Position] | None = None) -> None:
        self._positions = positions or []

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        if symbol:
            return [p for p in self._positions if p.symbol == symbol]
        return list(self._positions)


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


class TestComputeDrift:
    """Test the _compute_drift helper."""

    def test_zero_drift(self) -> None:
        assert _compute_drift(0.01, 0.01) == 0.0

    def test_both_zero(self) -> None:
        assert _compute_drift(0.0, 0.0) == 0.0

    def test_partial_drift(self) -> None:
        # local=0.011, broker=0.010 → drift = 0.001/0.011 ≈ 9.1%
        drift = _compute_drift(0.011, 0.010)
        assert 0.08 < drift < 0.10

    def test_full_drift_missing_broker(self) -> None:
        # local has position, broker doesn't → drift = 1.0 (100%)
        assert _compute_drift(0.01, 0.0) == 1.0

    def test_full_drift_missing_local(self) -> None:
        # broker has position, local doesn't → drift = 1.0 (100%)
        assert _compute_drift(0.0, 0.01) == 1.0


# ---------------------------------------------------------------------------
# DriftFact matching
# ---------------------------------------------------------------------------


class TestReconcilerDriftFact:
    """Test Reconciler.matches() with DriftFact."""

    def test_critical_on_excessive_drift(self) -> None:
        reconciler = Reconciler(drift_tolerance_pct=0.05)
        fact = DriftFact(symbol="BTC-USDT-PERP", local_qty=0.02, broker_qty=0.01, drift_pct=0.50)
        alert = reconciler.matches(fact)
        assert alert is not None
        assert alert.severity == Severity.CRITICAL
        assert "drift" in alert.source

    def test_info_on_minor_drift(self) -> None:
        reconciler = Reconciler(drift_tolerance_pct=0.05)
        fact = DriftFact(symbol="BTC-USDT-PERP", local_qty=0.0101, broker_qty=0.01, drift_pct=0.01)
        alert = reconciler.matches(fact)
        assert alert is not None
        assert alert.severity == Severity.INFO

    def test_info_on_exact_tolerance(self) -> None:
        reconciler = Reconciler(drift_tolerance_pct=0.05)
        fact = DriftFact(symbol="BTC-USDT-PERP", local_qty=0.01, broker_qty=0.01, drift_pct=0.05)
        alert = reconciler.matches(fact)
        # Exact tolerance = not exceeding, so INFO
        assert alert is not None
        assert alert.severity == Severity.INFO

    def test_no_alert_on_zero_drift(self) -> None:
        """Zero drift returns INFO (within tolerance)."""
        reconciler = Reconciler(drift_tolerance_pct=0.05)
        fact = DriftFact(symbol="BTC-USDT-PERP", local_qty=0.01, broker_qty=0.01, drift_pct=0.0)
        alert = reconciler.matches(fact)
        assert alert is not None
        assert alert.severity == Severity.INFO


# ---------------------------------------------------------------------------
# OrphanFact matching
# ---------------------------------------------------------------------------


class TestReconcilerOrphanFact:
    """Test Reconciler.matches() with OrphanFact."""

    def test_warning_on_orphan(self) -> None:
        reconciler = Reconciler()
        fact = OrphanFact(order_id="ord-123", symbol="BTC-USDT-PERP", age_seconds=300)
        alert = reconciler.matches(fact)
        assert alert is not None
        assert alert.severity == Severity.WARNING
        assert "orphan" in alert.source

    def test_unknown_fact_returns_none(self) -> None:
        reconciler = Reconciler()
        result = reconciler.matches("not a fact")
        assert result is None


# ---------------------------------------------------------------------------
# Async reconcile
# ---------------------------------------------------------------------------


class TestReconcilerAsync:
    """Test the async reconcile() method with store and broker."""

    @pytest.mark.asyncio
    async def test_reconcile_no_drift(self) -> None:
        """When local and broker positions match, no drift alerts."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            broker = _FakeBroker(positions=[_pos(qty=0.01)])
            store.write_position(_pos(qty=0.01))

            reconciler = Reconciler(
                drift_tolerance_pct=0.05,
                grace_seconds=0,
                reconcile_interval_seconds=0,
                store=store,
                broker=broker,
            )
            alerts = await reconciler.reconcile()
            # No drift (qty matches), but reconcile still produces INFO-level alerts
            critical = [a for a in alerts if a.severity == Severity.CRITICAL]
            assert len(critical) == 0
            store.close()

    @pytest.mark.asyncio
    async def test_reconcile_detects_drift(self) -> None:
        """When local and broker positions differ, drift is detected."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            # Local says 0.01, broker says 0.02 → 50% drift
            broker = _FakeBroker(positions=[_pos(qty=0.02)])
            store.write_position(_pos(qty=0.01))

            reconciler = Reconciler(
                drift_tolerance_pct=0.05,
                grace_seconds=0,
                reconcile_interval_seconds=0,
                store=store,
                broker=broker,
            )
            alerts = await reconciler.reconcile()
            critical = [a for a in alerts if a.severity == Severity.CRITICAL]
            assert len(critical) >= 1
            assert any("drift" in a.source for a in critical)
            store.close()

    @pytest.mark.asyncio
    async def test_reconcile_no_false_drift_on_fill_during_broker_fetch(self) -> None:
        """A fill landing during the (slow) broker fetch must not produce a
        false drift alert.

        Simulates the live read-skew: the broker's get_positions takes time
        (REST latency), and the fill-drain task writes the new position to the
        store while we are awaiting the broker. Because local is read AFTER
        the broker, it sees the fill and no 100% drift alert fires. With the
        old local-first ordering, local would read 0 before the fill and the
        alert would fire spuriously.
        """
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")

            class _SlowBroker:
                """Broker that writes a position to the store mid-fetch."""

                async def get_positions(self, symbol: str | None = None) -> list[Position]:
                    # Simulate REST latency; the fill-drain task writes during this.
                    import asyncio
                    await asyncio.sleep(0.05)
                    # The fill has now been persisted by the time we return.
                    return [_pos(qty=0.011)]

            async def _write_fill_during_fetch() -> None:
                import asyncio
                await asyncio.sleep(0.02)  # let the broker fetch start
                store.write_position(_pos(qty=0.011))

            broker = _SlowBroker()
            reconciler = Reconciler(
                drift_tolerance_pct=0.05,
                grace_seconds=0,
                reconcile_interval_seconds=0,
                store=store,
                broker=broker,
            )
            # Race the fill write against the reconcile call.
            import asyncio
            _, alerts = await asyncio.gather(
                _write_fill_during_fetch(), reconciler.reconcile()
            )
            critical = [a for a in alerts if a.severity == Severity.CRITICAL]
            assert len(critical) == 0, (
                f"expected no false drift alert, got {[a.source for a in critical]}"
            )
            store.close()

    @pytest.mark.asyncio
    async def test_reconcile_removes_closed_position(self) -> None:
        """When broker has no position but local does, local entry is deleted."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            broker = _FakeBroker(positions=[])  # No positions on broker
            store.write_position(_pos(qty=0.01))

            reconciler = Reconciler(
                drift_tolerance_pct=0.05,
                grace_seconds=0,
                reconcile_interval_seconds=0,
                store=store,
                broker=broker,
            )
            await reconciler.reconcile()
            # Local position should be deleted
            positions = store.get_positions()
            assert len(positions) == 0
            store.close()

    @pytest.mark.asyncio
    async def test_reconcile_updates_local_from_broker(self) -> None:
        """When broker has a position that local doesn't, local is updated."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            broker = _FakeBroker(positions=[_pos(qty=0.02)])
            # No local positions

            reconciler = Reconciler(
                drift_tolerance_pct=0.05,
                grace_seconds=0,
                reconcile_interval_seconds=0,
                store=store,
                broker=broker,
            )
            await reconciler.reconcile()
            # Local should now have the broker's position
            positions = store.get_positions()
            assert len(positions) == 1
            assert positions[0].qty == 0.02
            store.close()

    @pytest.mark.asyncio
    async def test_reconcile_throttles(self) -> None:
        """Reconcile skips if called too soon after the previous call."""
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            broker = _FakeBroker(positions=[_pos(qty=0.01)])

            reconciler = Reconciler(
                reconcile_interval_seconds=60,  # 60s minimum between calls
                store=store,
                broker=broker,
            )
            # First call succeeds
            alerts1 = await reconciler.reconcile()
            # Second call within 60s is skipped
            alerts2 = await reconciler.reconcile()
            assert len(alerts2) == 0
            store.close()

    @pytest.mark.asyncio
    async def test_reconcile_no_store_or_broker(self) -> None:
        """Reconcile with no store or broker returns empty alerts."""
        reconciler = Reconciler()
        alerts = await reconciler.reconcile()
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# AlertEngine integration
# ---------------------------------------------------------------------------


class TestReconcilerAlertEngineIntegration:
    """Test Reconciler as an AlertEngine.Rule."""

    def test_drift_fact_through_engine(self) -> None:
        reconciler = Reconciler(drift_tolerance_pct=0.05)
        engine = AlertEngine(rules=[reconciler], channels=[InMemoryChannel()])
        fact = DriftFact(symbol="BTC-USDT-PERP", local_qty=0.02, broker_qty=0.01, drift_pct=0.50)
        alerts = engine.evaluate(fact)
        assert len(alerts) >= 1
        assert any(a.severity == Severity.CRITICAL for a in alerts)

    def test_orphan_fact_through_engine(self) -> None:
        reconciler = Reconciler()
        engine = AlertEngine(rules=[reconciler], channels=[InMemoryChannel()])
        fact = OrphanFact(order_id="ord-123", symbol="BTC-USDT-PERP", age_seconds=300)
        alerts = engine.evaluate(fact)
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.WARNING


# ---------------------------------------------------------------------------
# Drift cache
# ---------------------------------------------------------------------------


class TestReconcilerDriftCache:
    """Test the drift cache."""

    @pytest.mark.asyncio
    async def test_get_drift_after_reconcile(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            broker = _FakeBroker(positions=[_pos(qty=0.02)])
            store.write_position(_pos(qty=0.01))

            reconciler = Reconciler(
                drift_tolerance_pct=0.05,
                grace_seconds=0,
                reconcile_interval_seconds=0,
                store=store,
                broker=broker,
            )
            await reconciler.reconcile()
            drift = reconciler.get_drift("BTC-USDT-PERP")
            assert drift is not None
            assert drift > 0.0  # There is drift
            store.close()

    def test_get_drift_no_data(self) -> None:
        reconciler = Reconciler()
        assert reconciler.get_drift("BTC-USDT-PERP") is None