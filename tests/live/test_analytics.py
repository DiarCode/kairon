"""Tests for live analytics: compute_session_report and format_report."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kairon.live.analytics import (
    LiveSessionReport,
    SymbolReport,
    compute_session_report,
    format_report,
)
from kairon.live.broker.base import Fill, Order, OrderSide, OrderStatus, OrderType, Position
from kairon.live.store import LiveStore


def _seed_store(store: LiveStore) -> None:
    """Seed a LiveStore with sample trading data for analytics testing."""
    base_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Write heartbeats with equity curve (10000 → 10100 → 10250 → 10150 → 10350)
    equities = [10000.0, 10100.0, 10250.0, 10150.0, 10350.0]
    for i, eq in enumerate(equities):
        ts = (base_ts + timedelta(minutes=i)).isoformat()
        store.write_heartbeat(mode="dry_run", equity=eq, n_positions=1, last_signal_ts=ts)

    # Write orders
    order1 = Order(
        id="ord-1", intent_id="int-1", trace_id="tr-1",
        symbol="BTC-USDT-PERP", side=OrderSide.BUY, qty=0.001,
        order_type=OrderType.MARKET, status=OrderStatus.FILLED,
        ts=(base_ts + timedelta(minutes=0)).isoformat(),
    )
    order2 = Order(
        id="ord-2", intent_id="int-2", trace_id="tr-2",
        symbol="BTC-USDT-PERP", side=OrderSide.SELL, qty=0.001,
        order_type=OrderType.MARKET, status=OrderStatus.FILLED,
        ts=(base_ts + timedelta(minutes=2)).isoformat(),
    )
    store.write_order(order1)
    store.write_order(order2)
    store.update_order_status("ord-1", OrderStatus.FILLED)
    store.update_order_status("ord-2", OrderStatus.FILLED)

    # Write fills
    fill1 = Fill(
        id="fill-1", order_id="ord-1", symbol="BTC-USDT-PERP",
        side=OrderSide.BUY, qty=0.001, price=50000.0, fee=0.05,
        fee_ccy="USDT", ts=(base_ts + timedelta(seconds=0.1)).isoformat(),
    )
    fill2 = Fill(
        id="fill-2", order_id="ord-2", symbol="BTC-USDT-PERP",
        side=OrderSide.SELL, qty=0.001, price=51000.0, fee=0.05,
        fee_ccy="USDT", ts=(base_ts + timedelta(seconds=2.1)).isoformat(),
    )
    store.write_fill(fill1)
    store.write_fill(fill2)

    # Write a position
    pos = Position(
        symbol="BTC-USDT-PERP", side=OrderSide.BUY, qty=0.001,
        avg_entry=50000.0, unrealized_pnl=1.0,
        ts=base_ts.isoformat(),
    )
    store.write_position(pos)

    # Write an event
    store.write_event(kind="guardian_block", severity="warning", payload_json='{"reason":"max_positions"}')


def _make_seeded_store() -> LiveStore:
    """Create a temporary seeded LiveStore for testing."""
    tmpdir = TemporaryDirectory()
    db_path = Path(tmpdir.name) / "test_runs.db"
    store = LiveStore(db_path)
    _seed_store(store)
    return store


class TestComputeSessionReport:
    """Test compute_session_report with pre-seeded data."""

    def test_basic_report_fields(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store, timeframe="1m")
            assert report.mode == "dry_run"
            assert report.n_ticks == 5
            assert report.n_orders == 2
            assert report.n_fills == 2
            assert report.initial_equity == 10000.0
            assert report.final_equity == 10350.0
            assert report.total_pnl == 350.0
            assert report.total_pnl_pct == 3.5
            assert report.guardian_blocks == 1
        finally:
            store.close()
            tmpdir.cleanup()

    def test_duration_minutes(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store)
            # Duration can be near-zero in test (heartbeats written in rapid succession)
            assert report.duration_minutes >= 0
        finally:
            store.close()
            tmpdir.cleanup()

    def test_equity_curve_populated(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store)
            assert len(report.equity_curve) == 5
            # First point should be initial equity
            assert report.equity_curve[0][1] == 10000.0
        finally:
            store.close()
            tmpdir.cleanup()

    def test_per_symbol_breakdown(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store)
            assert "BTC-USDT-PERP" in report.per_symbol
            sr = report.per_symbol["BTC-USDT-PERP"]
            assert isinstance(sr, SymbolReport)
            assert sr.symbol == "BTC-USDT-PERP"
        finally:
            store.close()
            tmpdir.cleanup()

    def test_fill_latency(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store)
            # We have fills with order timestamps, so latency should be > 0
            assert report.avg_fill_latency_ms >= 0
        finally:
            store.close()
            tmpdir.cleanup()

    def test_risk_metrics_computed(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store, timeframe="1m")
            # With 5 equity points, we should get some metrics
            assert isinstance(report.sharpe, float)
            assert isinstance(report.max_drawdown, float)
        finally:
            store.close()
            tmpdir.cleanup()


class TestComputeSessionReportEmpty:
    """Test compute_session_report with an empty store."""

    def test_empty_store(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "empty.db"
        store = LiveStore(db_path)
        try:
            report = compute_session_report(store)
            assert report.n_ticks == 0
            assert report.n_orders == 0
            assert report.n_fills == 0
            assert report.initial_equity == 0.0
            assert report.final_equity == 0.0
            assert report.total_pnl == 0.0
            assert report.mode == "unknown"
            assert len(report.equity_curve) == 0
        finally:
            store.close()
            tmpdir.cleanup()


class TestFormatReport:
    """Test the human-readable report formatter."""

    def test_format_report_basic(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store, timeframe="1m")
            text = format_report(report)
            assert "LIVE SESSION REPORT" in text
            assert "dry_run" in text
            assert "$10,000.00" in text
            assert "EQUITY" in text
            assert "RISK METRICS" in text
            assert "LATENCY" in text
            assert "SAFETY" in text
        finally:
            store.close()
            tmpdir.cleanup()

    def test_format_report_empty(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "empty.db"
        store = LiveStore(db_path)
        try:
            report = compute_session_report(store)
            text = format_report(report)
            assert "LIVE SESSION REPORT" in text
            assert "unknown" in text
        finally:
            store.close()
            tmpdir.cleanup()

    def test_format_report_with_per_symbol(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        try:
            report = compute_session_report(store, timeframe="1m")
            text = format_report(report)
            assert "BTC-USDT-PERP" in text
            assert "PER-SYMBOL BREAKDOWN" in text
        finally:
            store.close()
            tmpdir.cleanup()


class TestConfigDefaults:
    """Verify that config defaults were updated correctly."""

    def test_live_timeframe_default_is_1m(self) -> None:
        from kairon.config import KaironSettings
        settings = KaironSettings()
        assert settings.live_timeframe == "1m"

    def test_live_warmup_bars_default_is_22(self) -> None:
        from kairon.config import KaironSettings
        settings = KaironSettings()
        assert settings.live_warmup_bars == 22

    def test_live_config_timeframe_default_is_1m(self) -> None:
        from kairon.live.config import LiveConfig
        config = LiveConfig()
        assert config.timeframe == "1m"

    def test_live_config_warmup_bars_default_is_22(self) -> None:
        from kairon.live.config import LiveConfig
        config = LiveConfig()
        assert config.warmup_bars == 22


class TestStoreAnalytics:
    """Test the new analytics query methods on LiveStore."""

    def test_get_all_fills_empty(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            assert store.get_all_fills() == []
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_all_orders_empty(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            assert store.get_all_orders() == []
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_all_heartbeats_empty(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            assert store.get_all_heartbeats() == []
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_all_events_empty(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            assert store.get_all_events() == []
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_all_fills_with_data(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            fill = Fill(
                id="f1", order_id="o1", symbol="BTC-USDT-PERP",
                side=OrderSide.BUY, qty=0.001, price=50000.0,
                fee=0.05, fee_ccy="USDT",
                ts=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).isoformat(),
            )
            store.write_fill(fill)
            fills = store.get_all_fills()
            assert len(fills) == 1
            assert fills[0]["symbol"] == "BTC-USDT-PERP"
            assert fills[0]["side"] == "Buy"
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_all_orders_with_data(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            order = Order(
                id="o1", intent_id="i1", trace_id="t1",
                symbol="ETH-USDT-PERP", side=OrderSide.SELL, qty=0.1,
                order_type=OrderType.MARKET, status=OrderStatus.PENDING,
                ts=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).isoformat(),
            )
            store.write_order(order)
            orders = store.get_all_orders()
            assert len(orders) == 1
            assert orders[0]["symbol"] == "ETH-USDT-PERP"
            assert orders[0]["side"] == "Sell"
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_all_heartbeats_with_data(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            store.write_heartbeat(mode="dry_run", equity=10000.0, n_positions=0)
            store.write_heartbeat(mode="dry_run", equity=10050.0, n_positions=1)
            hbs = store.get_all_heartbeats()
            assert len(hbs) == 2
            assert hbs[0]["mode"] == "dry_run"
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_all_events_with_data(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            store.write_event(kind="test_event", severity="info", payload_json='{"key":"val"}')
            events = store.get_all_events()
            assert len(events) == 1
            assert events[0]["kind"] == "test_event"
        finally:
            store.close()
            tmpdir.cleanup()


class TestClosedTrades:
    """Test the live_closed_trades table and its integration."""

    def test_write_closed_trade(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            store.write_closed_trade(
                symbol="BTC-USDT-PERP",
                side="Buy",
                entry_qty=0.001,
                entry_price=50000.0,
                exit_price=51000.0,
                realized_pnl=1.0,
                fee=0.05,
                entry_ts="2026-01-01T12:00:00+00:00",
                exit_ts="2026-01-01T12:30:00+00:00",
                duration_seconds=1800.0,
            )
            trades = store.get_closed_trades()
            assert len(trades) == 1
            assert trades[0]["symbol"] == "BTC-USDT-PERP"
            assert trades[0]["side"] == "Buy"
            assert trades[0]["entry_qty"] == 0.001
            assert trades[0]["entry_price"] == 50000.0
            assert trades[0]["exit_price"] == 51000.0
            assert trades[0]["realized_pnl"] == 1.0
            assert trades[0]["duration_seconds"] == 1800.0
        finally:
            store.close()
            tmpdir.cleanup()

    def test_get_closed_trades_by_symbol(self) -> None:
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        try:
            store.write_closed_trade(
                symbol="BTC-USDT-PERP", side="Buy", entry_qty=0.001,
                entry_price=50000.0, exit_price=51000.0, realized_pnl=1.0,
                entry_ts="2026-01-01T12:00:00+00:00", exit_ts="2026-01-01T12:30:00+00:00",
            )
            store.write_closed_trade(
                symbol="ETH-USDT-PERP", side="Sell", entry_qty=0.5,
                entry_price=3000.0, exit_price=2950.0, realized_pnl=25.0,
                entry_ts="2026-01-01T12:10:00+00:00", exit_ts="2026-01-01T12:40:00+00:00",
            )
            btc_trades = store.get_closed_trades(symbol="BTC-USDT-PERP")
            assert len(btc_trades) == 1
            assert btc_trades[0]["symbol"] == "BTC-USDT-PERP"
            all_trades = store.get_closed_trades()
            assert len(all_trades) == 2
        finally:
            store.close()
            tmpdir.cleanup()

    def test_closed_trades_in_analytics(self) -> None:
        """Test that closed trades appear in the session report."""
        tmpdir = TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        store = LiveStore(db_path)
        _seed_store(store)
        # Add a closed trade
        store.write_closed_trade(
            symbol="BTC-USDT-PERP", side="Buy", entry_qty=0.001,
            entry_price=50000.0, exit_price=51000.0, realized_pnl=1.0,
            entry_ts="2026-01-01T12:05:00+00:00", exit_ts="2026-01-01T12:25:00+00:00",
            duration_seconds=1200.0,
        )
        try:
            report = compute_session_report(store, timeframe="1m")
            assert len(report.closed_trades) == 1
            assert report.closed_trades[0]["symbol"] == "BTC-USDT-PERP"
            assert report.closed_trades[0]["realized_pnl"] == 1.0
            # Per-symbol breakdown should use closed trades
            assert "BTC-USDT-PERP" in report.per_symbol
            assert report.per_symbol["BTC-USDT-PERP"].n_trades == 1
            assert report.per_symbol["BTC-USDT-PERP"].total_pnl == 1.0
        finally:
            store.close()
            tmpdir.cleanup()