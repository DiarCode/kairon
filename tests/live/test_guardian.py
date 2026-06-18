"""Tests for the Guardian: risk checks, kill switch, and cooldowns."""

from __future__ import annotations

import time

import pytest

from kairon.live.alerts import AlertEngine, InMemoryChannel, Severity
from kairon.live.broker.base import OrderSide, Position
from kairon.live.guardian import CooldownFact, DailyPnlFact, Guardian, PositionsFact


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


# ---------------------------------------------------------------------------
# Positions checks
# ---------------------------------------------------------------------------


class TestGuardianPositions:
    """Test Guardian position limit, leverage, and count checks."""

    def test_no_alert_when_within_limits(self) -> None:
        guardian = Guardian(max_open_positions=5, max_position_equity_fraction=0.20)
        positions = [_pos(qty=0.001, avg_entry=50000.0)]  # $50 notional, 0.5% of $10k
        alerts = guardian.check_positions(positions, equity=10_000.0)
        assert len(alerts) == 0

    def test_alert_on_too_many_positions(self) -> None:
        guardian = Guardian(max_open_positions=2)
        positions = [
            _pos(symbol="BTC-USDT-PERP"),
            _pos(symbol="ETH-USDT-PERP"),
            _pos(symbol="SOL-USDT-PERP"),
        ]
        alerts = guardian.check_positions(positions, equity=10_000.0)
        critical = [a for a in alerts if a.severity == Severity.CRITICAL]
        assert len(critical) >= 1
        assert "Too many open positions" in critical[0].message

    def test_alert_on_position_equity_cap_breach(self) -> None:
        guardian = Guardian(max_position_equity_fraction=0.20)
        # $3000 notional on $10k equity = 30%, exceeds 20% cap
        positions = [_pos(qty=0.06, avg_entry=50000.0)]  # $3000 notional
        alerts = guardian.check_positions(positions, equity=10_000.0)
        critical = [a for a in alerts if a.severity == Severity.CRITICAL]
        assert len(critical) >= 1
        assert "position_cap" in critical[0].source

    def test_alert_on_total_leverage_breach(self) -> None:
        guardian = Guardian(max_total_leverage=1.0)
        # Two positions: $6000 + $5000 = $11000 total notional on $10k = 1.1x
        positions = [
            _pos(symbol="BTC-USDT-PERP", qty=0.12, avg_entry=50000.0),  # $6000
            _pos(symbol="ETH-USDT-PERP", qty=0.20, avg_entry=25000.0),  # $5000
        ]
        alerts = guardian.check_positions(positions, equity=10_000.0)
        critical = [a for a in alerts if a.severity == Severity.CRITICAL]
        leverage_alerts = [a for a in critical if "leverage" in a.source]
        assert len(leverage_alerts) >= 1

    def test_no_alert_when_leverage_within_limit(self) -> None:
        guardian = Guardian(max_total_leverage=1.0)
        # $8000 total notional on $10k equity = 0.8x leverage
        positions = [_pos(qty=0.08, avg_entry=50000.0)]  # $4000
        alerts = guardian.check_positions(positions, equity=10_000.0)
        leverage_alerts = [a for a in alerts if "leverage" in a.source]
        assert len(leverage_alerts) == 0

    def test_zero_equity_returns_no_alerts(self) -> None:
        guardian = Guardian()
        alerts = guardian.check_positions([], equity=0.0)
        assert len(alerts) == 0

    def test_empty_positions_within_limits(self) -> None:
        guardian = Guardian()
        alerts = guardian.check_positions([], equity=10_000.0)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Daily-loss kill switch
# ---------------------------------------------------------------------------


class TestGuardianDailyLoss:
    """Test Guardian daily-loss kill switch."""

    def test_no_alert_when_within_limit(self) -> None:
        guardian = Guardian(max_daily_loss_pct=0.03)
        # -$200 loss on $10k equity = 2%, below 3% limit
        alert = guardian.check_daily_loss(daily_pnl=-200.0, equity=10_000.0)
        assert alert is None

    def test_alert_on_daily_loss_breach(self) -> None:
        guardian = Guardian(max_daily_loss_pct=0.03)
        # -$400 loss on $10k equity = 4%, exceeds 3% limit
        alert = guardian.check_daily_loss(daily_pnl=-400.0, equity=10_000.0)
        assert alert is not None
        assert alert.severity == Severity.CRITICAL
        assert "daily_loss" in alert.source

    def test_no_alert_on_profit(self) -> None:
        guardian = Guardian(max_daily_loss_pct=0.03)
        # Positive PnL, should never trigger
        alert = guardian.check_daily_loss(daily_pnl=500.0, equity=10_000.0)
        assert alert is None

    def test_halts_store_on_breach(self) -> None:
        """When a store is provided, check_daily_loss calls halt()."""
        from pathlib import Path
        from kairon.live.store import LiveStore
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LiveStore(Path(tmpdir) / "test.db")
            guardian = Guardian(max_daily_loss_pct=0.03, store=store)
            # -$400 on $10k = 4% > 3% limit → should halt
            alert = guardian.check_daily_loss(daily_pnl=-400.0, equity=10_000.0)
            assert alert is not None
            assert store.is_halted()
            store.close()

    def test_zero_equity_returns_none(self) -> None:
        guardian = Guardian()
        alert = guardian.check_daily_loss(daily_pnl=-100.0, equity=0.0)
        assert alert is None


# ---------------------------------------------------------------------------
# Cooldown management
# ---------------------------------------------------------------------------


class TestGuardianCooldown:
    """Test per-symbol cooldown after stop-loss."""

    def test_not_cooling_down_initially(self) -> None:
        guardian = Guardian(cooldown_seconds=3600)
        assert not guardian.is_cooling_down("BTC-USDT-PERP")

    def test_cooldown_after_sl(self) -> None:
        guardian = Guardian(cooldown_seconds=3600)
        now = time.time()
        guardian.record_sl("BTC-USDT-PERP", now=now)
        assert guardian.is_cooling_down("BTC-USDT-PERP", now=now)
        assert guardian.is_cooling_down("BTC-USDT-PERP", now=now + 1800)  # 30min later
        assert not guardian.is_cooling_down("BTC-USDT-PERP", now=now + 3700)  # past cooldown

    def test_cooldown_does_not_affect_other_symbols(self) -> None:
        guardian = Guardian(cooldown_seconds=3600)
        now = time.time()
        guardian.record_sl("BTC-USDT-PERP", now=now)
        assert not guardian.is_cooling_down("ETH-USDT-PERP")

    def test_cooldown_remaining(self) -> None:
        guardian = Guardian(cooldown_seconds=3600)
        now = time.time()
        guardian.record_sl("BTC-USDT-PERP", now=now)
        remaining = guardian.cooldown_remaining("BTC-USDT-PERP", now=now + 1000)
        assert abs(remaining - 2600) < 1  # ~2600s remaining

    def test_cooldown_remaining_no_sl(self) -> None:
        guardian = Guardian(cooldown_seconds=3600)
        assert guardian.cooldown_remaining("BTC-USDT-PERP") == 0.0


# ---------------------------------------------------------------------------
# AlertEngine integration
# ---------------------------------------------------------------------------


class TestGuardianAlertEngineIntegration:
    """Test Guardian as an AlertEngine.Rule via fact-based evaluation."""

    def test_positions_fact_through_alert_engine(self) -> None:
        guardian = Guardian(max_open_positions=2)
        engine = AlertEngine(rules=[guardian], channels=[InMemoryChannel()])
        fact = PositionsFact(
            positions=(
                _pos(symbol="BTC-USDT-PERP"),
                _pos(symbol="ETH-USDT-PERP"),
                _pos(symbol="SOL-USDT-PERP"),
            ),
            equity=10_000.0,
        )
        alerts = engine.evaluate(fact)
        assert len(alerts) >= 1
        assert any("Too many" in a.message for a in alerts)

    def test_daily_pnl_fact_through_alert_engine(self) -> None:
        guardian = Guardian(max_daily_loss_pct=0.03)
        engine = AlertEngine(rules=[guardian], channels=[InMemoryChannel()])
        fact = DailyPnlFact(daily_pnl=-500.0, equity=10_000.0)
        alerts = engine.evaluate(fact)
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.CRITICAL

    def test_cooldown_fact_through_alert_engine(self) -> None:
        guardian = Guardian(cooldown_seconds=3600)
        engine = AlertEngine(rules=[guardian], channels=[InMemoryChannel()])
        now = time.time()
        guardian.record_sl("BTC-USDT-PERP", now=now)
        fact = CooldownFact(symbol="BTC-USDT-PERP", now=now + 100)
        alerts = engine.evaluate(fact)
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.WARNING

    def test_unknown_fact_returns_none(self) -> None:
        guardian = Guardian()
        result = guardian.matches("not a fact")
        assert result is None