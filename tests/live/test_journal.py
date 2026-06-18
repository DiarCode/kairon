"""Tests for TradeDecision journal: construction, serialization, and LiveStore CRUD."""

from __future__ import annotations

from pathlib import Path

import pytest

from kairon.live.broker.base import Order, OrderSide, OrderStatus, OrderType
from kairon.live.journal import (
    IndicatorSnapshot,
    RiskSnapshot,
    TradeDecision,
    decision_to_row,
    row_to_decision,
)
from kairon.live.store import LiveStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "test_journal.db"


@pytest.fixture
def store(store_path: Path) -> LiveStore:
    s = LiveStore(store_path)
    yield s
    s.close()


def _make_decision(**overrides: object) -> TradeDecision:
    """Create a minimal TradeDecision with sensible defaults."""
    defaults = {
        "order_id": "ord-001",
        "symbol": "BTC-USDT-PERP",
        "timestamp": "2026-06-13T12:00:00+00:00",
        "strategy_name": "ComprehensiveStrategy",
        "direction": 1.0,
        "confidence": 0.72,
        "magnitude": 0.005,
        "volatility": 0.015,
        "horizon": "day",
        "justifications": ("Bullish EMA crossover", "RSI oversold (28.5)"),
    }
    defaults.update(overrides)
    return TradeDecision(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TradeDecision construction tests
# ---------------------------------------------------------------------------


class TestTradeDecision:
    """Test TradeDecision dataclass construction."""

    def test_minimal_construction(self) -> None:
        d = TradeDecision(
            order_id="ord-1",
            symbol="BTC-USDT-PERP",
            timestamp="2026-06-13T00:00:00+00:00",
            strategy_name="ComprehensiveStrategy",
            direction=1.0,
            confidence=0.65,
            magnitude=0.01,
            volatility=0.02,
            horizon="day",
        )
        assert d.order_id == "ord-1"
        assert d.direction == 1.0
        assert d.confidence == 0.65
        assert d.justifications == ()
        assert d.outcome is None
        assert d.indicators.rsi_14 is None

    def test_full_construction(self) -> None:
        indicators = IndicatorSnapshot(
            ema_fast=50001.0,
            ema_slow=49998.0,
            rsi_14=35.2,
            atr_14=250.0,
            macd_line=3.0,
            macd_signal=2.5,
            macd_histogram=0.5,
            adx=28.5,
            regime_prob_trending=0.6,
            regime_prob_ranging=0.2,
            regime_prob_volatile=0.15,
            regime_prob_stressed=0.05,
            close=50000.0,
        )
        risk = RiskSnapshot(
            sl_price=49500.0,
            tp_price=50750.0,
            equity_at_signal=10000.0,
        )
        d = TradeDecision(
            order_id="ord-2",
            symbol="ETH-USDT-PERP",
            timestamp="2026-06-13T01:00:00+00:00",
            strategy_name="ComprehensiveStrategy",
            direction=-1.0,
            confidence=0.80,
            magnitude=0.008,
            volatility=0.025,
            horizon="swing",
            trend_score=0.25,
            momentum_score=0.20,
            structure_score=0.15,
            volume_score=0.12,
            indicators=indicators,
            risk=risk,
            justifications=("Bearish EMA crossover", "MACD histogram negative"),
        )
        assert d.indicators.rsi_14 == 35.2
        assert d.risk.sl_price == 49500.0
        assert len(d.justifications) == 2
        assert d.trend_score == 0.25

    def test_frozen(self) -> None:
        d = _make_decision()
        with pytest.raises(AttributeError):
            d.direction = -1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------


class TestDecisionSerialization:
    """Test decision_to_row and row_to_decision round-trips."""

    def test_round_trip_minimal(self) -> None:
        d = _make_decision()
        row = decision_to_row(d)
        assert row["order_id"] == "ord-001"
        assert row["direction"] == 1.0
        assert row["confidence"] == 0.72
        assert row["strategy_name"] == "ComprehensiveStrategy"

        # Round-trip back
        restored = row_to_decision(row)
        assert restored.order_id == d.order_id
        assert restored.symbol == d.symbol
        assert restored.direction == d.direction
        assert restored.confidence == d.confidence
        assert restored.justifications == d.justifications

    def test_round_trip_full(self) -> None:
        indicators = IndicatorSnapshot(
            rsi_14=42.3,
            atr_14=180.5,
            macd_histogram=1.2,
            close=51000.0,
            regime_prob_trending=0.55,
        )
        risk = RiskSnapshot(
            sl_price=50600.0,
            tp_price=51540.0,
            equity_at_signal=10000.0,
        )
        d = TradeDecision(
            order_id="ord-full",
            symbol="BTC-USDT-PERP",
            timestamp="2026-06-13T02:00:00+00:00",
            strategy_name="MACrossoverStrategy",
            direction=1.0,
            confidence=0.55,
            magnitude=0.003,
            volatility=0.01,
            horizon="day",
            trend_score=0.15,
            momentum_score=0.10,
            indicators=indicators,
            risk=risk,
            justifications=("Bullish EMA crossover",),
        )
        row = decision_to_row(d)
        restored = row_to_decision(row)

        assert restored.indicators.rsi_14 == 42.3
        assert restored.indicators.atr_14 == 180.5
        assert restored.risk.sl_price == 50600.0
        assert restored.trend_score == 0.15
        assert restored.justifications == ("Bullish EMA crossover",)

    def test_json_fields_are_strings(self) -> None:
        d = _make_decision()
        row = decision_to_row(d)
        # indicators_json and risk_json should be valid JSON strings
        import json

        indicators = json.loads(row["indicators_json"])
        assert isinstance(indicators, dict)
        risk = json.loads(row["risk_json"])
        assert isinstance(risk, dict)
        justifications = json.loads(row["justifications_json"])
        assert isinstance(justifications, list)


# ---------------------------------------------------------------------------
# LiveStore decision CRUD tests
# ---------------------------------------------------------------------------


class TestLiveStoreDecisions:
    """Test write_decision, get_decisions, update_decision_outcome in LiveStore."""

    def test_write_and_get_decision(self, store: LiveStore) -> None:
        d = _make_decision()
        store.write_decision(d)
        decisions = store.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].order_id == "ord-001"
        assert decisions[0].strategy_name == "ComprehensiveStrategy"
        assert decisions[0].direction == 1.0
        assert decisions[0].confidence == 0.72

    def test_write_decision_with_indicators(self, store: LiveStore) -> None:
        indicators = IndicatorSnapshot(rsi_14=45.0, atr_14=200.0, close=50000.0)
        d = _make_decision(
            order_id="ord-ind",
            indicators=indicators,
        )
        store.write_decision(d)
        decisions = store.get_decisions()
        assert decisions[0].indicators.rsi_14 == 45.0
        assert decisions[0].indicators.atr_14 == 200.0

    def test_update_decision_outcome(self, store: LiveStore) -> None:
        d = _make_decision()
        store.write_decision(d)
        store.update_decision_outcome("ord-001", "hit_tp", pnl=150.0)
        decisions = store.get_decisions()
        assert decisions[0].outcome == "hit_tp"
        assert decisions[0].outcome_pnl == 150.0
        assert decisions[0].outcome_ts is not None

    def test_get_decisions_by_symbol(self, store: LiveStore) -> None:
        d1 = _make_decision(order_id="ord-a", symbol="BTC-USDT-PERP")
        d2 = _make_decision(order_id="ord-b", symbol="ETH-USDT-PERP")
        store.write_decision(d1)
        store.write_decision(d2)

        btc_decisions = store.get_decisions(symbol="BTC-USDT-PERP")
        assert len(btc_decisions) == 1
        assert btc_decisions[0].symbol == "BTC-USDT-PERP"

        eth_decisions = store.get_decisions(symbol="ETH-USDT-PERP")
        assert len(eth_decisions) == 1
        assert eth_decisions[0].symbol == "ETH-USDT-PERP"

    def test_get_decisions_by_outcome(self, store: LiveStore) -> None:
        d1 = _make_decision(order_id="ord-w")
        d2 = _make_decision(order_id="ord-l")
        store.write_decision(d1)
        store.write_decision(d2)
        store.update_decision_outcome("ord-w", "hit_tp", pnl=100.0)
        store.update_decision_outcome("ord-l", "hit_sl", pnl=-50.0)

        wins = store.get_decisions(outcome="hit_tp")
        assert len(wins) == 1
        assert wins[0].outcome == "hit_tp"

        losses = store.get_decisions(outcome="hit_sl")
        assert len(losses) == 1
        assert losses[0].outcome == "hit_sl"

    def test_write_decision_upsert(self, store: LiveStore) -> None:
        d1 = _make_decision(confidence=0.5)
        store.write_decision(d1)
        # Write again with same order_id (upsert)
        d2 = _make_decision(confidence=0.8)
        store.write_decision(d2)
        decisions = store.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].confidence == 0.8

    def test_multiple_decisions(self, store: LiveStore) -> None:
        for i in range(5):
            d = _make_decision(
                order_id=f"ord-{i}",
                confidence=0.5 + i * 0.1,
            )
            store.write_decision(d)

        decisions = store.get_decisions(limit=100)
        assert len(decisions) == 5