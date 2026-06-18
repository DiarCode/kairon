"""Trade decision journal: persist the full analysis context at decision time.

Every time the trading loop generates a signal, we snapshot the complete
indicator state, confidence, and corroboration rationale. This creates a
rich historical record that enables:

1. Post-hoc review of why trades were taken
2. Pattern analysis on winning vs losing decisions
3. Iterative improvement of strategy parameters
4. Avoiding repeated mistakes

The ``TradeDecision`` dataclass holds all fields; the ``IndicatorSnapshot``
dataclass groups the technical indicators for cleaner serialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Technical indicator values captured at decision time.

    All fields default to None so that partial snapshots are valid — not
    every strategy computes every indicator.
    """

    # Trend
    ema_fast: float | None = None
    ema_slow: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    adx: float | None = None
    bollinger_upper: float | None = None
    bollinger_mid: float | None = None
    bollinger_lower: float | None = None

    # Momentum
    rsi_14: float | None = None
    stochastic_k: float | None = None
    stochastic_d: float | None = None
    cci: float | None = None
    williams_r: float | None = None

    # Volatility
    atr_14: float | None = None
    garch_vol: float | None = None
    hurst_exp: float | None = None

    # Structure
    ew_wave_position: float | None = None
    ew_wave_direction: float | None = None
    ew_is_impulse: float | None = None
    ew_completion_prob: float | None = None
    ew_fib_confluence: float | None = None
    fib_dist_236: float | None = None
    fib_dist_382: float | None = None
    fib_dist_500: float | None = None
    fib_dist_618: float | None = None
    fib_dist_786: float | None = None
    fvg_bullish: float | None = None
    fvg_bearish: float | None = None
    fvg_fill_pct: float | None = None
    ob_in_bullish_zone: float | None = None
    ob_in_bearish_zone: float | None = None
    ob_bullish_near: float | None = None
    ob_bearish_near: float | None = None
    bos_direction: int | None = None  # 1=bullish, -1=bearish, 0=none
    choch_direction: int | None = None

    # Regime
    regime_prob_trending: float | None = None
    regime_prob_ranging: float | None = None
    regime_prob_volatile: float | None = None
    regime_prob_stressed: float | None = None

    # Volume
    obv: float | None = None
    vwap: float | None = None
    cvd: float | None = None
    volume_imbalance: float | None = None
    volume_vs_avg: float | None = None  # current volume / 20-bar avg

    # Support / Resistance
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    swing_high: float | None = None
    swing_low: float | None = None

    # Price context
    close: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    """Risk management context at decision time."""

    sl_price: float | None = None
    tp_price: float | None = None
    position_size_fraction: float | None = None
    equity_at_signal: float | None = None
    atr_distance_pct: float | None = None  # ATR as % of price


@dataclass(frozen=True, slots=True)
class TradeDecision:
    """Complete analysis snapshot at the moment a trading decision is made.

    This is the unit of the trade journal — one row per order intent.
    The full indicator state, signal confidence, and rationale are stored
    so we can review past decisions and learn from mistakes.
    """

    # Identity
    order_id: str
    symbol: str
    timestamp: str  # ISO UTC

    # Signal source
    strategy_name: str
    direction: float  # -1.0, 0.0, +1.0
    confidence: float  # 0.0 to 1.0
    magnitude: float
    volatility: float
    horizon: str

    # Confluence scores (from ComprehensiveStrategy)
    trend_score: float | None = None
    momentum_score: float | None = None
    structure_score: float | None = None
    volume_score: float | None = None

    # Full indicator snapshot
    indicators: IndicatorSnapshot = field(default_factory=IndicatorSnapshot)

    # Risk context
    risk: RiskSnapshot = field(default_factory=RiskSnapshot)

    # Corroboration justifications (human-readable reasons)
    justifications: tuple[str, ...] = ()

    # Outcome (filled after trade closes)
    outcome: str | None = None  # 'hit_tp', 'hit_sl', 'manual_close', 'reversed', 'timeout'
    outcome_pnl: float | None = None
    outcome_ts: str | None = None


def decision_to_row(decision: TradeDecision) -> dict[str, Any]:
    """Convert a TradeDecision to a flat dict suitable for SQLite storage.

    Indicators and risk are serialized as JSON strings; scalar fields
    are stored as native SQLite types.
    """
    return {
        "order_id": decision.order_id,
        "symbol": decision.symbol,
        "timestamp": decision.timestamp,
        "strategy_name": decision.strategy_name,
        "direction": decision.direction,
        "confidence": decision.confidence,
        "magnitude": decision.magnitude,
        "volatility": decision.volatility,
        "horizon": decision.horizon,
        "trend_score": decision.trend_score,
        "momentum_score": decision.momentum_score,
        "structure_score": decision.structure_score,
        "volume_score": decision.volume_score,
        "indicators_json": json.dumps(asdict(decision.indicators), default=_json_default),
        "risk_json": json.dumps(asdict(decision.risk), default=_json_default),
        "justifications_json": json.dumps(list(decision.justifications)),
        "outcome": decision.outcome,
        "outcome_pnl": decision.outcome_pnl,
        "outcome_ts": decision.outcome_ts,
    }


def row_to_decision(row: dict[str, Any]) -> TradeDecision:
    """Reconstruct a TradeDecision from a database row dict."""
    indicators_data = row.get("indicators_json", "{}")
    if isinstance(indicators_data, str):
        indicators_data = json.loads(indicators_data)
    risk_data = row.get("risk_json", "{}")
    if isinstance(risk_data, str):
        risk_data = json.loads(risk_data)
    justifications_data = row.get("justifications_json", "[]")
    if isinstance(justifications_data, str):
        justifications_data = json.loads(justifications_data)

    indicators = IndicatorSnapshot(
        **{k: v for k, v in indicators_data.items() if k in IndicatorSnapshot.__slots__}
    )
    risk = RiskSnapshot(
        **{k: v for k, v in risk_data.items() if k in RiskSnapshot.__slots__}
    )

    return TradeDecision(
        order_id=row["order_id"],
        symbol=row["symbol"],
        timestamp=row["timestamp"],
        strategy_name=row["strategy_name"],
        direction=row["direction"],
        confidence=row["confidence"],
        magnitude=row["magnitude"],
        volatility=row["volatility"],
        horizon=row["horizon"],
        trend_score=row.get("trend_score"),
        momentum_score=row.get("momentum_score"),
        structure_score=row.get("structure_score"),
        volume_score=row.get("volume_score"),
        indicators=indicators,
        risk=risk,
        justifications=tuple(justifications_data),
        outcome=row.get("outcome"),
        outcome_pnl=row.get("outcome_pnl"),
        outcome_ts=row.get("outcome_ts"),
    )


def _json_default(obj: Any) -> Any:
    """Handle non-serializable types in JSON dumps."""
    if obj is None:
        return None
    return str(obj)


__all__ = [
    "IndicatorSnapshot",
    "RiskSnapshot",
    "TradeDecision",
    "decision_to_row",
    "row_to_decision",
]
