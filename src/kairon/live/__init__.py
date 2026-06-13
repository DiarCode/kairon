"""Live inference and trading: broker protocol, store, config, and predictor.

The :class:`LivePredictor` wraps a fitted model behind a latency-tracking
API. The :class:`Broker` protocol and its implementations handle order
execution. :class:`LiveStore` persists trading state. :class:`LiveConfig`
bundles live-trading parameters. The :class:`Guardian` enforces risk limits
and the :class:`Reconciler` detects position drift.

Heavy imports (models, numpy) are deferred to avoid pulling in the entire
ML stack when only broker or store functionality is needed.
"""

from __future__ import annotations

from kairon.live.analytics import (
    LiveSessionReport,
    SymbolReport,
    compute_session_report,
    format_report,
)
from kairon.live.broker import (
    Balance,
    Broker,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.config import LiveConfig
from kairon.live.guardian import CooldownFact, DailyPnlFact, Guardian, PositionsFact
from kairon.live.journal import IndicatorSnapshot, RiskSnapshot, TradeDecision
from kairon.live.orchestrator import TradingLoop
from kairon.live.predictor import (
    InferenceResult,
    LivePrediction,
    LivePredictor,
    LivePredictorAdapter,
    LivePredictorConfig,
)
from kairon.live.reconciler import DriftFact, OrphanFact, Reconciler
from kairon.live.store import LiveStore
from kairon.live.strategy import (
    ComprehensiveStrategy,
    MACrossoverStrategy,
    MomentumStrategy,
    SignalStrategy,
)

__all__ = [
    # Broker
    "Broker",
    "Order",
    "Fill",
    "Position",
    "Balance",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    # Config
    "LiveConfig",
    # Guardian
    "Guardian",
    "PositionsFact",
    "DailyPnlFact",
    "CooldownFact",
    # Orchestrator
    "TradingLoop",
    # Predictor
    "InferenceResult",
    "LivePredictor",
    "LivePredictorConfig",
    "LivePrediction",
    "LivePredictorAdapter",
    # Reconciler
    "Reconciler",
    "DriftFact",
    "OrphanFact",
    # Strategy
    "MACrossoverStrategy",
    "MomentumStrategy",
    "ComprehensiveStrategy",
    "SignalStrategy",
    # Journal
    "TradeDecision",
    "IndicatorSnapshot",
    "RiskSnapshot",
    # Analytics
    "LiveSessionReport",
    "SymbolReport",
    "compute_session_report",
    "format_report",
    # Store
    "LiveStore",
]
