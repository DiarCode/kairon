"""Typed feature registry.

The registry is a list of named feature builders. Each builder
takes a pyarrow Table and returns a Table with one or more new
columns appended. The pipeline (in :mod:`kairon.features.pipeline`)
runs them in order; the order matters when one feature depends on
another (e.g., ATR is needed by the regime classifier).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pyarrow as pa

FeatureBuilder = Callable[[pa.Table], pa.Table]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A registered feature builder."""

    name: str
    description: str
    builder: FeatureBuilder
    category: str


_REGISTRY: list[FeatureSpec] = []


def register(
    name: str,
    *,
    description: str,
    category: str = "general",
) -> Callable[[FeatureBuilder], FeatureBuilder]:
    """Decorator that adds a feature builder to the registry."""

    def _dec(fn: FeatureBuilder) -> FeatureBuilder:
        _REGISTRY.append(
            FeatureSpec(name=name, description=description, builder=fn, category=category)
        )
        return fn

    return _dec


def all_features() -> tuple[FeatureSpec, ...]:
    """Return all registered features (in registration order)."""
    return tuple(_REGISTRY)


def get(name: str) -> FeatureSpec:
    """Look up a feature by name. Raises ``KeyError`` if missing."""
    for spec in _REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown feature: {name!r}")


# Default feature set used by ``FeaturePipeline`` when no overrides
# are provided.  Each entry is a registered ``FeatureSpec`` name.
DEFAULT_FEATURES: Final[tuple[str, ...]] = (
    # trend
    "trend.ema_5",
    "trend.ema_50",
    "trend.ema_200",
    "trend.macd",
    "trend.adx",
    "momentum.rsi_14",
    "volatility.bollinger",
    "volatility.atr_14",
    "volume.obv",
    "volume.vwap",
    "structure.bos_choch",
    "structure.candlestick",
)

# Phase 1 quick-win features (BOCPD regime, Ichimoku derived,
# autoregressive returns, VWAP deviation, volume imbalance, temporal)
PHASE1_FEATURES: Final[tuple[str, ...]] = (
    "regime.bocpd_probs",
    "regime.bocpd_run_length",
    "regime.bocpd_changepoint",
    "trend.ichimoku",
    "trend.ichimoku_derived",
    "technical.lagged_returns",
    "technical.rolling_momentum",
    "volume.vwap_deviation",
    "volume.volume_imbalance",
    "temporal.hour_of_day",
)

# Phase 2 structural features (Elliott Wave, Fibonacci, FVG, order blocks,
# swing derived)
PHASE2_FEATURES: Final[tuple[str, ...]] = (
    "structure.elliott_wave",
    "structure.fibonacci_proximity",
    "structure.fvg",
    "structure.order_blocks",
    "structure.swing_derived",
)

# Phase 3 volatility features (GARCH, Hurst exponent)
PHASE3_FEATURES: Final[tuple[str, ...]] = (
    "volatility.garch",
    "volatility.hurst",
)

# Full feature set: baseline + all phases
ALL_FEATURES: Final[tuple[str, ...]] = (
    *DEFAULT_FEATURES,
    *PHASE1_FEATURES,
    *PHASE2_FEATURES,
    *PHASE3_FEATURES,
)


__all__ = [
    "ALL_FEATURES",
    "DEFAULT_FEATURES",
    "PHASE1_FEATURES",
    "PHASE2_FEATURES",
    "PHASE3_FEATURES",
    "FeatureBuilder",
    "FeatureSpec",
    "all_features",
    "get",
    "register",
]
