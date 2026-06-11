"""Deterministic, typed feature pipeline.

A ``FeaturePipeline`` is an ordered list of feature builders. The
pipeline is **deterministic**: given the same input table, the same
output is produced on every run. The pipeline is also **typed**:
each builder's input/output is a pyarrow Table; the column types
are checked at the boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pyarrow as pa

from kairon.features.registry import FeatureSpec, get

# We import the registration helpers here so the feature implementations
# are bound to the registry.  Each feature registers itself on import.
from kairon.features.registry import register as _register
from kairon.features.regime_features import (
    bocpd_changepoint,
    bocpd_regime_probabilities,
    bocpd_run_length,
)
from kairon.features.technical.autoregressive import lagged_returns, rolling_momentum
from kairon.features.technical.ichimoku_derived import ichimoku_derived
from kairon.features.technical.momentum import cci, rsi, stochastic, williams_r
from kairon.features.technical.structure import bos_choch, candlestick_patterns
from kairon.features.technical.temporal import hour_of_day
from kairon.features.technical.trend import adx, ema, ichimoku, macd, sma
from kairon.features.technical.volatility import atr, bollinger
from kairon.features.technical.volume import cvd, obv, vwap
from kairon.features.technical.elliott_wave import elliott_wave
from kairon.features.technical.fvg import fair_value_gap
from kairon.features.technical.garch import garch_variance
from kairon.features.technical.hurst import hurst_exponent
from kairon.features.technical.order_blocks import order_blocks
from kairon.features.technical.structure import fibonacci_proximity
from kairon.features.technical.structure_derived import swing_structure_features
from kairon.features.technical.volume_derived import vwap_deviation, volume_imbalance

_register(
    "trend.ema_5",
    description="EMA of close with period 5",
    category="trend",
)(
    lambda t: ema(t, period=5, source="close", out="ema_5_close")
)

_register(
    "trend.ema_50",
    description="EMA of close with period 50",
    category="trend",
)(
    lambda t: ema(t, period=50, source="close", out="ema_50")
)

_register(
    "trend.ema_200",
    description="EMA of close with period 200",
    category="trend",
)(
    lambda t: ema(t, period=200, source="close", out="ema_200")
)

_register(
    "trend.sma_20",
    description="SMA of close with period 20",
    category="trend",
)(
    lambda t: sma(t, period=20, source="close", out="sma_20")
)

_register(
    "trend.macd",
    description="MACD 12/26/9",
    category="trend",
)(macd)

_register(
    "trend.adx",
    description="ADX 14",
    category="trend",
)(adx)

_register(
    "trend.ichimoku",
    description="Ichimoku 9/26/52",
    category="trend",
)(ichimoku)

_register("momentum.rsi_14", description="RSI 14", category="momentum")(
    lambda t: rsi(t, period=14, source="close")
)
_register("momentum.stochastic", description="Stochastic 14/3/3", category="momentum")(
    lambda t: stochastic(t)
)
_register("momentum.williams_r", description="Williams %R 14", category="momentum")(
    lambda t: williams_r(t, period=14)
)
_register("momentum.cci", description="CCI 20", category="momentum")(
    lambda t: cci(t, period=20)
)
_register("volatility.bollinger", description="Bollinger Bands 20/2", category="volatility")(
    lambda t: bollinger(t, period=20, std_dev=2.0)
)
_register("volatility.atr_14", description="ATR 14", category="volatility")(
    lambda t: atr(t, period=14)
)
_register("volume.obv", description="On-Balance Volume", category="volume")(obv)
_register("volume.vwap", description="Cumulative VWAP", category="volume")(vwap)
_register("volume.cvd", description="Cumulative Volume Delta (proxy)", category="volume")(cvd)
_register("structure.bos_choch", description="BOS / CHoCH 5-bar order", category="structure")(
    lambda t: bos_choch(t, order=5)
)
_register(
    "structure.candlestick",
    description="Doji / hammer / engulfing patterns",
    category="structure",
)(candlestick_patterns)

# ---------------------------------------------------------------------------
# Phase 1 feature registrations (quick-win additions)
# ---------------------------------------------------------------------------

# 1A: BOCPD regime probabilities, run-length, changepoint
_register(
    "regime.bocpd_probs",
    description="BOCPD regime probabilities (trending/ranging/volatile/stressed)",
    category="regime",
)(bocpd_regime_probabilities)
_register(
    "regime.bocpd_run_length",
    description="BOCPD run-length mean and MAP",
    category="regime",
)(bocpd_run_length)
_register(
    "regime.bocpd_changepoint",
    description="BOCPD changepoint detection binary",
    category="regime",
)(bocpd_changepoint)

# 1B: Ichimoku-derived signals
_register(
    "trend.ichimoku_derived",
    description="Ichimoku cloud position, TK cross, cloud twist, chikou displacement",
    category="trend",
)(ichimoku_derived)

# 1C: Autoregressive returns
_register(
    "technical.lagged_returns",
    description="Log-returns at lags 1/2/3/5/10/20",
    category="technical",
)(lagged_returns)
_register(
    "technical.rolling_momentum",
    description="5d/20d cumulative returns and z-scored momentum",
    category="technical",
)(rolling_momentum)

# 1D: VWAP deviation and volume imbalance
_register(
    "volume.vwap_deviation",
    description="VWAP percent deviation and z-score",
    category="volume",
)(vwap_deviation)
_register(
    "volume.volume_imbalance",
    description="BVC volume imbalance and relative volume",
    category="volume",
)(volume_imbalance)

# 1E: Temporal features (hour-of-day, trading sessions)
_register(
    "temporal.hour_of_day",
    description="Cyclical hour encoding and trading session indicators",
    category="temporal",
)(hour_of_day)

# ---------------------------------------------------------------------------
# Phase 2 feature registrations (Elliott Wave + structural)
# ---------------------------------------------------------------------------

# 2A: Elliott Wave detection engine
_register(
    "structure.elliott_wave",
    description="Elliott Wave position, direction, Fib confluence, completion prob",
    category="structure",
)(elliott_wave)

# 2B: Fibonacci level proximity (ATR-normalized distances)
_register(
    "structure.fibonacci_proximity",
    description="ATR-normalized distance to Fibonacci retracement levels",
    category="structure",
)(fibonacci_proximity)

# 2C: Fair Value Gap detection
_register(
    "structure.fvg",
    description="Fair Value Gap detection (bullish/bearish, fill pct, nearest distance)",
    category="structure",
)(fair_value_gap)

# 2D: Order Block detection
_register(
    "structure.order_blocks",
    description="Order block proximity and zone membership",
    category="structure",
)(order_blocks)

# 2E: Swing structure derived features
_register(
    "structure.swing_derived",
    description="Distance to swing high/low, swing range pct, structure break strength",
    category="structure",
)(swing_structure_features)

# ---------------------------------------------------------------------------
# Phase 3 feature registrations (volatility models)
# ---------------------------------------------------------------------------

# 3B: GARCH(1,1) conditional variance
_register(
    "volatility.garch",
    description="GARCH(1,1) conditional variance and volatility",
    category="volatility",
)(garch_variance)

# 3C: Hurst exponent
_register(
    "volatility.hurst",
    description="Hurst exponent via rolling R/S analysis",
    category="volatility",
)(hurst_exponent)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Result of a feature pipeline run."""

    table: pa.Table
    feature_names: tuple[str, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class Provenance:
    """Reproducibility record for a pipeline run."""

    input_hash: str
    pipeline_hash: str
    feature_names: tuple[str, ...]


def _hash_table(table: pa.Table) -> str:
    h = hashlib.sha256()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    h.update(sink.getvalue().to_pybytes())
    return h.hexdigest()


def _hash_pipeline(specs: Sequence[FeatureSpec]) -> str:
    h = hashlib.sha256()
    for spec in specs:
        # include name + qualname of the underlying builder function
        fn = spec.builder
        qualname = getattr(fn, "__qualname__", repr(fn))
        h.update(f"{spec.name}|{qualname}|".encode())
    return h.hexdigest()


class FeaturePipeline:
    """Ordered, deterministic pipeline of feature builders."""

    def __init__(self, features: Sequence[str] | None = None) -> None:
        if features is None:
            features = tuple(s.name for s in get("trend.ema_5").__class__.__mro__[0].__module__ and ())  # type: ignore[func-returns-value]
        # Default: a curated set covering all categories
        if not features:
            features = (
                "trend.ema_5",
                "trend.ema_50",
                "trend.ema_200",
                "trend.sma_20",
                "trend.macd",
                "trend.adx",
                "momentum.rsi_14",
                "momentum.stochastic",
                "momentum.williams_r",
                "momentum.cci",
                "volatility.bollinger",
                "volatility.atr_14",
                "volume.obv",
                "volume.vwap",
                "volume.cvd",
                "structure.bos_choch",
                "structure.candlestick",
            )
        specs = tuple(get(name) for name in features)
        self._specs: Final[tuple[FeatureSpec, ...]] = specs
        self._hash: Final[str] = _hash_pipeline(specs)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._specs)

    @property
    def pipeline_hash(self) -> str:
        return self._hash

    def run(self, table: pa.Table) -> PipelineResult:
        """Run the pipeline; return the new table plus provenance."""
        if "ts" not in table.column_names:
            raise ValueError("input must contain a 'ts' column")
        cur = table
        for spec in self._specs:
            cur = spec.builder(cur)
        return PipelineResult(
            table=cur,
            feature_names=self.feature_names,
            provenance=Provenance(
                input_hash=_hash_table(table),
                pipeline_hash=self._hash,
                feature_names=self.feature_names,
            ),
        )
