"""Live inference: a thin runtime that wires features → model → prediction.

The :class:`LivePredictor` takes:

- a fitted :class:`kairon.models.base.TrainedModel`,
- a :class:`kairon.models.contracts.FeatureMatrix`,
- a ``live_window`` (the trailing slice to score on each tick),

and produces a stream of :class:`InferenceResult` objects
on every call. It also tracks the latency per call and counts any
errors so the alert engine can fire on stalls.

This module deliberately does *not* include a WebSocket adapter — the
caller feeds the predictor on whatever schedule they like (e.g. a
1-minute cron, an exchange websocket, etc.).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairon.models.base import Model, ModelError, Prediction, TrainedModel
from kairon.models.contracts import FeatureMatrix


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """The output of one :meth:`LivePredictor.predict` call."""

    y_class: np.ndarray
    y_proba: np.ndarray | None
    latency_ms: float
    timestamp_ns: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LivePredictorConfig:
    """Configuration for the live predictor."""

    latency_window: int = 100
    fail_on_missing_features: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


class LivePredictor:
    """Wraps a fitted model behind a latency-tracking API."""

    def __init__(
        self,
        model: Model[Any],
        trained: TrainedModel,
        config: LivePredictorConfig | None = None,
    ) -> None:
        if trained.backend != model.name:
            raise ModelError(
                f"trained model is {trained.backend!r}, "
                f"this model is {model.name!r}"
            )
        self.model = model
        self.trained = trained
        self.config = config or LivePredictorConfig()
        self._latencies_ms: deque[float] = deque(maxlen=self.config.latency_window)
        self._n_calls: int = 0
        self._n_errors: int = 0
        self._last_ts_ns: int = 0

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def n_errors(self) -> int:
        return self._n_errors

    @property
    def mean_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return float(np.mean(self._latencies_ms))

    @property
    def last_latency_ms(self) -> float:
        return self._latencies_ms[-1] if self._latencies_ms else 0.0

    def predict(
        self,
        features: FeatureMatrix,
    ) -> InferenceResult:
        """Run inference on ``features`` and return an :class:`InferenceResult`.

        Increments error counters instead of raising so the live loop
        stays running through transient failures.
        """
        if self.config.fail_on_missing_features:
            if features.feature_names != self.trained.feature_names:
                self._n_errors += 1
                raise ModelError(
                    f"feature mismatch: trained on {self.trained.feature_names}, "
                    f"got {features.feature_names}"
                )
        t0 = time.perf_counter()
        try:
            pred = self.model.predict(self.trained, features)
        except Exception:
            self._n_errors += 1
            raise
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies_ms.append(latency_ms)
        self._n_calls += 1
        self._last_ts_ns = time.time_ns()
        return InferenceResult(
            y_class=pred.y_class,
            y_proba=pred.y_proba,
            latency_ms=latency_ms,
            timestamp_ns=self._last_ts_ns,
            extras={"backend": pred.backend},
        )

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the runtime stats for the alert engine."""
        return {
            "n_calls": self.n_calls,
            "n_errors": self.n_errors,
            "mean_latency_ms": self.mean_latency_ms,
            "last_latency_ms": self.last_latency_ms,
        }


__all__ = [
    "InferenceResult",
    "LivePrediction",
    "LivePredictor",
    "LivePredictorAdapter",
    "LivePredictorConfig",
]


# ---------------------------------------------------------------------------
# LivePredictorAdapter — thin wrapper that converts Prediction → LivePrediction
# for the TradingLoop. Does NOT call run_analysis(); uses the pre-loaded
# TrainedModel's predict() method directly (cheap inference, no re-training).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LivePrediction:
    """The output of a per-bar inference call for the TradingLoop.

    Maps the model's raw Prediction (direction, probability, magnitude, vol)
    into a structured form that the sizer and guardian can consume.
    """

    symbol: str
    direction: float  # +1 (buy), -1 (sell), 0 (flat)
    magnitude: float  # predicted magnitude (e.g. log-return forecast)
    volatility: float  # predicted volatility (e.g. vol quantile forecast)
    confidence: float  # 0..1, derived from y_proba or ensemble weight
    horizon: str  # "day" | "swing" | "long"
    ts: str  # ISO-8601 UTC timestamp of the prediction
    justifications: tuple[str, ...] = ()  # human-readable reasons for the signal


class LivePredictorAdapter:
    """Wraps a pre-loaded TrainedModel and produces :class:`LivePrediction`
    on each closed bar.

    Unlike :func:`kairon.analysis.engine.run_analysis` (which re-trains
    models on every invocation), this adapter calls
    ``model.predict(trained, features)`` — a cheap inference call that
    takes milliseconds, not minutes.

    The adapter is constructed at boot time with a loaded TrainedModel.
    If the model is missing, it fails closed (raises ``ModelError``).
    """

    def __init__(
        self,
        model: Model[Any],
        trained: TrainedModel,
        *,
        horizon: str = "day",
        confidence_floor: float = 0.3,
    ) -> None:
        if trained.backend != model.name:
            raise ModelError(
                f"trained model is {trained.backend!r}, "
                f"this model is {model.name!r}"
            )
        self.model = model
        self.trained = trained
        self.horizon = horizon
        self.confidence_floor = confidence_floor
        self._n_calls: int = 0
        self._n_errors: int = 0

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def n_errors(self) -> int:
        return self._n_errors

    def predict(
        self,
        features: FeatureMatrix,
        *,
        symbol: str,
    ) -> LivePrediction:
        """Run inference on ``features`` and return a :class:`LivePrediction`.

        This is a thin wrapper around ``model.predict(trained, features)``
        that converts the raw Prediction into a TradingLoop-friendly format.
        """
        self._n_calls += 1
        try:
            pred = self.model.predict(self.trained, features)
        except Exception:
            self._n_errors += 1
            raise

        # Convert y_proba to confidence
        confidence = self._compute_confidence(pred)

        # Extract direction from y_class
        # y_class is (n,) int64 array; take the last element (most recent bar)
        direction = float(pred.y_class[-1])

        # Extract magnitude (predicted log-return or similar)
        # y_magnitude is W6.4 multi-head output; default to y_score if absent
        if pred.y_magnitude is not None:
            magnitude = float(pred.y_magnitude[-1])
        elif pred.y_score is not None:
            magnitude = float(pred.y_score[-1])
        else:
            magnitude = 0.0

        # Extract volatility (W6.4 vol head)
        volatility = float(pred.y_vol[-1]) if pred.y_vol is not None else 0.0

        return LivePrediction(
            symbol=symbol,
            direction=direction,
            magnitude=magnitude,
            volatility=volatility,
            confidence=confidence,
            horizon=self.horizon,
            ts=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            justifications=(),
        )

    def _compute_confidence(self, pred: Any) -> float:
        """Derive a confidence score from the prediction's probability output."""
        if pred.y_proba is not None and len(pred.y_proba.shape) == 2:
            # Binary or multi-class: max probability as confidence
            return float(max(pred.y_proba[-1]))
        if pred.y_proba is not None and len(pred.y_proba.shape) == 1:
            return float(pred.y_proba[-1])
        # Fallback: use absolute y_score as proxy
        if pred.y_score is not None:
            score = abs(float(pred.y_score[-1]))
            return min(1.0, score)
        return self.confidence_floor
