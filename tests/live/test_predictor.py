"""Tests for the live predictor."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.live import InferenceResult, LivePredictor, LivePredictorConfig
from kairon.models.base import ModelError
from kairon.models.contracts import FeatureMatrix
from kairon.models.linear import LinearConfig, LogisticRegressionModel


def _toy(n: int = 200, seed: int = 17) -> tuple[FeatureMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.5 * x2 + 0.05 * rng.normal(size=n) > 0).astype(np.int64)
    fm = FeatureMatrix(
        values=np.column_stack([x1, x2]).astype(np.float64),
        feature_names=("x1", "x2"),
    )
    return fm, y


def test_predictor_validates_backend() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    other = LinearConfig(C=2.0)
    other_model = LogisticRegressionModel(other)
    other_model.name = "logreg_v2"  # force a backend mismatch
    with pytest.raises(ModelError):
        LivePredictor(other_model, trained)


def test_predictor_runs_and_returns_result() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    p = LivePredictor(m, trained)
    res = p.predict(fm)
    assert isinstance(res, InferenceResult)
    assert res.y_class.shape == (fm.n_rows,)
    assert res.y_proba is not None
    assert res.latency_ms >= 0
    assert res.timestamp_ns > 0
    assert p.n_calls == 1
    assert p.n_errors == 0


def test_predictor_tracks_latency() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    p = LivePredictor(m, trained, LivePredictorConfig(latency_window=10))
    for _ in range(5):
        p.predict(fm)
    assert p.n_calls == 5
    assert p.mean_latency_ms >= 0
    stats = p.stats()
    assert stats["n_calls"] == 5
    assert stats["n_errors"] == 0


def test_predictor_raises_on_feature_mismatch() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    p = LivePredictor(m, trained)
    bad = FeatureMatrix(values=fm.values, feature_names=("x1", "x3"))
    with pytest.raises(ModelError):
        p.predict(bad)
    assert p.n_errors == 1


def test_predictor_can_disable_strict_check() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    p = LivePredictor(
        m, trained, LivePredictorConfig(fail_on_missing_features=False)
    )
    # Even with mismatch we don't raise; we just count it as an error.
    bad = FeatureMatrix(values=fm.values, feature_names=("x1", "x3"))
    with pytest.raises(ModelError):
        p.predict(bad)  # underlying model.predict still raises
    assert p.n_errors == 1


def test_predictor_empty_state() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig())
    trained = m.fit(fm, y)
    p = LivePredictor(m, trained)
    assert p.n_calls == 0
    assert p.mean_latency_ms == 0
    assert p.last_latency_ms == 0
