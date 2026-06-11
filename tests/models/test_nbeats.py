"""Tests for the N-BEATS backend (skipped if torch isn't installed)."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.models.base import ModelError
from kairon.models.contracts import FeatureMatrix
from kairon.models.nbeats import (
    NBEATSConfig,
    NBEATSModel,
    _make_predict_sequences_n,
    _make_sequences_n,
    _polynomial_basis,
)

torch = pytest.importorskip("torch", reason="torch not installed")


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


def test_nbeats_config_validation() -> None:
    with pytest.raises(ValueError):
        NBEATSConfig(lookback=1)
    with pytest.raises(ValueError):
        NBEATSConfig(horizon=0)
    with pytest.raises(ValueError):
        NBEATSConfig(hidden_size=0)
    with pytest.raises(ValueError):
        NBEATSConfig(n_blocks=0)
    with pytest.raises(ValueError):
        NBEATSConfig(n_layers=0)
    with pytest.raises(ValueError):
        NBEATSConfig(theta_dim=0)
    with pytest.raises(ValueError):
        NBEATSConfig(basis_degree=0)
    with pytest.raises(ValueError):
        NBEATSConfig(epochs=0)


def test_polynomial_basis_shape() -> None:
    b = _polynomial_basis(10, 3)
    assert b.shape == (10, 4)
    assert b[0, 0] == pytest.approx(1.0)


def test_make_sequences_n_basic() -> None:
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.int64)
    xs, ys, yw = _make_sequences_n(x, y, lookback=4, horizon=2)
    assert xs.shape == (6, 4, 2)
    assert ys.shape == (6,)
    assert yw.shape == (6, 2)
    assert ys[0] == 4
    assert ys[-1] == 9


def test_make_predict_sequences_n_basic() -> None:
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    xs, yw = _make_predict_sequences_n(x, lookback=4, horizon=2)
    assert xs.shape == (6, 4, 2)
    assert yw.shape == (6, 2)


def test_nbeats_fit_predict_smoke() -> None:
    fm, y = _toy()
    m = NBEATSModel(
        NBEATSConfig(
            lookback=8,
            horizon=1,
            hidden_size=8,
            n_blocks=2,
            n_layers=1,
            theta_dim=2,
            basis_degree=2,
            epochs=3,
            patience=2,
            random_state=0,
        )
    )
    trained = m.fit(fm, y)
    pred = m.predict(trained, fm)
    n_pred = fm.n_rows - 8
    assert pred.y_class.shape == (n_pred,)
    assert pred.y_proba is not None
    assert pred.y_score is not None  # N-BEATS returns a forecast as y_score
    assert pred.y_score.shape == (n_pred,)


def test_nbeats_rejects_too_few_classes() -> None:
    fm = FeatureMatrix(
        values=np.zeros((20, 2)), feature_names=("a", "b")
    )
    y = np.zeros(20, dtype=np.int64)
    m = NBEATSModel(NBEATSConfig(lookback=4, epochs=1, random_state=0))
    with pytest.raises(ModelError, match=">= 2 classes"):
        m.fit(fm, y)
