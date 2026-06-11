"""Tests for the LSTM backend (skipped if torch isn't installed)."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.models.base import ModelError
from kairon.models.contracts import FeatureMatrix
from kairon.models.lstm import (
    LSTMConfig,
    LSTMModel,
    _make_predict_sequences,
    _make_sequences,
)

torch = pytest.importorskip("torch", reason="torch not installed")


def _toy(n: int = 200, seed: int = 17) -> tuple[FeatureMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 + 0.1 * rng.normal(size=n) > 0).astype(np.int64)
    fm = FeatureMatrix(
        values=np.column_stack([x1, x2]).astype(np.float64),
        feature_names=("x1", "x2"),
    )
    return fm, y


def test_lstm_config_validation() -> None:
    with pytest.raises(ValueError):
        LSTMConfig(sequence_length=1)
    with pytest.raises(ValueError):
        LSTMConfig(hidden_size=0)
    with pytest.raises(ValueError):
        LSTMConfig(num_layers=0)
    with pytest.raises(ValueError):
        LSTMConfig(dropout=-0.1)
    with pytest.raises(ValueError):
        LSTMConfig(dropout=1.0)
    with pytest.raises(ValueError):
        LSTMConfig(epochs=0)
    with pytest.raises(ValueError):
        LSTMConfig(batch_size=0)


def test_make_sequences_basic() -> None:
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.int64)
    xs, ys = _make_sequences(x, y, seq_len=4)
    assert xs.shape == (6, 4, 2)
    assert ys.shape == (6,)
    assert ys[0] == 4
    assert ys[-1] == 9


def test_make_sequences_too_short() -> None:
    x = np.zeros((3, 2), dtype=np.float64)
    y = np.array([0, 1, 0], dtype=np.int64)
    xs, ys = _make_sequences(x, y, seq_len=4)
    assert xs.shape == (1, 4, 2)
    assert ys.shape == (1,)


def test_make_predict_sequences() -> None:
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    xs = _make_predict_sequences(x, seq_len=4)
    assert xs.shape == (6, 4, 2)
    assert xs[0, -1, 0] == 3.0


def test_lstm_fit_predict_smoke() -> None:
    fm, y = _toy()
    m = LSTMModel(
        LSTMConfig(
            sequence_length=8,
            hidden_size=8,
            num_layers=1,
            epochs=3,
            patience=2,
            random_state=0,
        )
    )
    trained = m.fit(fm, y)
    pred = m.predict(trained, fm)
    # Predictions cover the "tail" of the feature matrix, not all rows
    assert pred.y_class.shape[0] == fm.n_rows - fm.n_rows + (fm.n_rows - 8)
    assert pred.y_proba is not None
    assert set(pred.y_class.tolist()).issubset(set(trained.classes or (0, 1)))


def test_lstm_rejects_too_few_classes() -> None:
    fm = FeatureMatrix(
        values=np.zeros((20, 2)), feature_names=("a", "b")
    )
    y = np.zeros(20, dtype=np.int64)
    m = LSTMModel(LSTMConfig(sequence_length=4, epochs=1, random_state=0))
    with pytest.raises(ModelError, match=">= 2 classes"):
        m.fit(fm, y)
