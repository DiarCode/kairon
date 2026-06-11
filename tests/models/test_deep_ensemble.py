"""Tests for the DeepEnsemble backend (always available, but skips torch
constituents if torch is not installed).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from kairon.models.contracts import FeatureMatrix
from kairon.models.deep_ensemble import DeepEnsemble, DeepEnsembleConfig
from kairon.models.ensemble import EnsembleSpec


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None  # type: ignore[attr-defined]


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


def test_deep_ensemble_config_validation() -> None:
    with pytest.raises(ValueError):
        DeepEnsembleConfig(lookback=1)


def test_deep_ensemble_tabular_only() -> None:
    """Always-available constituents: LR + RF. Should fit and predict."""
    fm, y = _toy()
    cfg = DeepEnsembleConfig(
        lookback=4,
        include_mlp=False,
        include_lstm=False,
        include_nbeats=False,
        include_rf=True,
        include_lr=True,
    )
    e = DeepEnsemble(cfg)
    assert e.n_constituents == 2
    assert "logreg" in e.constituent_names
    assert "random_forest" in e.constituent_names
    trained = e.fit(fm, y)
    pred = e.predict(trained, fm)
    assert pred.y_class.shape == (fm.n_rows,)
    assert pred.y_proba is not None
    assert pred.y_proba.shape[0] == fm.n_rows


def test_deep_ensemble_requires_one_constituent() -> None:
    cfg = DeepEnsembleConfig(
        include_mlp=False,
        include_lstm=False,
        include_nbeats=False,
        include_rf=False,
        include_lr=False,
    )
    with pytest.raises(ValueError, match="at least one"):
        DeepEnsemble(cfg)


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_deep_ensemble_with_torch_constituents() -> None:
    fm, y = _toy(n=120, seed=11)
    cfg = DeepEnsembleConfig(
        lookback=4,
        include_mlp=True,
        include_lstm=True,
        include_nbeats=True,
        include_rf=True,
        include_lr=True,
        spec=EnsembleSpec(min_k=1, max_k=4, confidence_floor=0.34),
    )
    e = DeepEnsemble(cfg)
    assert e.n_constituents == 5
    trained = e.fit(fm, y)
    pred = e.predict(trained, fm)
    assert pred.y_class.shape == (fm.n_rows,)
    assert pred.y_proba is not None
