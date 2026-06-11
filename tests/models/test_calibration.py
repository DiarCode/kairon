"""Tests for isotonic + Platt calibration."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.models.base import ModelError
from kairon.models.calibration import (
    IsotonicCalibrator,
    IsotonicConfig,
    PlattCalibrator,
    PlattConfig,
    calibrated_proba,
)
from kairon.models.contracts import FeatureMatrix


def _overconfident_proba() -> tuple[FeatureMatrix, np.ndarray]:
    """Score column that's an overconfident scaling of a binary outcome."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.0, 1.0, size=300)
    y = (p + 0.2 * rng.normal(size=p.size) > 0.5).astype(np.int64)
    # Distort the score so it's "overconfident" (sigmoid push)
    score = 1.0 / (1.0 + np.exp(-(p - 0.5) * 6.0))  # sharp, but still 0..1
    fm = FeatureMatrix(values=score.reshape(-1, 1), feature_names=("score",))
    return fm, y


def test_isotonic_basic() -> None:
    fm, y = _overconfident_proba()
    cal = IsotonicCalibrator()
    trained = cal.fit(fm, y)
    pred = cal.predict(trained, fm)
    assert pred.y_proba is not None
    assert pred.y_proba.shape == (fm.n_rows,)
    # Predicted probs are in (y_min, y_max) (clipped)
    assert (pred.y_proba >= 0.0).all()
    assert (pred.y_proba <= 1.0).all()


def test_isotonic_rejects_multi_column() -> None:
    fm = FeatureMatrix(values=np.zeros((10, 2)), feature_names=("a", "b"))
    cal = IsotonicCalibrator()
    with pytest.raises(ModelError, match="1 score column"):
        cal.fit(fm, np.zeros(10, dtype=np.int64))


def test_isotonic_validates_config() -> None:
    with pytest.raises(ValueError):
        IsotonicConfig(out_of_bounds="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        IsotonicConfig(y_min=0.6, y_max=0.4)


def test_platt_basic() -> None:
    fm, y = _overconfident_proba()
    cal = PlattCalibrator(PlattConfig(C=1.0, max_iter=200))
    trained = cal.fit(fm, y)
    pred = cal.predict(trained, fm)
    assert pred.y_proba is not None
    assert pred.y_proba.shape == (fm.n_rows,)


def test_platt_rejects_multi_column() -> None:
    fm = FeatureMatrix(values=np.zeros((10, 2)), feature_names=("a", "b"))
    cal = PlattCalibrator()
    with pytest.raises(ModelError, match="1 score column"):
        cal.fit(fm, np.zeros(10, dtype=np.int64))


def test_calibrated_proba_helper() -> None:
    from kairon.models.base import Prediction

    fm, y = _overconfident_proba()
    cal = IsotonicCalibrator()
    trained = cal.fit(fm, y)
    pred = Prediction(
        y_class=np.zeros(fm.n_rows, dtype=np.int64),
        y_proba=fm.values[:, 0],
    )
    out = calibrated_proba(pred, cal, trained)
    assert out.shape == (fm.n_rows,)


def test_calibrated_proba_rejects_missing_proba() -> None:
    from kairon.models.base import Prediction

    fm, y = _overconfident_proba()
    cal = IsotonicCalibrator()
    trained = cal.fit(fm, y)
    pred = Prediction(y_class=np.zeros(3, dtype=np.int64), y_proba=None)
    with pytest.raises(ModelError, match="without y_proba"):
        calibrated_proba(pred, cal, trained)
