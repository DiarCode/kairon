"""Tests for Model ABC + linear/tree backends (LSTM/XGB/LGBM skipped if no dep)."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.models.base import ModelError, Prediction
from kairon.models.contracts import FeatureMatrix
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.tree import (
    LightGBMConfig,
    LightGBMModel,
    RandomForestConfig,
    RandomForestModel,
    XGBoostConfig,
    XGBoostModel,
)


def _has(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def _toy_data(n: int = 200, seed: int = 7) -> tuple[FeatureMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.5 * x2 + 0.1 * rng.normal(size=n) > 0).astype(np.int64)
    fm = FeatureMatrix(
        values=np.column_stack([x1, x2]).astype(np.float64),
        feature_names=("x1", "x2"),
    )
    return fm, y


def test_logreg_fit_predict() -> None:
    fm, y = _toy_data()
    m = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    trained = m.fit(fm, y)
    assert trained.target_kind == "classification"
    assert trained.classes is not None
    pred = m.predict(trained, fm)
    assert isinstance(pred, Prediction)
    assert pred.y_class.shape == (fm.n_rows,)
    assert pred.y_proba is not None
    assert pred.y_proba.shape == (fm.n_rows,)
    assert set(pred.y_class.tolist()).issubset({0, 1})


def test_logreg_validates_config() -> None:
    with pytest.raises(ValueError):
        LinearConfig(C=0)
    with pytest.raises(ValueError):
        LinearConfig(class_weight="bogus")  # type: ignore[arg-type]


def test_logreg_rejects_zero_rows() -> None:
    fm = FeatureMatrix(values=np.zeros((0, 2)), feature_names=("a", "b"))
    y = np.zeros(0, dtype=np.int64)
    m = LogisticRegressionModel()
    with pytest.raises(ModelError, match="zero rows"):
        m.fit(fm, y)


def test_logreg_rejects_length_mismatch() -> None:
    fm = FeatureMatrix(
        values=np.zeros((10, 2)), feature_names=("a", "b")
    )
    m = LogisticRegressionModel()
    with pytest.raises(ModelError, match="rows"):
        m.fit(fm, np.zeros(5, dtype=np.int64))


def test_logreg_predict_rejects_wrong_backend() -> None:
    fm, y = _toy_data()
    m = LogisticRegressionModel()
    trained = m.fit(fm, y)
    # Predict with a different model
    other = RandomForestModel(RandomForestConfig(n_estimators=10, random_state=0, n_jobs=1))
    with pytest.raises(ModelError, match="trained model is"):
        other.predict(trained, fm)


def test_logreg_predict_rejects_feature_mismatch() -> None:
    fm, y = _toy_data()
    m = LogisticRegressionModel()
    trained = m.fit(fm, y)
    wrong = FeatureMatrix(values=np.zeros((3, 2)), feature_names=("a", "b"))
    with pytest.raises(ModelError, match="feature mismatch"):
        m.predict(trained, wrong)


def test_random_forest_fit_predict() -> None:
    fm, y = _toy_data()
    m = RandomForestModel(
        RandomForestConfig(n_estimators=20, random_state=0, n_jobs=1)
    )
    trained = m.fit(fm, y)
    pred = m.predict(trained, fm)
    assert pred.y_class.shape == (fm.n_rows,)
    assert pred.backend == "random_forest"


def test_random_forest_validates_config() -> None:
    with pytest.raises(ValueError):
        RandomForestConfig(n_estimators=0)
    with pytest.raises(ValueError):
        RandomForestConfig(max_depth=0)


@pytest.mark.skipif(
    not _has("xgboost"), reason="xgboost not installed (uv sync --extra ml)"
)
def test_xgboost_fit_predict() -> None:
    fm, y = _toy_data()
    m = XGBoostModel(
        XGBoostConfig(n_estimators=20, max_depth=3, n_jobs=1, random_state=0)
    )
    trained = m.fit(fm, y)
    pred = m.predict(trained, fm)
    assert pred.backend == "xgboost"


def test_xgboost_validates_config() -> None:
    with pytest.raises(ValueError):
        XGBoostConfig(n_estimators=0)
    with pytest.raises(ValueError):
        XGBoostConfig(learning_rate=0)
    with pytest.raises(ValueError):
        XGBoostConfig(subsample=1.5)


def test_xgboost_raises_when_missing() -> None:
    if _has("xgboost"):
        pytest.skip("xgboost installed; can't test missing-dep path")
    with pytest.raises(ModelError, match="xgboost is not installed"):
        XGBoostModel()


@pytest.mark.skipif(
    not _has("lightgbm"), reason="lightgbm not installed (uv sync --extra ml)"
)
def test_lightgbm_fit_predict() -> None:
    fm, y = _toy_data()
    m = LightGBMModel(
        LightGBMConfig(n_estimators=20, n_jobs=1, random_state=0)
    )
    trained = m.fit(fm, y)
    pred = m.predict(trained, fm)
    assert pred.backend == "lightgbm"


def test_lightgbm_validates_config() -> None:
    with pytest.raises(ValueError):
        LightGBMConfig(n_estimators=0)
    with pytest.raises(ValueError):
        LightGBMConfig(num_leaves=1)


def test_lightgbm_raises_when_missing() -> None:
    if _has("lightgbm"):
        pytest.skip("lightgbm installed; can't test missing-dep path")
    with pytest.raises(ModelError, match="lightgbm is not installed"):
        LightGBMModel()


def test_prediction_post_init() -> None:
    with pytest.raises(ValueError, match="y_class must be 1-D"):
        Prediction(y_class=np.zeros((3, 2)), y_proba=None)


def test_prediction_rejects_row_mismatch() -> None:
    with pytest.raises(ValueError, match="y_proba has"):
        Prediction(
            y_class=np.zeros(3, dtype=np.int64),
            y_proba=np.zeros(4),
        )
