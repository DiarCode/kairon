"""Tests for the W3.3 meta-learner model.

The meta-learner is a *secondary* model in a stacked pipeline: it
consumes the primary model's ``y_proba`` plus a small set of side
features and emits a ``p_meta in [0, 1]`` -- the probability that
the primary's signal is worth taking. The tests in this module
exercise the model contract and the OOF (out-of-fold) fit/predict
path that the W3.6 protocol depends on.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from kairon.models.contracts import FeatureMatrix
from kairon.models.metalabel import MLConfig, MetaLearnerModel


# ---------------------------------------------------------------------------
# Side-feature schema (the OOF meta-feature columns)
# ---------------------------------------------------------------------------
# The schema is fixed and pinned by the test: a downstream change to the
# set or order of these columns is a breaking change for the W3.4 / W3.6
# wiring and must come with an updated acceptance criterion.
OOF_META_FEATURE_COLUMNS: tuple[str, ...] = (
    "p_primary",
    "vol_z",
    "spread_bps",
    "regime",
    "time_of_day_sin",
    "time_of_day_cos",
    "recent_meta_win_rate",
)


def _toy_oof_features(n: int = 200, seed: int = 7) -> FeatureMatrix:
    """Build a synthetic OOF meta-feature matrix with the right schema.

    The features are deliberately noisy except for ``p_primary`` -- the
    perfectly-predictive test (``test_meta_learner_predicts_in_oof_setting``)
    relies on ``p_primary`` carrying the signal.
    """
    rng = np.random.default_rng(seed)
    p_primary = rng.uniform(0.0, 1.0, size=n)
    vol_z = rng.normal(loc=0.0, scale=1.0, size=n)
    spread_bps = rng.uniform(0.5, 5.0, size=n)
    regime = rng.integers(0, 4, size=n).astype(np.float64)
    time_of_day_sin = rng.uniform(-1.0, 1.0, size=n)
    time_of_day_cos = rng.uniform(-1.0, 1.0, size=n)
    recent_meta_win_rate = rng.uniform(0.0, 1.0, size=n)
    values = np.column_stack(
        [
            p_primary,
            vol_z,
            spread_bps,
            regime,
            time_of_day_sin,
            time_of_day_cos,
            recent_meta_win_rate,
        ]
    ).astype(np.float64)
    return FeatureMatrix(values=values, feature_names=OOF_META_FEATURE_COLUMNS)


def _perfectly_predictive_y(p_primary: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """y_meta = 1 iff p_primary > 0.5 (the OOF fixture's ground truth)."""
    return (p_primary > threshold).astype(np.int64)


def _has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_oof_meta_features() -> None:
    """The OOF meta-feature matrix has the expected schema and row count.

    A synthetic 200-row OOF fixture is built with the canonical
    side-feature columns; the resulting ``FeatureMatrix.feature_names``
    tuple must match the documented schema exactly (same order, same
    set, no extra columns, no missing columns).
    """
    fm = _toy_oof_features(n=200, seed=7)
    assert fm.n_rows == 200
    assert fm.feature_names == OOF_META_FEATURE_COLUMNS
    assert fm.n_features == len(OOF_META_FEATURE_COLUMNS)
    # Sanity: every value is finite, no NaNs propagated from the
    # synthetic generator.
    assert np.isfinite(fm.values).all()


def test_meta_learner_predicts_in_oof_setting() -> None:
    """The meta-learner is well-calibrated on a held-out OOF fold.

    A 600-row OOF fixture is generated with
    ``y_meta = (p_primary > 0.5)`` -- the perfectly-predictive
    synthetic setting. The first 400 rows are used for training, the
    remaining 200 for a held-out fold. The trained meta-learner must
    produce a ``p_meta`` whose Brier score on the held-out fold is
    strictly below 0.10 (the calibration bar documented in the PRD
    W3.3 acceptance criterion #2).
    """
    fm = _toy_oof_features(n=600, seed=42)
    y = _perfectly_predictive_y(fm.values[:, 0])

    train_fm = FeatureMatrix(
        values=fm.values[:400],
        feature_names=fm.feature_names,
    )
    test_fm = FeatureMatrix(
        values=fm.values[400:],
        feature_names=fm.feature_names,
    )
    y_train = y[:400]
    y_test = y[400:]

    model = MetaLearnerModel(MLConfig(n_estimators=100, max_depth=3, learning_rate=0.1))
    trained = model.fit(train_fm, y_train)
    pred = model.predict(trained, test_fm)

    assert pred.y_proba is not None
    assert pred.y_proba.shape == (test_fm.n_rows,)
    # y_proba is a probability: in [0, 1].
    assert float(pred.y_proba.min()) >= 0.0
    assert float(pred.y_proba.max()) <= 1.0

    brier = float(((pred.y_proba - y_test.astype(np.float64)) ** 2).mean())
    assert brier < 0.10, f"Brier score {brier:.4f} exceeds the 0.10 calibration bar"


def test_meta_learner_handles_missing_xgboost() -> None:
    """The sklearn fallback path is exercised when xgboost is unavailable.

    With ``use_xgboost_if_available=False``, the implementation MUST
    use :class:`sklearn.ensemble.GradientBoostingClassifier` regardless
    of whether xgboost is installed. The test pins that contract: the
    fit/predict path succeeds, the predicted probability is in [0, 1],
    and the resulting Brier score on the held-out fold is below 0.10
    (matching the perfectly-predictive OOF fixture).
    """
    fm = _toy_oof_features(n=400, seed=11)
    y = _perfectly_predictive_y(fm.values[:, 0])

    train_fm = FeatureMatrix(
        values=fm.values[:300],
        feature_names=fm.feature_names,
    )
    test_fm = FeatureMatrix(
        values=fm.values[300:],
        feature_names=fm.feature_names,
    )
    y_train = y[:300]
    y_test = y[300:]

    # The fallback must work even when xgboost IS installed (the
    # ``use_xgboost_if_available=False`` flag forces sklearn).
    model = MetaLearnerModel(MLConfig(use_xgboost_if_available=False))
    trained = model.fit(train_fm, y_train)

    # The fitted artifact must be a sklearn GradientBoostingClassifier,
    # not an xgboost XGBClassifier. This is the load-bearing assertion:
    # a regression that swaps in xgboost unconditionally would break
    # environments without the xgboost wheel.
    from sklearn.ensemble import GradientBoostingClassifier

    assert isinstance(trained.state, GradientBoostingClassifier)

    pred = model.predict(trained, test_fm)
    assert pred.y_proba is not None
    assert pred.y_proba.shape == (test_fm.n_rows,)
    assert float(pred.y_proba.min()) >= 0.0
    assert float(pred.y_proba.max()) <= 1.0

    brier = float(((pred.y_proba - y_test.astype(np.float64)) ** 2).mean())
    assert brier < 0.10, f"sklearn-fallback Brier score {brier:.4f} exceeds 0.10"

    # Cross-check: in this environment, xgboost is NOT installed, so
    # the xgboost preferred-backend path is never taken; the test is
    # runnable in both environments. When xgboost is installed
    # later, the fallback path still works (the type assertion above
    # proves it).
    if not _has_xgboost():
        # The default-config (use_xgboost_if_available=True) build also
        # works in the no-xgboost environment because the runtime
        # probe selects sklearn.
        default_model = MetaLearnerModel()
        default_trained = default_model.fit(train_fm, y_train)
        assert isinstance(default_trained.state, GradientBoostingClassifier)


def test_meta_learner_predict_returns_probability() -> None:
    """Predict returns a valid ``Prediction`` with ``y_proba in [0, 1]``.

    Uses a larger fixture (1500 rows) to verify the probability vector
    is densely populated (no collapse to a constant). The empirical
    Brier score is asserted within the 0.10 calibration bar on the
    same perfectly-predictive ground truth.
    """
    fm = _toy_oof_features(n=1500, seed=99)
    y = _perfectly_predictive_y(fm.values[:, 0])

    train_fm = FeatureMatrix(
        values=fm.values[:1000],
        feature_names=fm.feature_names,
    )
    test_fm = FeatureMatrix(
        values=fm.values[1000:],
        feature_names=fm.feature_names,
    )
    y_train = y[:1000]
    y_test = y[1000:]

    model = MetaLearnerModel(
        MLConfig(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=123)
    )
    trained = model.fit(train_fm, y_train)
    pred = model.predict(trained, test_fm)

    # Output shape and contract checks.
    assert pred.y_proba is not None
    assert pred.y_proba.shape == (test_fm.n_rows,)
    assert pred.y_class.shape == (test_fm.n_rows,)
    assert set(pred.y_class.tolist()).issubset({0, 1})

    # Probability range.
    assert float(pred.y_proba.min()) >= 0.0
    assert float(pred.y_proba.max()) <= 1.0

    # Brier score within the calibration tolerance.
    brier = float(((pred.y_proba - y_test.astype(np.float64)) ** 2).mean())
    assert brier < 0.10, f"Brier score {brier:.4f} exceeds the 0.10 calibration bar"


__all__ = [
    "OOF_META_FEATURE_COLUMNS",
    "test_oof_meta_features",
    "test_meta_learner_predicts_in_oof_setting",
    "test_meta_learner_handles_missing_xgboost",
    "test_meta_learner_predict_returns_probability",
]
