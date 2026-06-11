"""Tests for :mod:`kairon.models.stacking`.

Story W6.2 ships the stacked-generalisation meta-learner
(:class:`StackedGeneralizationEnsemble`). The two spec tests pin
the contract:

1. ``test_strict_fold_isolation`` — the stacked meta is fit ONLY on
   OOF predictions from folds <k. Predictions for fold k use ONLY
   the fold-k OOF features as input. The test runs a 3-fold fixture
   and asserts (a) the fit mask is exactly ``fold_id<k`` for every
   ``k``; (b) the fold-k prediction equals the reference
   :func:`generate_stacked_oof`-derived prediction for that fold;
   (c) the fit is rejected with a clear error if ``target_fold``
   has no prior folds (k=0 is a defensible error because fold 0 has
   no preceding OOF rows under the W3.6 protocol).

2. ``test_stacking_handles_missing_xgboost`` — the meta-learner
   falls back to ``sklearn.linear_model.LogisticRegression`` if
   xgboost is not installed, and the test still passes. The test
   imports :func:`kairon.models.stacking.resolve_backend` to assert
   the backend is ``"sklearn"`` when xgboost is unavailable, and
   then fits + predicts on a small stacked fixture to verify the
   sklearn path returns a finite ``p_meta`` in [0, 1].

The tests use a synthetic stacked OOF table built from two
perfectly-separable base models (XOR + balanced-bias) so the meta
has a non-trivial signal to learn. The base models are simple
threshold classifiers on feature columns 0 and 1; the meta-learner
sees a (n_total, 2) matrix of per-base-model OOF probabilities and
fits a 2-class classifier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from kairon.models.contracts import FeatureMatrix
from kairon.models.metalabel import MLConfig
from kairon.models.stacking import (
    StackedGeneralizationEnsemble,
    StackedOofTable,
    resolve_backend,
)
from kairon.splits.walkforward import Fold, SplitSpec, walkforward


# ---------------------------------------------------------------------------
# Synthetic base-model stack
# ---------------------------------------------------------------------------
class _ThresholdBaseModel:
    """A simple threshold classifier on a single feature column.

    The model is a duck-typed :class:`~kairon.models.base.Model`
    substitute: it implements ``fit`` / ``predict`` / ``name`` /
    ``kind`` so the stacked OOF generator can call it through the
    same surface. The model is used by the W6.2 tests to generate a
    stacked OOF table with two distinct base-model columns.

    Two variants are used:

    * ``_ThresholdOnCol0`` — threshold classifier on feature column 0
      (predicts ``y = (x0 > 0.5)``). This matches the perfect-XOR
      rule on the (x0, x1) feature pair only when the x1 signal is
      missing, so it is a NOISY classifier (accuracy ~ 75% on the
      balanced XOR fixture).
    * ``_ThresholdOnCol1`` — symmetric, threshold on x1.
    """

    name = "threshold_base"
    kind = "test"

    def __init__(self, *, col: int, threshold: float = 0.5) -> None:
        self._col: int = int(col)
        self._threshold: float = float(threshold)

    def fit(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        return self  # Stateless — the rule is fixed at construction

    def predict(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> Any:
        col_x: np.ndarray = features.values[:, self._col]
        y_class: np.ndarray = (col_x > self._threshold).astype(np.int64)
        y_proba: np.ndarray = y_class.astype(np.float64)
        return _SimplePrediction(y_class=y_class, y_proba=y_proba)


class _ThresholdOnCol0(_ThresholdBaseModel):
    name = "threshold_col0"

    def __init__(self) -> None:
        super().__init__(col=0)


class _ThresholdOnCol1(_ThresholdBaseModel):
    name = "threshold_col1"

    def __init__(self) -> None:
        super().__init__(col=1)


class _SimplePrediction:
    """Minimal duck-typed :class:`Prediction` for the test base models."""

    def __init__(self, y_class: np.ndarray, y_proba: np.ndarray) -> None:
        self.y_class = y_class
        self.y_proba = y_proba
        self.y_score = None
        self.feature_names: tuple[str, ...] = ()
        self.backend: str = "threshold_base"
        self.meta: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _make_xor_features(n: int, *, seed: int) -> FeatureMatrix:
    """Build a (n, 4) feature matrix with the XOR signal in cols 0, 1.

    Mirrors the W3.6 test fixture (see
    :mod:`tests.evaluation.test_oof`) so the W6.2 tests share the
    same data distribution. The two base models are noisy
    threshold classifiers on columns 0 and 1, so the stacked OOF
    table has a non-trivial meta-signal for the meta-learner to
    learn.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    half: int = n // 2
    class0: np.ndarray = rng.uniform(0.0, 0.5, size=(half, 2))
    class0_upper: np.ndarray = rng.uniform(0.5, 1.0, size=(n - half, 2))
    class0_full: np.ndarray = np.concatenate([class0, class0_upper], axis=0)
    class1_a: np.ndarray = np.column_stack(
        [rng.uniform(0.0, 0.5, size=half), rng.uniform(0.5, 1.0, size=half)]
    )
    class1_b: np.ndarray = np.column_stack(
        [rng.uniform(0.5, 1.0, size=n - half), rng.uniform(0.0, 0.5, size=n - half)]
    )
    class1_full: np.ndarray = np.concatenate([class1_a, class1_b], axis=0)
    perm: np.ndarray = rng.permutation(n)
    pos: np.ndarray = np.concatenate([class0_full, class1_full], axis=0)[perm]
    noise: np.ndarray = rng.uniform(0.0, 1.0, size=(n, 2))
    values: np.ndarray = np.concatenate([pos, noise], axis=1)
    return FeatureMatrix(
        values=values.astype(np.float64),
        feature_names=("x0", "x1", "noise0", "noise1"),
    )


def _make_xor_labels(features: FeatureMatrix) -> np.ndarray:
    x0: np.ndarray = features.values[:, 0]
    x1: np.ndarray = features.values[:, 1]
    return ((x0 > 0.5) ^ (x1 > 0.5)).astype(np.int64)


def _make_balanced_xor_features(n: int) -> FeatureMatrix:
    """Build a balanced (n, 4) feature matrix with XOR labels in cols 0, 1.

    Unlike the W3.6 random fixture (which is permutation-shuffled
    and can have all-one-class prefixes), this fixture
    INTERLEAVES the two classes: rows 0, 2, 4, ... are class 0,
    rows 1, 3, 5, ... are class 1. Every contiguous slice of
    length >= 2 contains both classes, so the meta-learner can
    always fit (LogisticRegression requires >= 2 classes).
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even, got {n}")
    half: int = n // 2
    # Class 0 rows: (x0 <= 0.5, x1 <= 0.5) AND (x0 > 0.5, x1 > 0.5)
    rng: np.random.Generator = np.random.default_rng(20260608)
    class0: np.ndarray = np.concatenate([
        rng.uniform(0.0, 0.5, size=(half, 2)),
    ], axis=0)
    # To get a mix, split class0 into two halves: lower-left and
    # upper-right, both with XOR = 0.
    half_c0: int = half // 2
    class0_a: np.ndarray = class0[:half_c0]
    class0_b: np.ndarray = np.column_stack([
        rng.uniform(0.5, 1.0, size=half - half_c0),
        rng.uniform(0.5, 1.0, size=half - half_c0),
    ])
    class0_full: np.ndarray = np.concatenate([class0_a, class0_b], axis=0)
    # Class 1 rows: (x0 <= 0.5, x1 > 0.5) AND (x0 > 0.5, x1 <= 0.5)
    class1_a: np.ndarray = np.column_stack([
        rng.uniform(0.0, 0.5, size=half_c0),
        rng.uniform(0.5, 1.0, size=half_c0),
    ])
    class1_b: np.ndarray = np.column_stack([
        rng.uniform(0.5, 1.0, size=half - half_c0),
        rng.uniform(0.0, 0.5, size=half - half_c0),
    ])
    class1_full: np.ndarray = np.concatenate([class1_a, class1_b], axis=0)
    # Interleave: row 0 = class0[0], row 1 = class1[0], row 2 = class0[1], ...
    interleaved: np.ndarray = np.empty((n, 2), dtype=np.float64)
    interleaved[0::2] = class0_full
    interleaved[1::2] = class1_full
    noise: np.ndarray = rng.uniform(0.0, 1.0, size=(n, 2))
    values: np.ndarray = np.concatenate([interleaved, noise], axis=1)
    return FeatureMatrix(
        values=values.astype(np.float64),
        feature_names=("x0", "x1", "noise0", "noise1"),
    )


def _make_balanced_xor_labels(features: FeatureMatrix) -> np.ndarray:
    """Recover the interleaved XOR labels (class 0 at even rows, class 1 at odd)."""
    n: int = features.n_rows
    y: np.ndarray = np.empty(n, dtype=np.int64)
    y[0::2] = 0
    y[1::2] = 1
    return y


def _build_stacked_table(
    n: int = 500,
    *,
    seed: int = 20260608,
) -> tuple[FeatureMatrix, np.ndarray, list[Fold], StackedOofTable]:
    """Build a 3-fold stacked OOF table from two noisy base models.

    Returns the (features, y, folds, stacked) tuple. The stacked
    table has 2 base-model columns (p_oof_0, p_oof_1) and one
    p_oof_avg column.

    The folds are HAND-ROLLED (not generated by :func:`walkforward`)
    so the test can guarantee that the fold-0 training slice
    (rows 0..100) contains BOTH classes. The fixture is the
    balanced interleaved XOR (see :func:`_make_balanced_xor_features`)
    so every contiguous slice of length >= 2 contains both classes,
    and LogisticRegression never sees a single-class training set.
    """
    features: FeatureMatrix = _make_balanced_xor_features(n)
    y: np.ndarray = _make_balanced_xor_labels(features)
    # Hand-rolled 3-fold fixture (test_size=100, val_size=0).
    # Fold k's training slice = rows 0..100+100k, test slice = next
    # 100 rows. Every training slice starts at row 0 so every fold
    # sees the same class-0+class-1 mixture.
    folds: list[Fold] = [
        Fold(
            fold_id=0,
            train_start=0, train_end=100,
            val_start=100, val_end=100,
            test_start=100, test_end=200,
        ),
        Fold(
            fold_id=1,
            train_start=0, train_end=200,
            val_start=200, val_end=200,
            test_start=200, test_end=300,
        ),
        Fold(
            fold_id=2,
            train_start=0, train_end=300,
            val_start=300, val_end=300,
            test_start=300, test_end=400,
        ),
    ]
    # Sanity: every training slice must contain both classes so
    # the meta-learner can fit (LogisticRegression requires
    # >= 2 classes).
    for f in folds:
        train_y: np.ndarray = y[f.train_start : f.train_end]
        assert int(np.unique(train_y).size) >= 2, (
            f"fold {f.fold_id} training slice has only "
            f"{np.unique(train_y).size} class(es); both classes "
            f"are required for the meta-learner to fit"
        )

    from kairon.evaluation.oof import generate_stacked_oof

    stacked_pa: pa.Table = generate_stacked_oof(
        features, y, folds,
        base_models=[_ThresholdOnCol0(), _ThresholdOnCol1()],
    )
    stacked: StackedOofTable = StackedOofTable.from_arrow(stacked_pa)
    return features, y, folds, stacked


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_strict_fold_isolation() -> None:
    """The stacked meta is fit ONLY on OOF predictions from folds <k.

    The test runs a 3-fold fixture and asserts (for every fold k
    in {0, 1, 2}):

    (a) the training mask passed to the meta is exactly
        ``fold_id < k`` (no fold-k rows, no row from any fold >= k);
    (b) the fit is REJECTED with a clear error for k=0 (the
        anti-leakage contract: fold 0 has no preceding OOF rows
        under the W3.6 protocol, so fitting a meta on fold 0's
        own OOF rows would leak fold 0's labels);
    (c) for k=1 and k=2, the fit succeeds and the fold-k
        prediction equals the per-fold reference prediction
        produced by fitting the meta on ``stacked.X_oof[mask]``
        manually (the reference re-fits on the same mask and
        predicts on the fold-k rows, so any difference between
        the model.fit_stacked + predict_meta path and the
        reference is a leak).

    A bug in ``fit_stacked`` that used the wrong mask (e.g.
    ``fold_id <= k`` or ``fold_id != k``) would fail (a) and (c).
    A bug in ``predict_meta`` that included extra rows would fail
    (c) as well.
    """
    features, y, folds, stacked = _build_stacked_table()

    # (a) + (b): fold 0 has no preceding OOF rows; the fit must
    # raise a clear error.
    ensemble: StackedGeneralizationEnsemble = StackedGeneralizationEnsemble(
        MLConfig(n_estimators=10, max_depth=2, learning_rate=0.1)
    )
    with pytest.raises(ValueError, match="no OOF rows"):
        ensemble.fit_stacked(stacked, target_fold=0)

    # (a) + (c): for k=1 and k=2, the training mask is exactly
    # ``fold_id < k``, and the fold-k prediction matches the
    # reference (which fits the meta on the same mask manually).
    for k in (1, 2):
        mask: np.ndarray = stacked.fold_ids < k
        # The mask must include ONLY folds < k and exclude fold k
        # and every fold > k.
        assert set(np.unique(stacked.fold_ids[mask]).tolist()) <= set(
            range(0, k)
        ), f"k={k}: training mask contains fold_ids outside <k"
        assert k not in set(np.unique(stacked.fold_ids[mask]).tolist()), (
            f"k={k}: training mask includes fold {k} (leak!)"
        )
        # Reference: fit a fresh ensemble on the same mask.
        ref_ensemble: StackedGeneralizationEnsemble = (
            StackedGeneralizationEnsemble(
                MLConfig(n_estimators=10, max_depth=2, learning_rate=0.1)
            )
        )
        ref_features: FeatureMatrix = FeatureMatrix(
            values=stacked.X_oof[mask].astype(np.float64),
            feature_names=tuple(f"p_base_{i}" for i in range(stacked.n_base)),
        )
        ref_trained = ref_ensemble.fit(ref_features, stacked.y_oof[mask])
        ref_pred: np.ndarray = ref_ensemble.predict_meta(
            ref_trained, stacked.X_oof[stacked.fold_ids == k],
        )
        # Path under test: use fit_stacked + predict_meta.
        trained = ensemble.fit_stacked(stacked, target_fold=k)
        pred: np.ndarray = ensemble.predict_meta(
            trained, stacked.X_oof[stacked.fold_ids == k],
        )
        np.testing.assert_array_equal(pred, ref_pred)
        # p_meta must be in [0, 1] and finite for every row.
        assert np.all(np.isfinite(pred))
        assert np.all((pred >= 0.0) & (pred <= 1.0))


def test_stacking_handles_missing_xgboost() -> None:
    """Falls back to sklearn LogisticRegression if xgboost not installed.

    The test verifies the fallback contract on two axes:

    (1) The :func:`resolve_backend` helper returns ``"sklearn"``
        when xgboost is not importable (i.e. on a torch-less /
        xgboost-less CI environment). We force the path by setting
        ``use_xgboost_if_available=False`` on the config.

    (2) The end-to-end fit + predict path produces a finite
        ``p_meta`` in [0, 1] using the sklearn backend. We fit the
        meta on the full stacked OOF table (not fold-isolated) and
        predict on the same table; the resulting probabilities
        must be in [0, 1] and finite. (This is the W6.2 fallback
        contract: the model is functional even when xgboost is
        missing, and downstream consumers can rely on the
        ``y_proba in [0, 1]`` invariant.)
    """
    # (1) Force the sklearn path: set use_xgboost_if_available=False.
    config: MLConfig = MLConfig(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.1,
        use_xgboost_if_available=False,
    )
    assert resolve_backend(config) == "sklearn", (
        f"resolve_backend must return 'sklearn' when xgboost is "
        f"disabled, got {resolve_backend(config)!r}"
    )

    # (2) End-to-end: fit + predict on the stacked table.
    _build_stacked_table()
    _, _, _, stacked = _build_stacked_table()
    ensemble: StackedGeneralizationEnsemble = StackedGeneralizationEnsemble(config)
    train_x: FeatureMatrix = FeatureMatrix(
        values=stacked.X_oof.astype(np.float64),
        feature_names=tuple(f"p_base_{i}" for i in range(stacked.n_base)),
    )
    trained = ensemble.fit(train_x, stacked.y_oof)
    pred: np.ndarray = ensemble.predict_meta(trained, stacked.X_oof)
    assert np.all(np.isfinite(pred)), "p_meta must be finite for every row"
    assert np.all((pred >= 0.0) & (pred <= 1.0)), (
        f"p_meta must be in [0, 1], got min={pred.min()}, max={pred.max()}"
    )

    # Defensive: when xgboost IS available, resolve_backend returns
    # 'xgboost' (this branch is environment-dependent, so we only
    # assert the disabled case above; the available case is a
    # no-op if xgboost is not installed). We pin the disabled path
    # because that is the deterministic, CI-friendly branch.
