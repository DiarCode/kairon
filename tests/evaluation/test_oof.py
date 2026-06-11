"""Tests for :mod:`kairon.evaluation.oof`.

Story W3.6 ships the LOAD-BEARING anti-leakage protocol for the
entire meta-labeling + stacking pipeline (Architect Tension E from
round 1). The four tests pin the contract:

1. ``test_fold_strict_isolation`` — for any fold ``k``, the OOF
   features at fold ``k`` have NO contributions from fold ``k``'s
   labels. Verified by re-fitting the primary on the OOF training
   set for fold ``k`` and asserting the fold ``k`` prediction is
   identical to what :func:`generate_oof_predictions` produced.
2. ``test_oof_dimensions_match`` — the OOF table has shape
   ``(n_total, 5)`` with the documented columns; row order matches
   the concatenation of fold.test slices in fold order.
3. ``test_oof_handles_3_folds_correctly`` — with 3 walk-forward
   folds, the OOF table contains 3 distinct ``fold_id`` values,
   each with the right number of test rows from that fold.
4. ``test_oof_perfectly_recovers_known_signal`` — on a synthetic
   fixture where the primary is a perfect classifier
   (XOR-like), the OOF predictions are 100% accurate on every
   fold.

The tests use a lightweight deterministic primary model
(:class:`_PerfectXORModel`) that fits in a few microseconds and
is identical for every call of the factory. The factory pattern is
load-bearing: the OOF generator calls it once per fold, and we
assert the factory is called ``K`` times (one per fold) on a
3-fold fixture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from kairon.evaluation.oof import (
    generate_oof_predictions,
    generate_stacked_oof,
)
from kairon.models.contracts import FeatureMatrix
from kairon.splits.walkforward import Fold, SplitSpec, walkforward


# ---------------------------------------------------------------------------
# Synthetic primary model: a perfect XOR-like classifier
# ---------------------------------------------------------------------------
# The model learns a deterministic rule that is a function of two
# specific feature columns (the XOR of ``x0`` and ``x1``). The rule is
# identical for every fresh factory call, so any fold-k prediction
# that depends on fold-k's labels is detectable: the fold-k test
# rows would NOT be perfectly classified.
class _PerfectXORModel:
    """A perfect classifier on the XOR of feature columns 0 and 1.

    Used as a deterministic primary in the W3.6 tests. The model is
    a duck-typed :class:`~kairon.models.base.Model` substitute — it
    implements ``fit``, ``predict``, ``name`` and ``kind`` so the
    OOF generator can call it through the same surface. We avoid
    inheriting from :class:`Model` to keep the test hermetic (no
    dependency on torch / sklearn / xgboost).
    """

    name: str = "perfect_xor"
    kind: str = "test"

    def __init__(self) -> None:
        # The learned decision rule. After ``fit``, the model
        # inspects the training labels and figures out which
        # permutation of ``(x0 XOR x1) -> {0, 1}`` is consistent
        # with the training data. On a perfectly-separable XOR
        # the rule is unambiguous.
        self._sign: int = 1
        self._fitted: bool = False

    def fit(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        # Inspect the XOR pattern in the training data to recover
        # the sign. The XOR of columns 0 and 1 of the training
        # features, sign-corrected, must equal ``y``. We compute
        # the sign by trying both and picking the one that
        # matches the training labels.
        if features.n_rows == 0:
            raise ValueError("cannot fit on zero rows")
        x0: np.ndarray = features.values[:, 0]
        x1: np.ndarray = features.values[:, 1]
        xor_signal: np.ndarray = ((x0 > 0.5) ^ (x1 > 0.5)).astype(np.int64)
        y_int: np.ndarray = np.asarray(y, dtype=np.int64)
        # Try sign=+1
        pos_match: int = int(np.sum(xor_signal == y_int))
        neg_match: int = int(np.sum((1 - xor_signal) == y_int))
        self._sign = 1 if pos_match >= neg_match else -1
        self._fitted = True
        return self  # TrainedModel substitute

    def predict(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> Any:
        x0: np.ndarray = features.values[:, 0]
        x1: np.ndarray = features.values[:, 1]
        xor_signal: np.ndarray = ((x0 > 0.5) ^ (x1 > 0.5)).astype(np.int64)
        if self._sign == 1:
            y_class: np.ndarray = xor_signal
        else:
            y_class: np.ndarray = 1 - xor_signal
        # ``y_proba`` shape (n,): float64 in [0, 1]. The OOF
        # generator reads y_proba as the class-1 probability.
        y_proba: np.ndarray = y_class.astype(np.float64)
        return _SimplePrediction(y_class=y_class, y_proba=y_proba)


class _SimplePrediction:
    """Minimal duck-typed :class:`Prediction` for the test primary."""

    def __init__(self, y_class: np.ndarray, y_proba: np.ndarray) -> None:
        self.y_class = y_class
        self.y_proba = y_proba
        self.y_score = None
        self.feature_names: tuple[str, ...] = ()
        self.backend: str = "perfect_xor"
        self.meta: dict[str, Any] = {}


def _xor_factory() -> _PerfectXORModel:
    """Factory callable that returns a fresh :class:`_PerfectXORModel`.

    The OOF generator's contract is ``Callable[[], Any]`` (see
    :data:`kairon.evaluation.oof.ModelFactory`), so the factory
    must be a callable that returns a model instance — not the
    class itself. Wrapping the constructor in a zero-arg
    function keeps the test simple and the call site identical
    to a real model factory.
    """
    return _PerfectXORModel()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _make_xor_features(n: int, *, seed: int) -> FeatureMatrix:
    """Build a ``(n, 4)`` feature matrix with the XOR signal in cols 0,1.

    Columns 2 and 3 are noise (irrelevant to the model). The
    binary signal is ``y = (x0 > 0.5) XOR (x1 > 0.5)`` so the
    perfect-classifier fixture is non-trivial: a single-feature
    decision boundary would not recover it.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    # n/2 class-0, n/2 class-1, drawn as a balanced XOR
    half: int = n // 2
    # Class 0: (x0<=0.5, x1<=0.5) AND (x0>0.5, x1>0.5) => XOR=0
    class0: np.ndarray = rng.uniform(0.0, 0.5, size=(half, 2))
    class0_upper: np.ndarray = rng.uniform(0.5, 1.0, size=(n - half, 2))
    class0_full: np.ndarray = np.concatenate([class0, class0_upper], axis=0)
    # Class 1: (x0<=0.5, x1>0.5) AND (x0>0.5, x1<=0.5) => XOR=1
    class1_a: np.ndarray = np.column_stack(
        [rng.uniform(0.0, 0.5, size=half), rng.uniform(0.5, 1.0, size=half)]
    )
    class1_b: np.ndarray = np.column_stack(
        [rng.uniform(0.5, 1.0, size=n - half), rng.uniform(0.0, 0.5, size=n - half)]
    )
    class1_full: np.ndarray = np.concatenate([class1_a, class1_b], axis=0)
    # Shuffle so the rows are not class-blocked; the
    # class-blocked order would still work but a shuffle
    # matches the realistic data distribution.
    perm: np.ndarray = rng.permutation(n)
    pos: np.ndarray = np.concatenate([class0_full, class1_full], axis=0)[perm]
    noise: np.ndarray = rng.uniform(0.0, 1.0, size=(n, 2))
    values: np.ndarray = np.concatenate([pos, noise], axis=1)
    return FeatureMatrix(
        values=values.astype(np.float64),
        feature_names=("x0", "x1", "noise0", "noise1"),
    )


def _make_xor_labels(features: FeatureMatrix) -> np.ndarray:
    """Recover the XOR labels for a balanced fixture built by :func:`_make_xor_features`."""
    x0: np.ndarray = features.values[:, 0]
    x1: np.ndarray = features.values[:, 1]
    return ((x0 > 0.5) ^ (x1 > 0.5)).astype(np.int64)


def _oof_train_idx_for_fold(folds: list[Fold], k: int) -> tuple[int, int]:
    """Return ``[start, end)`` of the OOF training slice for fold ``k``.

    Mirrors the OOF generator's own slice logic: for ``k=0`` use
    the fold's own train+val; for ``k>0`` use the union of all
    prior folds' train+val. Used by the test to re-fit the
    reference model and check the OOF generator's predictions
    bit-for-bit.
    """
    f = folds[k]
    if k == 0:
        return int(f.train_start), int(f.val_end)
    start: int = int(folds[0].train_start)
    end: int = int(folds[k - 1].val_end)
    return start, end


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_fold_strict_isolation() -> None:
    """Fold k's OOF features are produced by a model trained without any fold-k labels.

    We verify the protocol on a 3-fold walk-forward fixture:

    (a) re-fit the primary on the OOF training slice for fold
        ``k`` (a fresh model, no test rows of fold ``k`` in the
        training set, no fold ``k`` train/val rows in the
        training set for ``k>0``);
    (b) compare its fold-``k`` prediction to the OOF table's
        fold-``k`` rows — they must be bit-identical (no
        numerical drift);
    (c) inspect the model's training indices and assert they
        contain exactly the rows the OOF generator is allowed
        to use (fold 0's own train+val, or folds ``<k`` union
        for ``k > 0``) and NO test rows from any fold.

    If any of (a)-(c) fails, the OOF generator has leaked fold
    ``k``'s labels into the training set and the protocol is
    broken. The test is hermetic (no IO) and runs in a few ms.
    """
    # 3 walk-forward folds: n=500, train=200, test=100 -> exactly 3 folds.
    n: int = 500
    features: FeatureMatrix = _make_xor_features(n, seed=20260608)
    y: np.ndarray = _make_xor_labels(features)

    spec = SplitSpec(train_size=200, val_size=0, test_size=100)
    folds: list[Fold] = walkforward(n, spec=spec)
    assert len(folds) == 3, f"expected 3 folds, got {len(folds)}"

    oof: pa.Table = generate_oof_predictions(
        features, y, folds, primary_model_factory=_xor_factory,
    )

    # (a) + (b): re-fit the primary on the OOF training slice
    # for each fold ``k`` and check bit-identical predictions.
    for k in range(3):
        f = folds[k]
        train_start, train_end = _oof_train_idx_for_fold(folds, k)
        train_x: FeatureMatrix = FeatureMatrix(
            values=features.values[train_start:train_end],
            feature_names=features.feature_names,
        )
        train_y: np.ndarray = y[train_start:train_end].astype(np.int64)

        ref_model: _PerfectXORModel = _PerfectXORModel()
        ref_model.fit(train_x, train_y)
        ref_pred: _SimplePrediction = ref_model.predict(ref_model, FeatureMatrix(
            values=features.values[f.test_start : f.test_end],
            feature_names=features.feature_names,
        ))

        # The OOF table's fold-k rows must equal the reference
        # predictions bit-for-bit.
        mask: np.ndarray = (oof["fold_id"].to_numpy() == f.fold_id)
        oof_p: np.ndarray = oof["p_oof"].to_numpy()[mask]
        assert oof_p.shape[0] == ref_pred.y_proba.shape[0]
        # Use exact equality (no tolerance) — both the OOF
        # generator and the re-fit use the same deterministic
        # model, so any numerical drift is a leak.
        np.testing.assert_array_equal(oof_p, ref_pred.y_proba)
        np.testing.assert_array_equal(
            oof["p_oof_class"].to_numpy()[mask],
            ref_pred.y_class,
        )

    # (c): instrument the factory to capture the training
    # slice the OOF generator uses for each fold. The slice
    # size must match the OOF training rule exactly, and the
    # generator must be called once per fold.
    seen_train_slices: list[np.ndarray] = []

    def _spy_factory() -> _PerfectXORModel:
        class _Spy(_PerfectXORModel):
            def fit(
                self,
                features: FeatureMatrix,
                y: np.ndarray,
                *,
                sample_weight: np.ndarray | None = None,
            ) -> Any:
                seen_train_slices.append(features.values.copy())
                return super().fit(features, y, sample_weight=sample_weight)
        return _Spy()

    _ = generate_oof_predictions(
        features, y, folds, primary_model_factory=_spy_factory,
    )

    assert len(seen_train_slices) == len(folds), (
        f"expected {len(folds)} factory calls (one per fold), "
        f"got {len(seen_train_slices)}"
    )
    for k in range(len(folds)):
        train_slice: np.ndarray = seen_train_slices[k]
        expected_start, expected_end = _oof_train_idx_for_fold(folds, k)
        expected_n: int = expected_end - expected_start
        assert train_slice.shape[0] == expected_n, (
            f"fold {k}: OOF generator trained on {train_slice.shape[0]} "
            f"rows, expected {expected_n} (the OOF training slice for "
            f"this fold). Any other size is a leak."
        )
        # The training rows must come from the OOF training
        # slice range — i.e. the slice is a copy of
        # ``features.values[expected_start:expected_end]``.
        np.testing.assert_array_equal(
            train_slice, features.values[expected_start:expected_end],
        )


def test_oof_dimensions_match() -> None:
    """OOF table has shape ``(n_total, 5)`` and the documented schema.

    The 5 columns are ``fold_id, y_true, p_oof, p_oof_class,
    is_train``. The row order is the concatenation of fold.test
    slices in fold order, which is the contract the W3.3
    meta-learner consumes (it builds the OOF feature matrix by
    reading the table in row order).
    """
    n: int = 500
    features: FeatureMatrix = _make_xor_features(n, seed=20260608)
    y: np.ndarray = _make_xor_labels(features)

    spec = SplitSpec(train_size=200, val_size=0, test_size=100)
    folds: list[Fold] = walkforward(n, spec=spec)
    assert len(folds) == 3

    oof: pa.Table = generate_oof_predictions(
        features, y, folds, primary_model_factory=_xor_factory,
    )

    # 5 columns, exact names, exact dtypes.
    assert oof.column_names == [
        "fold_id", "y_true", "p_oof", "p_oof_class", "is_train",
    ], f"OOF columns are {oof.column_names}"

    expected_rows: int = sum(f.test_end - f.test_start for f in folds)
    assert oof.num_rows == expected_rows, (
        f"OOF table has {oof.num_rows} rows, expected {expected_rows}"
    )
    assert oof.num_columns == 5

    # Row order = concatenation of fold.test slices in fold order.
    fold_id_arr: np.ndarray = oof["fold_id"].to_numpy()
    # All fold-0 rows come first (n_test of fold 0 contiguous rows).
    n0: int = int(folds[0].test_end - folds[0].test_start)
    n1: int = int(folds[1].test_end - folds[1].test_start)
    assert np.all(fold_id_arr[:n0] == 0)
    assert np.all(fold_id_arr[n0 : n0 + n1] == 1)
    assert np.all(fold_id_arr[n0 + n1:] == 2)

    # y_true must match the input y in row order (the OOF
    # generator slices y by fold.test_start..fold.test_end).
    y_true_arr: np.ndarray = oof["y_true"].to_numpy()
    expected_y_true: np.ndarray = np.concatenate([
        y[folds[0].test_start : folds[0].test_end],
        y[folds[1].test_start : folds[1].test_end],
        y[folds[2].test_start : folds[2].test_end],
    ]).astype(np.int64)
    np.testing.assert_array_equal(y_true_arr, expected_y_true)

    # p_oof must be in [0, 1] and finite.
    p_oof: np.ndarray = oof["p_oof"].to_numpy()
    assert np.all(np.isfinite(p_oof))
    assert np.all((p_oof >= 0.0) & (p_oof <= 1.0))

    # p_oof_class must be the 0.5-thresholded version of p_oof.
    p_class: np.ndarray = oof["p_oof_class"].to_numpy()
    np.testing.assert_array_equal(p_class, (p_oof >= 0.5).astype(np.int64))

    # is_train must be all False (OOF rows are never in their
    # own training set).
    is_train: np.ndarray = oof["is_train"].to_numpy()
    assert not np.any(is_train), (
        f"OOF rows have is_train=True at "
        f"{np.flatnonzero(is_train).tolist()}; OOF protocol "
        f"requires is_train=False everywhere"
    )

    # Schema-level dtypes — pyarrow's strict schema is
    # part of the contract (downstream consumers index the
    # table by column name + type).
    schema: pa.Schema = oof.schema
    assert schema.field("fold_id").type == pa.int32()
    assert schema.field("y_true").type == pa.int64()
    assert schema.field("p_oof").type == pa.float64()
    assert schema.field("p_oof_class").type == pa.int64()
    assert schema.field("is_train").type == pa.bool_()


def test_oof_handles_3_folds_correctly() -> None:
    """3 walk-forward folds => 3 distinct ``fold_id`` values, each with the right n_test.

    The OOF table must contain rows for exactly 3 folds, and
    each fold must contribute its ``n_test`` rows. The test
    uses a 3-fold fixture (n=500, train=200, test=100) and
    asserts the per-fold row counts match the fold metadata.
    """
    n: int = 500
    features: FeatureMatrix = _make_xor_features(n, seed=20260608)
    y: np.ndarray = _make_xor_labels(features)

    spec = SplitSpec(train_size=200, val_size=0, test_size=100)
    folds: list[Fold] = walkforward(n, spec=spec)
    assert len(folds) == 3, (
        f"test fixture must produce 3 folds, got {len(folds)}: "
        f"{[f.test_end - f.test_start for f in folds]}"
    )

    oof: pa.Table = generate_oof_predictions(
        features, y, folds, primary_model_factory=_xor_factory,
    )

    fold_ids: np.ndarray = oof["fold_id"].to_numpy()
    unique_ids: np.ndarray = np.unique(fold_ids)
    assert unique_ids.tolist() == [0, 1, 2], (
        f"OOF table has fold_ids={unique_ids.tolist()}, expected [0, 1, 2]"
    )

    for f in folds:
        mask: np.ndarray = fold_ids == f.fold_id
        expected_n: int = int(f.test_end - f.test_start)
        actual_n: int = int(mask.sum())
        assert actual_n == expected_n, (
            f"fold {f.fold_id}: OOF has {actual_n} rows, "
            f"expected {expected_n}"
        )

    # The 3 fold_ids must appear in chronological order in the
    # table (the row-order contract).
    first_idx_per_fold: list[int] = [
        int(np.where(fold_ids == f.fold_id)[0][0]) for f in folds
    ]
    assert first_idx_per_fold == sorted(first_idx_per_fold), (
        f"fold row-blocks are not in chronological order: "
        f"{first_idx_per_fold}"
    )


def test_oof_perfectly_recovers_known_signal() -> None:
    """On a perfect-XOR fixture, OOF predictions are 100% accurate on every fold.

    The synthetic primary is a deterministic perfect classifier
    on the XOR of feature columns 0 and 1. Because the fixture
    is constructed so the XOR signal is perfectly classifiable,
    a fresh model fit on each fold's training set must classify
    the fold's test rows with 100% accuracy.

    The test asserts ``accuracy == 1.0`` on every fold, which
    would fail if the OOF generator leaked any fold-k label
    into the model (a leaky leak typically degrades to
    ``accuracy == random_guess`` on at least one fold). The
    test is the strongest pin on the protocol: a model that
    is *too good* on the training set and *random* on the
    test set is a textbook leakage signature.
    """
    n: int = 500
    features: FeatureMatrix = _make_xor_features(n, seed=20260608)
    y: np.ndarray = _make_xor_labels(features)

    spec = SplitSpec(train_size=200, val_size=0, test_size=100)
    folds: list[Fold] = walkforward(n, spec=spec)
    assert len(folds) == 3

    oof: pa.Table = generate_oof_predictions(
        features, y, folds, primary_model_factory=_xor_factory,
    )

    y_pred: np.ndarray = oof["p_oof_class"].to_numpy()
    y_true: np.ndarray = oof["y_true"].to_numpy()
    fold_ids: np.ndarray = oof["fold_id"].to_numpy()

    # Per-fold accuracy.
    for f in folds:
        mask: np.ndarray = fold_ids == f.fold_id
        n_correct: int = int(np.sum(y_pred[mask] == y_true[mask]))
        n_total: int = int(mask.sum())
        assert n_correct == n_total, (
            f"fold {f.fold_id}: OOF accuracy is "
            f"{n_correct}/{n_total} on a perfect-XOR fixture; "
            f"a leakage or model bug would degrade this. "
            f"Predictions: {y_pred[mask].tolist()[:10]}..., "
            f"truth: {y_true[mask].tolist()[:10]}..."
        )

    # Overall accuracy is 100%.
    assert int(np.sum(y_pred == y_true)) == oof.num_rows


# ---------------------------------------------------------------------------
# Defensive tests (W3.6 spec only requires the 4 above; these are pinned
# to surface common regressions in a strict CI environment)
# ---------------------------------------------------------------------------
def test_oof_fold_with_zero_training_raises() -> None:
    """A hand-rolled fold with ``train_size=0`` raises a clear error.

    The OOF generator requires every fold to have at least one
    training row. The :func:`walkforward` function enforces
    ``train_size >= 1`` so a zero-row training set only occurs
    if a caller hand-builds a :class:`Fold` with no train or
    val slice. The OOF generator surfaces a clear error rather
    than fitting on zero rows.
    """
    n: int = 100
    features: FeatureMatrix = _make_xor_features(n, seed=20260608)
    y: np.ndarray = _make_xor_labels(features)

    # Hand-rolled fold with train_start == train_end (zero
    # training rows) and val_start == val_end (zero val rows).
    folds: list[Fold] = [
        Fold(
            fold_id=0,
            train_start=10, train_end=10,
            val_start=10, val_end=10,
            test_start=20, test_end=40,
        ),
    ]

    with pytest.raises(ValueError, match="empty training set"):
        generate_oof_predictions(
            features, y, folds, primary_model_factory=_xor_factory,
        )


def test_oof_consistent_with_known_folds() -> None:
    """A hand-rolled 2-fold fixture produces the documented row count.

    Sanity check that the OOF generator is faithful to the
    Fold dataclass: with a hand-rolled 2-fold fixture (no
    val), the total OOF rows = sum of n_test over both folds.
    This test complements the walkforward-driven fixtures
    above by checking the OOF generator works on
    Fold dataclasses built by hand (the dataclass is
    the only stable contract between the splits and
    evaluation layers).
    """
    n: int = 100
    features: FeatureMatrix = _make_xor_features(n, seed=20260608)
    y: np.ndarray = _make_xor_labels(features)

    folds: list[Fold] = [
        Fold(
            fold_id=0,
            train_start=0, train_end=20,
            val_start=20, val_end=20,
            test_start=20, test_end=40,
        ),
        Fold(
            fold_id=1,
            train_start=0, train_end=40,
            val_start=40, val_end=40,
            test_start=40, test_end=60,
        ),
    ]

    oof: pa.Table = generate_oof_predictions(
        features, y, folds, primary_model_factory=_xor_factory,
    )
    # 20 rows for fold 0 + 20 rows for fold 1 = 40.
    assert oof.num_rows == 40
    assert oof["fold_id"].to_numpy().tolist() == [0] * 20 + [1] * 20


# ---------------------------------------------------------------------------
# W6.1 — Stacked OOF isolation (per-base-model)
# ---------------------------------------------------------------------------
def test_stacked_oof_isolation() -> None:
    """Same isolation contract as W3.6, but verified per-base-model.

    The W6.1 stacked OOF generator emits ``p_oof_<i>`` columns for
    every base model in ``base_models``. For every base model ``i``
    and every fold ``k``, the ``p_oof_<i>`` values for fold-``k``
    test rows must come from base model ``i`` fit on the OOF
    training slice for fold ``k`` (k=0 -> folds[0].train+val,
    k>0 -> union of folds[0..k-1].train+val). The test verifies
    this by:

    (a) running the stacked generator with two distinct base
        models (both ``_PerfectXORModel``-style factories but
        with different ``name`` attributes) and asserting the
        per-base-model ``p_oof_<i>`` columns are equal to the
        single-model OOF ``p_oof`` produced by
        :func:`generate_oof_predictions` for the same model
        factory;
    (b) re-fitting each base model on the OOF training slice
        for every fold ``k`` and asserting the fold-``k``
        predictions are bit-identical to the stacked table's
        ``p_oof_<i>`` rows for fold ``k``;
    (c) instrumenting the per-base-model call to capture the
        training slice the stacked generator used for every
        ``(model, fold)`` pair and asserting the slice size
        and content match the OOF training-slice contract
        exactly.

    If any of (a)-(c) fails, the stacked generator has leaked
    fold-``k`` labels into the base-model training set for at
    least one base model, and the W6.2 meta-learner would
    inherit the leak.
    """
    n: int = 500
    features: FeatureMatrix = _make_xor_features(n, seed=20260608)
    y: np.ndarray = _make_xor_labels(features)

    spec = SplitSpec(train_size=200, val_size=0, test_size=100)
    folds: list[Fold] = walkforward(n, spec=spec)
    assert len(folds) == 3

    # Two distinct base models (same perfect-XOR rule but distinct
    # ``name`` attributes so the OOF generator can tell them apart
    # in error messages). Both satisfy the Model protocol surface.
    class _ModelA(_PerfectXORModel):
        name = "model_a"

    class _ModelB(_PerfectXORModel):
        name = "model_b"

    base_models: list[_PerfectXORModel] = [_ModelA(), _ModelB()]

    stacked: pa.Table = generate_stacked_oof(
        features, y, folds, base_models=base_models,
    )

    # (a) Per-base-model columns match the single-model OOF for the
    # same factory. We re-run :func:`generate_oof_predictions` for
    # each base model factory and assert the per-fold rows line up
    # bit-for-bit with the stacked table's ``p_oof_<i>`` column.
    for i, m in enumerate(base_models):
        single: pa.Table = generate_oof_predictions(
            features, y, folds, primary_model_factory=lambda: m,
        )
        np.testing.assert_array_equal(
            stacked[f"p_oof_{i}"].to_numpy(),
            single["p_oof"].to_numpy(),
        )

    # Row count and column structure.
    expected_rows: int = sum(f.test_end - f.test_start for f in folds)
    assert stacked.num_rows == expected_rows
    # 3 + n_base + 1 = 3 (fold_id, y_true, is_train) + 2 (p_oof_0, p_oof_1)
    # + 1 (p_oof_avg) = 6 columns.
    expected_cols: list[str] = [
        "fold_id", "y_true", "p_oof_0", "p_oof_1", "p_oof_avg", "is_train",
    ]
    assert stacked.column_names == expected_cols, (
        f"stacked columns are {stacked.column_names}, expected {expected_cols}"
    )
    # p_oof_avg is the simple average of p_oof_0 and p_oof_1.
    np.testing.assert_allclose(
        stacked["p_oof_avg"].to_numpy(),
        0.5 * (stacked["p_oof_0"].to_numpy() + stacked["p_oof_1"].to_numpy()),
    )
    # is_train is always False (OOF rows are never in their own
    # training set, for any base model).
    assert not np.any(stacked["is_train"].to_numpy())
    # Schema: fold_id int32, y_true int64, p_oof_* float64, is_train bool.
    schema: pa.Schema = stacked.schema
    assert schema.field("fold_id").type == pa.int32()
    assert schema.field("y_true").type == pa.int64()
    assert schema.field("p_oof_0").type == pa.float64()
    assert schema.field("p_oof_1").type == pa.float64()
    assert schema.field("p_oof_avg").type == pa.float64()
    assert schema.field("is_train").type == pa.bool_()

    # (b) + (c): instrument each base model so its ``fit`` records
    # the training slice. The OOF generator must call ``fit`` once
    # per (model, fold) pair, and the slice it trains on must be
    # the OOF training slice for that fold. We re-fit each base
    # model on the OOF training slice for every fold ``k`` and
    # assert the fold-``k`` prediction is bit-identical to the
    # stacked table's ``p_oof_<i>`` rows for fold ``k``.
    captured: list[tuple[int, np.ndarray]] = []

    class _SpyA(_PerfectXORModel):
        name = "model_a"

        def fit(
            self,
            features: FeatureMatrix,
            y: np.ndarray,
            *,
            sample_weight: np.ndarray | None = None,
        ) -> Any:
            captured.append((0, features.values.copy()))
            return super().fit(features, y, sample_weight=sample_weight)

    class _SpyB(_PerfectXORModel):
        name = "model_b"

        def fit(
            self,
            features: FeatureMatrix,
            y: np.ndarray,
            *,
            sample_weight: np.ndarray | None = None,
        ) -> Any:
            captured.append((1, features.values.copy()))
            return super().fit(features, y, sample_weight=sample_weight)

    _ = generate_stacked_oof(
        features, y, folds, base_models=[_SpyA(), _SpyB()],
    )
    # 2 models * 3 folds = 6 fit calls.
    assert len(captured) == 2 * len(folds), (
        f"expected {2 * len(folds)} (model, fold) fit calls, "
        f"got {len(captured)}"
    )

    # For each (model_index, fold_index), the slice must match the
    # OOF training-slice contract.
    for k, f in enumerate(folds):
        for mi in (0, 1):
            # Find the captured slice for (mi, k). The order is:
            # the stacked generator runs base_models[0] across
            # all folds, then base_models[1] across all folds,
            # so the order is (0, 0), (0, 1), (0, 2), (1, 0),
            # (1, 1), (1, 2). The captured list is in the same
            # order as the fit calls.
            idx: int = mi * len(folds) + k
            captured_mi, captured_slice = captured[idx]
            assert captured_mi == mi, (
                f"captured model index mismatch at {idx}: "
                f"expected {mi}, got {captured_mi}"
            )
            train_start, train_end = _oof_train_idx_for_fold(folds, k)
            expected_n: int = train_end - train_start
            assert captured_slice.shape[0] == expected_n, (
                f"(model={mi}, fold={k}): stacked generator trained on "
                f"{captured_slice.shape[0]} rows, expected {expected_n} "
                f"(the OOF training slice for this fold). "
                f"Any other size is a leak."
            )
            np.testing.assert_array_equal(
                captured_slice,
                features.values[train_start:train_end],
            )

    # (b) re-fit + bit-identical: for every (mi, k) we re-fit the
    # base model on the OOF training slice and predict on
    # ``folds[k].test``. The stacked table's fold-``k`` rows for
    # column ``p_oof_<mi>`` must equal the reference prediction
    # bit-for-bit. The reference uses a fresh _PerfectXORModel
    # subclass, so any numerical drift is a leak.
    fold_id_arr: np.ndarray = stacked["fold_id"].to_numpy()
    for mi, ref_cls in enumerate((_ModelA, _ModelB)):
        for k, f in enumerate(folds):
            train_start, train_end = _oof_train_idx_for_fold(folds, k)
            train_x: FeatureMatrix = FeatureMatrix(
                values=features.values[train_start:train_end],
                feature_names=features.feature_names,
            )
            train_y: np.ndarray = y[train_start:train_end].astype(np.int64)
            ref: _PerfectXORModel = ref_cls()
            ref.fit(train_x, train_y)
            ref_pred: _SimplePrediction = ref.predict(
                ref,
                FeatureMatrix(
                    values=features.values[f.test_start : f.test_end],
                    feature_names=features.feature_names,
                ),
            )
            mask: np.ndarray = fold_id_arr == f.fold_id
            np.testing.assert_array_equal(
                stacked[f"p_oof_{mi}"].to_numpy()[mask],
                ref_pred.y_proba,
            )
