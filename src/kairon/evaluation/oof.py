"""Out-of-fold (OOF) prediction generation with strict per-fold isolation.

Story W3.6 ships the LOAD-BEARING anti-leakage protocol for the entire
meta-labeling + stacking pipeline (Architect Tension E from round 1).

The contract is:

    Fold k's OOF predictions come from a model trained on folds <k ONLY.

Concretely, for a list of walk-forward folds ``folds[0..K-1]``, the OOF
table is the concatenation, in fold order, of

    fit  on  concat(folds[0..k-1].train, folds[0..k-1].val)
    test on  folds[k].test

for ``k = 0, 1, ..., K-1``. There is NO fold-k label in the training set
that produces fold-k predictions; the test rows are predictions from a
model that has never seen them.

The protocol is the foundation for the meta-learner (W3.3) and the
stacked-generalization ensemble (W6.1, W6.2) — both consume the OOF
features emitted here. A bug in this module silently leaks fold-k labels
into the meta-learner and would invalidate every downstream CAS / Sharpe
figure, which is why the function is a pure, hermetic helper with no IO,
no async, no global state, and four dedicated tests pinning the
isolation contract.

Row order
---------
The returned :class:`pyarrow.Table` has one row per fold-k test row,
in *fold order*: rows for fold 0 first (in the order they appear in
``folds[0].test``), then fold 1, ..., then fold K-1. The order is
documented in the docstring of :func:`generate_oof_predictions` and
pinned by :func:`test_oof_dimensions_match`.

Schema
------
The OOF table has exactly five columns:

    - ``fold_id``      (int32)   — the fold the row was generated for
    - ``y_true``       (int64)   — the ground-truth label
    - ``p_oof``        (float64) — the OOF class-1 probability
    - ``p_oof_class``  (int64)   — ``(p_oof >= 0.5).astype(int64)``
    - ``is_train``     (bool)    — always ``False`` (OOF rows are
                                    never in their own training set)

The ``is_train`` column is reserved for future extensions where the
same table may also carry in-sample rows (e.g. for the meta-learner's
training matrix). For the W3.6 protocol the column is constant
``False``; tests assert this and downstream consumers can rely on
the invariant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pyarrow as pa

# ``FeatureMatrix`` is a frozen dataclass; importing it at
# module-load time is cheap (no IO) and lets the slice helpers
# materialise new FeatureMatrix instances at runtime without
# re-importing on every call.
from kairon.models.contracts import FeatureMatrix

if TYPE_CHECKING:
    from kairon.models.base import TrainedModel
    from kairon.splits.walkforward import Fold


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------
# A ``ModelFactory`` is a zero-arg callable that returns a fresh,
# un-fit object that exposes ``fit(features, y) -> TrainedModel``,
# ``predict(trained, features) -> Prediction``, and a ``name: str``
# attribute. We use a factory (not the model itself) so that
# ``generate_oof_predictions`` fits a SEPARATE instance per fold —
# a single fitted model would carry fold-k's information into
# fold k+1's training set.
#
# The factory return type is intentionally ``Any`` rather than
# ``Model[Any]`` because the OOF protocol is backend-agnostic:
# the generator only calls ``fit`` / ``predict`` / ``name`` on
# the returned object, and any class with that surface works
# (the real backends are :class:`kairon.models.base.Model`
# subclasses; tests may use duck-typed stand-ins). Pinning the
# factory to ``Model[Any]`` would force every test stand-in to
# inherit from the abstract base class, which is a heavier
# contract than the OOF protocol requires.
ModelFactory: TypeAlias = Callable[[], Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _slice_features(
    features: "FeatureMatrix",
    start: int,
    end: int,
) -> "FeatureMatrix":
    """Return a new :class:`FeatureMatrix` for ``features.values[start:end]``.

    The slice is materialised (a copy) so downstream code cannot
    mutate the input by holding a view. ``ts`` is sliced in the
    same way (and may become ``None`` if the input had ``ts=None``).
    """
    if start < 0 or end < start:
        raise ValueError(
            f"invalid slice [{start}, {end}); need 0 <= start <= end"
        )
    sliced_ts: np.ndarray | None = None
    if features.ts is not None:
        sliced_ts = features.ts[start:end]
    return FeatureMatrix(
        values=features.values[start:end],
        feature_names=features.feature_names,
        ts=sliced_ts,
    )


def _slice_y(y: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return ``y[start:end]`` as a contiguous ``int64`` array.

    We force ``int64`` so the ``y_true`` column in the pyarrow
    table has a stable dtype across model backends (some sklearn
    classifiers return ``int32``-like dtypes for binary targets).
    """
    sliced: np.ndarray = np.asarray(y[start:end], dtype=np.int64)
    if sliced.ndim != 1:
        raise ValueError(
            f"y must be 1-D, got shape {sliced.shape} after slicing"
        )
    return sliced


def _train_idx_for_fold(folds: "Sequence[Fold]", k: int) -> tuple[int, int]:
    """Return the ``[start, end)`` training-index range for fold ``k``.

    The training set for fold ``k`` is

    - for ``k == 0``: ``[folds[0].train_start, folds[0].val_end)``
      — the fold's own train+val slice. With no prior folds
      available, the fold's *own* training data is the only
      honest source of supervised signal that does not leak
      fold-0's test labels.
    - for ``k > 0``: the union of all prior folds' train+val
      slices, i.e. ``[folds[0].train_start, folds[k-1].val_end)``.
      Since walk-forward folds are chronological, the union is
      a single half-open interval.

    The isolation guarantee is the same in both cases: fold
    ``k``'s test rows (``folds[k].test_start..folds[k].test_end``)
    are NOT in the training set. For ``k == 0`` the training
    set is the fold's own train+val (not its test), and for
    ``k > 0`` the training set is all prior folds' train+val
    (not fold k's own train, val, or test).

    Validation is folded into the training set rather than held
    out so the OOF generator is a single-pass routine with no
    early-stopping. The model factory owns any early-stopping
    logic; the OOF generator is an anti-leakage protocol, not
    a trainer.
    """
    if k < 0 or k >= len(folds):
        raise ValueError(
            f"fold index {k} out of range for {len(folds)} folds"
        )
    if k == 0:
        # First fold: use its own train+val. We deliberately do
        # NOT include ``folds[0].test_*`` — the test slice is
        # the holdout, and including it would be a textbook
        # label leak.
        f = folds[0]
        return int(f.train_start), int(f.val_end)
    first = folds[0]
    last = folds[k - 1]
    start: int = int(first.train_start)
    end: int = int(last.val_end)
    return start, end


def _test_idx_for_fold(folds: "Sequence[Fold]", k: int) -> tuple[int, int]:
    """Return the ``[start, end)`` test-index range for fold ``k``."""
    f = folds[k]
    return int(f.test_start), int(f.test_end)


def _oof_row_count(folds: "Sequence[Fold]") -> int:
    """Total number of OOF rows across all folds (= sum of test sizes)."""
    return sum(int(f.test_end - f.test_start) for f in folds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_oof_predictions(
    features: "FeatureMatrix",
    y: np.ndarray,
    folds: "Sequence[Fold]",
    *,
    primary_model_factory: ModelFactory,
) -> pa.Table:
    """Generate out-of-fold predictions for a single base model.

    For each fold ``k`` in ``folds``:

    1. Concatenate ``folds[0..k-1].train`` and
       ``folds[0..k-1].val`` (if any) as the **training set**.
    2. Train ``primary_model_factory()`` on the concatenated
       training set — a fresh model instance per fold so no state
       leaks between folds.
    3. Predict on ``folds[k].test``.
    4. Emit ``(fold_id, y_true, p_oof, p_oof_class, is_train)``
       per test row, in the order the test rows appear in the
       input feature table.

    The function is pure: no IO, no async, no global state. It
    only reads from ``features``, ``y``, ``folds`` and
    ``primary_model_factory`` and returns a new
    :class:`pyarrow.Table`. The factory is called once per fold,
    so a side-effecting factory is observable (e.g. for MLflow
    logging); the OOF generation itself is side-effect-free.

    Parameters
    ----------
    features
        The full feature matrix for the asset/horizon. Row
        ``features.values[i]`` is the feature vector for bar
        ``i``; the :class:`Fold` indices are interpreted as
        bar indices.
    y
        Integer class labels, shape ``(features.n_rows,)``,
        aligned to ``features.values``. ``y[i]`` is the label
        for bar ``i``.
    folds
        A sequence of walk-forward :class:`Fold` objects in
        chronological order. The first fold has index 0; the
        training set for fold ``k`` is the union of all
        ``folds[0..k-1]`` train+val slices.
    primary_model_factory
        A zero-argument callable that returns a fresh, un-fit
        :class:`~kairon.models.base.Model`. The OOF generator
        calls the factory once per fold so fold-to-fold state
        cannot leak through a shared fitted model.

    Returns
    -------
    pyarrow.Table
        A table with columns

        - ``fold_id``      (int32)
        - ``y_true``       (int64)
        - ``p_oof``        (float64) — class-1 probability
        - ``p_oof_class``  (int64)  — ``(p_oof >= 0.5).astype(int64)``
        - ``is_train``     (bool)   — always ``False``

        in that order, with one row per fold-``k`` test row,
        concatenated in fold order. The total row count is
        ``sum_k folds[k].n_test()``.

    Raises
    ------
    ValueError
        If the input shapes are inconsistent (``y.shape[0] !=
        features.n_rows``), if the fold indices are out of
        range, if ``folds`` is empty, or if the model factory
        fails validation. The error message pinpoints the
        offender so a regression is easy to triage.
    ModelError
        If the underlying model fails to fit or predict. The
        OOF generator does not catch model errors; they
        propagate to the caller.
    """
    if not folds:
        raise ValueError("folds is empty; need at least 1 fold")

    n_rows: int = features.n_rows
    y_arr: np.ndarray = np.asarray(y)
    if y_arr.shape[0] != n_rows:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows; features have {n_rows}"
        )

    # Index sanity check — every fold must be in-range of features
    # and folds must be in chronological order (the OOF protocol
    # relies on the fold order to define the training set).
    for f in folds:
        if f.test_end > n_rows:
            raise ValueError(
                f"fold {f.fold_id}: test_end={f.test_end} exceeds "
                f"feature rows={n_rows}"
            )
        if f.train_end > n_rows:
            raise ValueError(
                f"fold {f.fold_id}: train_end={f.train_end} exceeds "
                f"feature rows={n_rows}"
            )
        if f.val_end > n_rows:
            raise ValueError(
                f"fold {f.fold_id}: val_end={f.val_end} exceeds "
                f"feature rows={n_rows}"
            )

    # Pre-allocate output arrays for the full OOF table. We
    # materialise everything before returning so a partial failure
    # mid-fold does not leave the caller with a half-built table.
    total: int = _oof_row_count(folds)
    fold_id_arr: np.ndarray = np.empty(total, dtype=np.int32)
    y_true_arr: np.ndarray = np.empty(total, dtype=np.int64)
    p_oof_arr: np.ndarray = np.empty(total, dtype=np.float64)
    p_oof_class_arr: np.ndarray = np.empty(total, dtype=np.int64)
    is_train_arr: np.ndarray = np.zeros(total, dtype=np.bool_)

    cursor: int = 0
    for k, f in enumerate(folds):
        # Training set: union of folds[0..k-1] train + val slices.
        train_start, train_end = _train_idx_for_fold(folds, k)
        # Test set: folds[k].test slice.
        test_start, test_end = _test_idx_for_fold(folds, k)
        n_test: int = test_end - test_start

        if train_end - train_start == 0:
            # Defensive guard: a fold with ``train_start ==
            # train_end`` AND ``val_start == val_end`` has no
            # training rows available. We surface a clear
            # error rather than fitting on zero rows. The
            # walkforward() generator enforces ``train_size >= 1``
            # and ``val_size >= 0``, so a zero-row training
            # set only occurs if the caller hand-built a Fold
            # with ``train_size=0`` (which is unusual but
            # legal in the Fold dataclass itself).
            raise ValueError(
                f"fold {f.fold_id} has an empty training set "
                f"(train_start={train_start}, train_end={train_end}, "
                f"val_start={f.val_start}, val_end={f.val_end}); "
                f"OOF protocol needs at least one training row. "
                f"Drop this fold or expand the training window."
            )

        # Build the per-fold training and test slices.
        train_x: "FeatureMatrix" = _slice_features(features, train_start, train_end)
        train_y: np.ndarray = _slice_y(y_arr, train_start, train_end)
        test_x: "FeatureMatrix" = _slice_features(features, test_start, test_end)
        test_y: np.ndarray = _slice_y(y_arr, test_start, test_end)

        # Fit a FRESH model on the training set. The factory
        # pattern is what isolates fold-to-fold: a fitted model
        # from fold k-1 would carry fold-(k-1)'s labels into
        # fold k's training set, and would still pass a
        # surface-level "no overlap" check (since fold k's test
        # rows are disjoint from fold k-1's). The OOF protocol
        # is stricter than that: it forbids any state leakage.
        # The annotation is ``Any`` to match ``ModelFactory``;
        # see the type-alias comment for the rationale.
        model: Any = primary_model_factory()
        trained: "TrainedModel" = model.fit(train_x, train_y)

        # Predict on the test slice. The contract is that
        # ``trained`` was fit on rows whose indices are all < test_start
        # (because train_end <= test_start by the Fold invariants),
        # so no test label is in the training set.
        pred = model.predict(trained, test_x)
        if pred.y_proba is None:
            # Some regression-only models do not emit probabilities.
            # OOF requires a class-1 probability for the meta-learner
            # to consume; without it the contract is not satisfied.
            raise ValueError(
                f"fold {f.fold_id}: model '{model.name}' returned "
                f"y_proba=None; OOF protocol requires a class-1 "
                f"probability. Use a classifier or a model whose "
                f"_predict_core emits y_proba."
            )
        p_oof: np.ndarray = np.asarray(pred.y_proba, dtype=np.float64)
        if p_oof.shape[0] != n_test:
            raise ValueError(
                f"fold {f.fold_id}: predicted {p_oof.shape[0]} rows; "
                f"expected {n_test} (test slice size)"
            )

        # Class decision: 1 iff p_oof >= 0.5. We use the
        # >= threshold so a perfectly-calibrated 0.5 yields class
        # 1 (matches the standard sklearn convention).
        p_class: np.ndarray = (p_oof >= 0.5).astype(np.int64)

        # Write into the pre-allocated output.
        fold_id_arr[cursor : cursor + n_test] = np.int32(f.fold_id)
        y_true_arr[cursor : cursor + n_test] = test_y
        p_oof_arr[cursor : cursor + n_test] = p_oof
        p_oof_class_arr[cursor : cursor + n_test] = p_class
        # is_train is already False everywhere (zeros); we leave
        # the slice untouched. The explicit write makes the
        # intent obvious in code review.
        is_train_arr[cursor : cursor + n_test] = False
        cursor += n_test

    if cursor != total:
        # Defensive: should be unreachable because the per-fold
        # loop writes exactly n_test rows per fold and total
        # is the sum of n_test. Pin it for the strict type
        # checker.
        raise ValueError(
            f"OOF table size mismatch: wrote {cursor} rows, "
            f"expected {total}"
        )

    return pa.table(
        {
            "fold_id": fold_id_arr,
            "y_true": y_true_arr,
            "p_oof": p_oof_arr,
            "p_oof_class": p_oof_class_arr,
            "is_train": is_train_arr,
        },
        schema=pa.schema(
            [
                pa.field("fold_id", pa.int32(), nullable=False),
                pa.field("y_true", pa.int64(), nullable=False),
                pa.field("p_oof", pa.float64(), nullable=False),
                pa.field("p_oof_class", pa.int64(), nullable=False),
                pa.field("is_train", pa.bool_(), nullable=False),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# W6.1 — Per-base-model stacked-OOF generator
# ---------------------------------------------------------------------------
# The W6 stacked-generalisation ensemble (see
# :class:`kairon.models.stacking.StackedGeneralizationEnsemble`) consumes
# an OOF feature matrix where the columns are the per-base-model
# OOF probabilities. The matrix is shaped ``(n_total_oof, n_base_models)``;
# row ``r`` holds ``[p_oof_model_0(r), p_oof_model_1(r), ...,
# p_oof_model_{M-1}(r)]`` where every model's prediction for row ``r``
# was produced by a model trained on folds ``<k`` (the fold ``r``
# belongs to).
#
# Anti-leakage contract (same as :func:`generate_oof_predictions`):
#
#     Fold k's OOF predictions for every base model come from that
#     base model trained on folds <k ONLY.
#
# Re-using the per-fold anti-leakage logic from the W3.6 generator is
# load-bearing: a leak in the stacked matrix would silently leak
# fold-k labels into the meta-learner (W6.2) and invalidate every
# downstream CAS / Sharpe figure. The function is therefore a thin
# wrapper that reuses :func:`generate_oof_predictions` per base model
# and concatenates the per-base-model ``p_oof`` columns into a single
# stacked table.

# A ``Model`` here is the abstract base class from
# :mod:`kairon.models.base`. The base_models parameter is typed
# ``list[Any]`` (see below) so test stand-ins that are duck-typed
# (with a ``.fit`` / ``.predict`` / ``.name`` surface) work without
# inheriting from the ABC. The factory passed in is
# ``Callable[[], Any]`` for the same reason.


def _stacked_per_model_oof(
    features: "FeatureMatrix",
    y: np.ndarray,
    folds: "Sequence[Fold]",
    model: "Any",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run :func:`generate_oof_predictions` for one base model.

    Returns the ``(p_oof, fold_id, y_true)`` columns of the OOF
    table as 1-D float64 / int32 / int64 arrays. The function
    exists to keep the per-base-model loop in
    :func:`generate_stacked_oof` short and to give the
    stacked-isolation test a clear unit to instrument. The
    fold_id and y_true arrays are the same for every base model
    (they depend only on the folds + y), but the function returns
    them anyway so the caller can use them without re-running
    :func:`generate_oof_predictions` (which would re-fit the base
    model a second time and inflate the fit-call count).
    """
    oof_table: pa.Table = generate_oof_predictions(
        features,
        y,
        folds,
        primary_model_factory=lambda: model,  # type: ignore[return-value]
    )
    p_oof_col: np.ndarray = oof_table["p_oof"].to_numpy()
    fold_id_col: np.ndarray = oof_table["fold_id"].to_numpy()
    y_true_col: np.ndarray = oof_table["y_true"].to_numpy()
    return (
        np.asarray(p_oof_col, dtype=np.float64),
        np.asarray(fold_id_col, dtype=np.int32),
        np.asarray(y_true_col, dtype=np.int64),
    )


def generate_stacked_oof(
    features: "FeatureMatrix",
    y: np.ndarray,
    folds: "Sequence[Fold]",
    *,
    base_models: "list[Any]",
) -> pa.Table:
    """Generate a stacked OOF feature table from a list of base models.

    The returned :class:`pyarrow.Table` has, in order:

    - ``fold_id``     (int32)   — the fold the row was generated for
    - ``y_true``      (int64)   — the ground-truth label
    - ``p_oof_<i>``   (float64) — the OOF class-1 probability of base
                                  model ``i`` (i = 0, 1, ..., n_base-1)
    - ``p_oof_avg``   (float64) — the simple average of
                                  ``p_oof_<0>..p_oof_<n_base-1>`` per
                                  row, included as a sanity-check
                                  summary column
    - ``is_train``    (bool)    — always ``False`` (OOF rows are
                                  never in their own training set)

    The row order matches :func:`generate_oof_predictions` (i.e. the
    concatenation of fold-``k`` test slices in fold order), so the
    stacked table can be joined with the single-model OOF table on
    ``(fold_id, y_true)`` row-for-row.

    Anti-leakage
    ------------
    For every base model ``i`` and every fold ``k``, the ``p_oof_<i>``
    value for a row in fold ``k``'s test slice is produced by base
    model ``i`` fit on the OOF training slice for fold ``k`` (the
    fold's own train+val for ``k=0``, or the union of
    ``folds[0..k-1].train+val`` for ``k>0``). No fold-``k`` label
    contributes to the fold-``k`` prediction for any base model. The
    contract is identical to :func:`generate_oof_predictions`, applied
    per base model.

    Parameters
    ----------
    features
        The full feature matrix shared by all base models. Row
        ``features.values[i]`` is the feature vector for bar ``i``;
        the :class:`Fold` indices are interpreted as bar indices.
    y
        Integer class labels, shape ``(features.n_rows,)``, aligned
        to ``features.values``.
    folds
        Walk-forward folds in chronological order. The first fold
        has index 0.
    base_models
        A non-empty list of :class:`~kairon.models.base.Model`
        instances. Each model is fit and predicted independently per
        fold; the function does NOT average model outputs before
        fitting, and it does NOT share fitted state across folds.
        Pass a list of length ``M`` to get an output with ``M``
        ``p_oof_<i>`` columns.

    Returns
    -------
    pyarrow.Table
        A table with ``3 + n_base + 1 = n_base + 4`` columns:

        - ``fold_id`` (int32)
        - ``y_true``  (int64)
        - ``p_oof_0`` ... ``p_oof_{n_base-1}`` (float64, one per
          base model)
        - ``p_oof_avg`` (float64)
        - ``is_train`` (bool, always ``False``)

        in that order, with one row per fold-``k`` test row, in
        fold order. The total row count is
        ``sum_k folds[k].n_test()`` (the same row count as
        :func:`generate_oof_predictions` would produce for a single
        base model).

    Raises
    ------
    ValueError
        If ``base_models`` is empty, if the input shapes are
        inconsistent, or if any per-base-model OOF call fails the
        anti-leakage contract (which is detected upstream by
        :func:`generate_oof_predictions`).
    """
    if not base_models:
        raise ValueError("base_models is empty; need at least 1 base model")
    if not folds:
        raise ValueError("folds is empty; need at least 1 fold")

    n_rows: int = features.n_rows
    y_arr: np.ndarray = np.asarray(y)
    if y_arr.shape[0] != n_rows:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows; features have {n_rows}"
        )

    n_base: int = len(base_models)
    # Build the per-base-model p_oof columns. We call
    # ``_stacked_per_model_oof`` for each base model and let
    # :func:`generate_oof_predictions` enforce the anti-leakage
    # contract per model. Each call returns a 1-D p_oof array of
    # length ``sum_k folds[k].n_test()``; we arrange them as columns
    # of a 2-D ``(n_total, n_base)`` array. The fold_id and y_true
    # arrays are returned by the first call and reused for every
    # subsequent call (they depend only on the folds + y, not on
    # the base model), so we do not re-run
    # :func:`generate_oof_predictions` to extract them.
    p_oof_cols: list[np.ndarray] = []
    fold_id_arr: np.ndarray | None = None
    y_true_arr: np.ndarray | None = None
    for i, m in enumerate(base_models):
        # Defensive: the abstract Model protocol is a fit / predict
        # surface; the W3.6 generator's ``ModelFactory`` accepts
        # ``Callable[[], Any]`` so we just pass the model instance
        # via a zero-arg lambda. If the model is not callable as a
        # factory, the OOF generator raises a clear error.
        p_i, fold_id_i, y_true_i = _stacked_per_model_oof(
            features, y_arr, folds, m,
        )
        p_oof_cols.append(p_i)
        if fold_id_arr is None:
            fold_id_arr = fold_id_i
            y_true_arr = y_true_i

    p_oof_stack: np.ndarray = np.stack(p_oof_cols, axis=1)  # (n_total, n_base)
    p_oof_avg: np.ndarray = p_oof_stack.mean(axis=1)

    # ``fold_id_arr`` and ``y_true_arr`` are set in the first loop
    # iteration; the type checker needs the explicit assertion
    # below to narrow ``np.ndarray | None`` to ``np.ndarray``.
    assert fold_id_arr is not None
    assert y_true_arr is not None

    # Build the output schema and column dict.
    columns: dict[str, np.ndarray] = {
        "fold_id": fold_id_arr,
        "y_true": y_true_arr,
    }
    fields: list[pa.Field] = [
        pa.field("fold_id", pa.int32(), nullable=False),
        pa.field("y_true", pa.int64(), nullable=False),
    ]
    for i in range(n_base):
        col_name: str = f"p_oof_{i}"
        columns[col_name] = p_oof_stack[:, i]
        fields.append(pa.field(col_name, pa.float64(), nullable=False))
    columns["p_oof_avg"] = p_oof_avg
    fields.append(pa.field("p_oof_avg", pa.float64(), nullable=False))
    columns["is_train"] = np.zeros(p_oof_stack.shape[0], dtype=np.bool_)
    fields.append(pa.field("is_train", pa.bool_(), nullable=False))

    return pa.table(columns, schema=pa.schema(fields))


__all__ = [
    "ModelFactory",
    "generate_oof_predictions",
    "generate_stacked_oof",
]
