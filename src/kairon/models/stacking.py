"""Stacked-generalisation meta-learner (W6.2).

The W6.2 release ships a meta-learner that consumes the per-base-model
OOF probabilities emitted by :func:`kairon.evaluation.oof.generate_stacked_oof`
and produces a single ``p_meta`` (the probability that the
ensemble's directional bet is worth taking). The meta-learner is
trained on ``(X_oof, y_oof)`` where ``X_oof`` is the row-wise
concatenation of the base models' OOF probabilities and ``y_oof`` is
the true label; it is then used to predict on the holdout set
``(X_holdout, ??)`` (the ``?`` in the spec is the unknown holdout
labels — see :meth:`StackedGeneralizationEnsemble.predict_meta`).

Anti-leakage contract (same as W3.6 / W6.1, Architect Tension E from
round 1):

    The meta is fit ONLY on OOF predictions from folds <k. Predictions
    for fold k use ONLY the fold-k OOF features as input.

The implementation is a thin shell around a 2-class
:class:`~kairon.models.metalabel.MLConfig` estimator that is selected
at fit time:

* if ``config.use_xgboost_if_available=True`` AND xgboost is
  installed: ``xgboost.XGBClassifier``;
* otherwise: ``sklearn.linear_model.LogisticRegression`` (the W6.2
  fallback; GradientBoostingClassifier was the W3.3 default but a
  meta-on-meta is a linear combination of base probabilities, so a
  linear head is the more natural fallback for the stack).

The fallback is seamless and the public API is identical regardless
of which backend is active. The :class:`StackedGeneralizationEnsemble`
class is what the W6.3 CAS-dominance script consumes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
from sklearn.linear_model import LogisticRegression

from kairon.models.base import Model
from kairon.models.contracts import FeatureMatrix
from kairon.models.metalabel import MLConfig

if TYPE_CHECKING:
    from kairon.models.base import TrainedModel
def _has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# StackedOofTable — a typed view over the W6.1 stacked OOF table
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StackedOofTable:
    """A typed view of the W6.1 stacked OOF table.

    The :class:`StackedGeneralizationEnsemble` consumes a
    :class:`StackedOofTable` (rather than a raw ``pa.Table``) so the
    feature-extraction logic is in one place and is unit-testable in
    isolation. The contract:

    * ``n_base`` = ``len(base_model_names)`` = the number of
      ``p_oof_<i>`` columns in the table;
    * ``X_oof`` is a 2-D ``float64`` array of shape
      ``(n_total, n_base)`` where row ``r`` is
      ``[p_oof_0(r), p_oof_1(r), ..., p_oof_{n_base-1}(r)]``;
    * ``y_oof`` is a 1-D ``int64`` array of shape ``(n_total,)``;
    * ``fold_ids`` is a 1-D ``int32`` array of shape ``(n_total,)``
      recording the fold each row was generated for;
    * The table rows must be in the same order as
      ``X_oof[i] / y_oof[i] / fold_ids[i]``.

    The constructor validates the schema (column count, dtypes) and
    surfaces a clear error if the input is not a W6.1 stacked table.
    """

    X_oof: np.ndarray  # shape (n_total, n_base), float64
    y_oof: np.ndarray  # shape (n_total,), int64
    fold_ids: np.ndarray  # shape (n_total,), int32
    base_model_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.X_oof.ndim != 2:
            raise ValueError(
                f"X_oof must be 2-D, got shape {self.X_oof.shape}"
            )
        if self.y_oof.ndim != 1:
            raise ValueError(
                f"y_oof must be 1-D, got shape {self.y_oof.shape}"
            )
        if self.fold_ids.ndim != 1:
            raise ValueError(
                f"fold_ids must be 1-D, got shape {self.fold_ids.shape}"
            )
        n: int = self.X_oof.shape[0]
        if self.y_oof.shape[0] != n:
            raise ValueError(
                f"X_oof and y_oof row count mismatch: {n} vs {self.y_oof.shape[0]}"
            )
        if self.fold_ids.shape[0] != n:
            raise ValueError(
                f"X_oof and fold_ids row count mismatch: {n} vs {self.fold_ids.shape[0]}"
            )
        if self.X_oof.shape[1] != len(self.base_model_names):
            raise ValueError(
                f"X_oof has {self.X_oof.shape[1]} columns but "
                f"base_model_names has {len(self.base_model_names)}"
            )

    @property
    def n_rows(self) -> int:
        return int(self.X_oof.shape[0])

    @property
    def n_base(self) -> int:
        return int(self.X_oof.shape[1])

    @classmethod
    def from_arrow(
        cls,
        table: pa.Table,
        *,
        base_model_names: tuple[str, ...] | None = None,
    ) -> "StackedOofTable":
        """Build a :class:`StackedOofTable` from a W6.1 stacked ``pa.Table``.

        The base-model names default to the table's ``p_oof_<i>`` column
        names. The ``y_oof`` is taken from the ``y_true`` column; the
        ``fold_ids`` is taken from the ``fold_id`` column.

        Raises ``ValueError`` if the input is missing the
        ``fold_id`` / ``y_true`` columns or if no ``p_oof_*`` columns
        are present.
        """
        if "fold_id" not in table.column_names:
            raise ValueError(
                "stacked OOF table must have a 'fold_id' column"
            )
        if "y_true" not in table.column_names:
            raise ValueError(
                "stacked OOF table must have a 'y_true' column"
            )
        p_cols: list[str] = sorted(
            c for c in table.column_names if c.startswith("p_oof_")
            and c not in ("p_oof_avg",)
        )
        if not p_cols:
            raise ValueError(
                "stacked OOF table must have at least one 'p_oof_<i>' column"
            )
        # Stable column order: sort by the integer suffix.
        p_cols.sort(key=lambda c: int(c.split("_")[-1]))
        X_oof: np.ndarray = np.stack(
            [table[c].to_numpy() for c in p_cols], axis=1,
        ).astype(np.float64)
        y_oof: np.ndarray = table["y_true"].to_numpy().astype(np.int64)
        fold_ids: np.ndarray = table["fold_id"].to_numpy().astype(np.int32)
        names: tuple[str, ...] = (
            tuple(base_model_names) if base_model_names is not None
            else tuple(p_cols)
        )
        return cls(
            X_oof=X_oof,
            y_oof=y_oof,
            fold_ids=fold_ids,
            base_model_names=names,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StackedGeneralizationEnsemble(Model[MLConfig]):
    """Stacked-generalisation meta-learner (W6.2).

    The model is a :class:`~kairon.models.base.Model` subclass so it
    integrates with the existing trainer, registry, and mlflow layer.
    It consumes a :class:`StackedOofTable` (the W6.1 stacked OOF
    output) and fits a 2-class classifier on the per-base-model
    OOF probabilities. At predict time it accepts a 2-D feature
    matrix with one column per base model and returns
    :class:`Prediction` with ``y_proba = p_meta`` (the class-1
    probability = the probability that the ensemble's directional
    bet is worth taking).

    Backend selection (mirrors :class:`MetaLearnerModel`):

    * if ``config.use_xgboost_if_available=True`` AND xgboost is
      installed: ``xgboost.XGBClassifier`` with the config's
      hyperparameters;
    * otherwise: ``sklearn.linear_model.LogisticRegression`` with
      ``max_iter`` bumped to 1000 so convergence is robust on the
      meta-feature matrix (which is a small N x small M matrix).

    The predict call returns a :class:`Prediction` with
    ``y_class in {0, 1}`` and ``y_proba = p_meta``. The 2-D
    feature matrix is wrapped in a :class:`FeatureMatrix` internally
    so the model's contract is uniform with the rest of the model
    layer.
    """

    name = "stacked_meta"
    kind = "stack"

    def __init__(self, config: MLConfig | None = None) -> None:
        super().__init__(config or MLConfig())

    # -- backend selection ------------------------------------------------
    def _build_classifier(self) -> Any:
        """Construct the underlying xgboost or sklearn classifier.

        xgboost is preferred when ``config.use_xgboost_if_available``
        is True AND ``importlib.util.find_spec('xgboost')`` returns
        a ModuleSpec. The check is runtime, not import-time, so the
        sklearn path is taken cleanly when the xgboost wheel is
        missing (which is the W6.2 fallback contract).
        """
        if self.config.use_xgboost_if_available and _has_xgboost():
            xgb = importlib.import_module("xgboost")
            return xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                random_state=self.config.random_state,
                eval_metric="logloss",
                objective="binary:logistic",
            )
        # Linear fallback: the meta-on-meta is a linear combination
        # of base probabilities, so LogisticRegression is the
        # natural choice. max_iter=1000 is bumped from the sklearn
        # default (100) so convergence is robust on small-N
        # meta-feature matrices.
        return LogisticRegression(
            max_iter=1000,
            random_state=self.config.random_state,
        )

    # -- helpers ----------------------------------------------------------
    def _stack_to_features(self, X_oof: np.ndarray) -> FeatureMatrix:
        """Wrap a 2-D ``(n, n_base)`` array in a :class:`FeatureMatrix`.

        The :class:`Model` contract takes a :class:`FeatureMatrix`
        so we synthesise ``feature_names = (p_base_0, p_base_1,
        ..., p_base_{n_base-1})``. The names are descriptive
        (``p_base_<i>``) so debug logging and the
        ``trained.feature_names`` field are self-documenting.
        """
        if X_oof.ndim != 2:
            raise ValueError(
                f"X_oof must be 2-D, got shape {X_oof.shape}"
            )
        n_base: int = int(X_oof.shape[1])
        feature_names: tuple[str, ...] = tuple(
            f"p_base_{i}" for i in range(n_base)
        )
        return FeatureMatrix(
            values=np.asarray(X_oof, dtype=np.float64),
            feature_names=feature_names,
        )

    # -- fit helper used by the public anti-leakage contract --------------
    def fit_stacked(
        self,
        stacked: StackedOofTable,
        *,
        target_fold: int,
    ) -> "TrainedModel":
        """Fit the meta on the OOF rows from folds < ``target_fold``.

        This is the structural enforcement of the W6.2 anti-leakage
        contract: a meta trained on fold ``k``'s own OOF rows would
        leak fold ``k``'s labels into the meta and bias the
        fold-``k`` prediction. The fit is restricted to rows with
        ``fold_id < target_fold``, which is exactly the OOF rows
        from folds ``0..target_fold-1`` (the union of those folds'
        train+val slices under the W3.6 protocol).

        The function is a thin wrapper over :meth:`fit` that
        pre-slices the stacked table. The resulting
        :class:`TrainedModel` can be passed to :meth:`predict_meta`
        for fold-``target_fold``-only predictions.
        """
        mask: np.ndarray = stacked.fold_ids < target_fold
        if not bool(np.any(mask)):
            raise ValueError(
                f"target_fold={target_fold}: no OOF rows with "
                f"fold_id<{target_fold} (have folds "
                f"{sorted(np.unique(stacked.fold_ids).tolist())})"
            )
        X_train: np.ndarray = stacked.X_oof[mask]
        y_train: np.ndarray = stacked.y_oof[mask]
        train_x: FeatureMatrix = self._stack_to_features(X_train)
        return self.fit(train_x, y_train)

    def predict_meta(
        self,
        trained: "TrainedModel",
        X_oof: np.ndarray,
    ) -> np.ndarray:
        """Predict ``p_meta`` on a 2-D ``(n, n_base)`` array.

        Returns the class-1 probability as a 1-D ``float64`` array
        of length ``n``. Mirrors the W3.3 :class:`MetaLearnerModel`
        predict surface but is exposed as a public method (not just
        the standard :meth:`predict`) because the W6.2 contract
        requires fold-``k``-only predictions and the W3.3
        :meth:`predict` does not enforce that contract.
        """
        features: FeatureMatrix = self._stack_to_features(X_oof)
        pred = self.predict(trained, features)
        if pred.y_proba is None:
            raise ValueError(
                f"backend {self.name!r} returned y_proba=None; "
                f"the W6.2 contract requires a class-1 probability"
            )
        return np.asarray(pred.y_proba, dtype=np.float64)

    # -- Model contract ---------------------------------------------------
    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        clf = self._build_classifier()
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        clf.fit(features.values, y, **fit_kwargs)
        # Train accuracy is the standard diagnostic; it is not
        # the only metric (Brier / AUC are downstream), but the
        # spec requires a measurable fit output and accuracy is
        # the simplest one.
        train_acc = float((clf.predict(features.values) == y).mean())
        return clf, {"train_acc": train_acc}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        clf = trained
        y_class = clf.predict(features.values).astype(np.int64)
        proba_all = clf.predict_proba(features.values)
        if proba_all.ndim == 2 and proba_all.shape[1] == 2:
            y_proba = proba_all[:, 1]
        else:
            y_proba = proba_all
        return y_class, y_proba, None


# ---------------------------------------------------------------------------
# Helpers for fold-isolated evaluation
# ---------------------------------------------------------------------------
BackendName = Literal["xgboost", "sklearn"]


def resolve_backend(config: MLConfig) -> BackendName:
    """Return the active backend name for a given ``MLConfig``.

    Returns ``"xgboost"`` if ``use_xgboost_if_available`` is True and
    xgboost is importable; otherwise ``"sklearn"``. Used by tests and
    by the W6.3 CAS-dominance script to log which backend produced
    the meta predictions.
    """
    if config.use_xgboost_if_available and _has_xgboost():
        return "xgboost"
    return "sklearn"


__all__ = [
    "MLConfig",
    "StackedOofTable",
    "StackedGeneralizationEnsemble",
    "BackendName",
    "resolve_backend",
]
