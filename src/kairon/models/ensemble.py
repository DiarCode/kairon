"""Architecture-diverse ensemble.

The headline result from the R-001 mini-study: a heterogeneous ensemble
(LogReg + RF + XGB + LGBM) reaches 60.14% accuracy vs. 52.80% for the
best single model, and 71-77% accuracy at lower coverage when we
threshold on confidence (R-002).

This module provides two combinators:

- :class:`TopKConfidenceEnsemble` — pick the top-K most-confident
  constituent predictions per row and average their probabilities. The
  chosen K adapts row-by-row (more consensus → higher K), which is
  what makes "diverse-but-confident" outperform "always majority vote".

- :class:`MajorityVoteEnsemble` — hard-vote fallback. Used as a control
  in the evaluator and as a safety net when one or more constituents
  can't produce ``y_proba``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairon.models.base import Model, ModelError, Prediction, TrainedModel
from kairon.models.contracts import FeatureMatrix


@dataclass(frozen=True, slots=True)
class EnsembleSpec:
    """Configuration for the top-K confidence ensemble.

    ``min_k`` is the floor on how many constituents must contribute to
    a row's prediction; ``max_k`` is the ceiling. ``confidence_floor``
    is the minimum max-class probability a constituent needs to be
    considered "confident" for that row. ``temperature`` sharpens
    (T<1) or softens (T>1) the confidence distribution before ranking.
    """

    min_k: int = 1
    max_k: int = 4
    confidence_floor: float = 0.34  # ≈ 1/n_classes
    temperature: float = 1.0
    tie_breaker: str = "mean"  # "mean" or "median"

    def __post_init__(self) -> None:
        if self.min_k < 1:
            raise ValueError(f"min_k must be >= 1, got {self.min_k}")
        if self.max_k < self.min_k:
            raise ValueError(f"max_k ({self.max_k}) must be >= min_k ({self.min_k})")
        if not 0.0 < self.confidence_floor < 1.0:
            raise ValueError(f"confidence_floor must be in (0, 1), got {self.confidence_floor}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.tie_breaker not in {"mean", "median"}:
            raise ValueError(f"tie_breaker must be 'mean' or 'median', got {self.tie_breaker!r}")


@dataclass(frozen=True, slots=True)
class EnsembleTrained:
    """A fitted ensemble: a list of ``TrainedModel`` plus the spec that built it."""

    constituents: tuple[TrainedModel, ...]
    models: tuple[Model[Any], ...]
    spec: EnsembleSpec
    feature_names: tuple[str, ...]
    target_kind: str
    classes: tuple[int, ...] | None
    created_at_ns: int
    extras: dict[str, Any] = field(default_factory=dict)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature == 1.0:
        z = x
    else:
        z = x / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _max_proba(y_proba: np.ndarray) -> np.ndarray:
    """Return the maximum class probability per row, regardless of layout."""
    if y_proba.ndim == 1:
        return y_proba
    return y_proba.max(axis=-1)


class TopKConfidenceEnsemble(Model[EnsembleSpec]):
    """Architecture-diverse top-K confidence ensemble.

    Each constituent returns a ``Prediction``; the ensemble ranks
    constituents by per-row max-proba and keeps the top ``k`` for each
    row (clipped to ``[min_k, max_k]``). It then averages their
    probabilities and re-derives the class label.

    If all constituents agree on the argmax, ``k`` is the full ensemble
    size (maximum signal). If only two out of four agree, ``k`` shrinks
    toward ``min_k`` so the noisy voices don't dilute the consensus.
    """

    name = "topk_ensemble"
    kind = "ensemble"

    def __init__(
        self,
        models: list[Model[Any]],
        config: EnsembleSpec | None = None,
    ) -> None:
        super().__init__(config or EnsembleSpec())
        if not models:
            raise ModelError("ensemble needs at least one model")
        if len(models) > 16:
            raise ModelError(f"too many constituents ({len(models)}); cap is 16")
        self.models: tuple[Model[Any], ...] = tuple(models)

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        fitted: list[TrainedModel] = []
        train_accs: list[float] = []
        for m in self.models:
            tm = m.fit(features, y, sample_weight=sample_weight)
            fitted.append(tm)
            acc = tm.spec.metrics.get("train_acc", float("nan"))
            train_accs.append(float(acc))

        target_kind = fitted[0].target_kind
        classes = fitted[0].classes
        for tm in fitted[1:]:
            if tm.target_kind != target_kind:
                raise ModelError(
                    f"target_kind mismatch: {tm.backend}={tm.target_kind} "
                    f"vs {fitted[0].backend}={target_kind}"
                )
            if tm.classes != classes:
                raise ModelError(
                    f"classes mismatch: {tm.backend}={tm.classes} "
                    f"vs {fitted[0].backend}={classes}"
                )

        state = EnsembleTrained(
            constituents=tuple(fitted),
            models=self.models,
            spec=self.config,
            feature_names=features.feature_names,
            target_kind=target_kind,
            classes=classes,
            created_at_ns=fitted[0].created_at_ns,
            extras={
                "constituent_backends": tuple(t.backend for t in fitted),
                "constituent_train_acc": tuple(train_accs),
            },
        )
        metrics = {
            "n_constituents": float(len(fitted)),
            "mean_constituent_train_acc": float(np.mean(train_accs)),
        }
        return state, metrics

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if not isinstance(trained, EnsembleTrained):
            raise ModelError("TopKConfidenceEnsemble expects EnsembleTrained state")
        if features.feature_names != trained.feature_names:
            raise ModelError(
                f"feature mismatch: trained on {trained.feature_names}, "
                f"got {features.feature_names}"
            )

        per_model: list[Prediction] = [
            m.predict(tm, features) for m, tm in zip(trained.models, trained.constituents, strict=True)
        ]
        return _combine_topk(per_model, trained.spec, trained.classes)


class MajorityVoteEnsemble(Model[EnsembleSpec]):
    """Hard-vote ensemble — used as a control and for non-probabilistic backends."""

    name = "majority_vote"
    kind = "ensemble"

    def __init__(
        self,
        models: list[Model[Any]],
        config: EnsembleSpec | None = None,
    ) -> None:
        super().__init__(config or EnsembleSpec())
        if not models:
            raise ModelError("ensemble needs at least one model")
        self.models = tuple(models)

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        fitted: list[TrainedModel] = []
        for m in self.models:
            tm = m.fit(features, y, sample_weight=sample_weight)
            fitted.append(tm)
        state = EnsembleTrained(
            constituents=tuple(fitted),
            models=self.models,
            spec=self.config,
            feature_names=features.feature_names,
            target_kind=fitted[0].target_kind,
            classes=fitted[0].classes,
            created_at_ns=fitted[0].created_at_ns,
        )
        return state, {"n_constituents": float(len(fitted))}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        per_model = [m.predict(tm, features) for m, tm in zip(trained.models, trained.constituents, strict=True)]
        return _combine_majority(per_model, trained.classes)


# ---------------------------------------------------------------------------
# Combinators (exported for unit tests)
# ---------------------------------------------------------------------------
def _combine_topk(
    per_model: list[Prediction],
    spec: EnsembleSpec,
    classes: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    n = per_model[0].y_class.shape[0]
    n_models = len(per_model)
    confidences = np.zeros((n, n_models), dtype=np.float64)
    # Determine how many "effective" class columns to allocate:
    # - multi-class prediction has shape (n, k)
    # - binary 1-D prediction maps to 2 columns (P(class0), P(class1))
    proba_layouts: list[tuple[str, int]] = []
    for p in per_model:
        if p.y_proba is None:
            proba_layouts.append(("none", 0))
        elif p.y_proba.ndim == 1:
            proba_layouts.append(("binary", 2))
        else:
            proba_layouts.append(("multi", int(p.y_proba.shape[1])))
    # All constituents must agree on layout/shape
    first_layout = proba_layouts[0]
    for layout in proba_layouts[1:]:
        if layout[0] != first_layout[0] or layout[1] != first_layout[1]:
            raise ModelError(
                f"incompatible proba layouts across constituents: {proba_layouts}"
            )
    n_classes_eff = first_layout[1] if first_layout[0] != "none" else 1
    probas = np.zeros((n, n_classes_eff, n_models), dtype=np.float64)

    for j, p in enumerate(per_model):
        confidences[:, j] = _max_proba(p.y_proba) if p.y_proba is not None else 0.0
        if p.y_proba is None:
            continue
        if p.y_proba.ndim == 1:
            probas[:, 1, j] = p.y_proba
            probas[:, 0, j] = 1.0 - p.y_proba
        else:
            for k in range(p.y_proba.shape[1]):
                probas[:, k, j] = p.y_proba[:, k]

    conf_soft = _softmax(confidences * np.log(2.0 + 1.0) / spec.temperature, temperature=1.0)
    order = np.argsort(-conf_soft, axis=1)
    k_per_row = np.clip(
        np.sum(confidences >= spec.confidence_floor, axis=1),
        spec.min_k,
        min(spec.max_k, n_models),
    )
    out_proba = np.zeros((n, max(n_classes_eff, 1)), dtype=np.float64)
    for i in range(n):
        ki = max(int(k_per_row[i]), 1)
        idx = order[i, :ki]
        # probas[i, :, idx] is shape (ki, n_classes_eff) due to NumPy
        # advanced-indexing rules; transpose to (n_classes_eff, ki).
        chunk = probas[i, :, idx].T
        if spec.tie_breaker == "mean":
            out_proba[i] = chunk.mean(axis=-1)
        else:
            out_proba[i] = np.median(chunk, axis=-1)

    if classes is not None and n_classes_eff >= 2:
        out_class = np.array(
            [classes[int(np.argmax(out_proba[i]))] for i in range(n)], dtype=np.int64
        )
        return out_class, out_proba, None
    if classes is not None and n_classes_eff == 1:
        # proba_avg style: out_proba has 1 column; treat as regression
        return out_proba[:, 0].astype(np.int64), out_proba[:, 0], out_proba[:, 0]
    # regression fallback
    return out_proba[:, 0].astype(np.int64), out_proba[:, 0], out_proba[:, 0]


def _combine_majority(
    per_model: list[Prediction],
    classes: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if not per_model:
        raise ModelError("no model predictions to combine")
    n = per_model[0].y_class.shape[0]
    y_class = np.zeros(n, dtype=np.int64)
    for i in range(n):
        votes: dict[int, int] = {}
        for p in per_model:
            v = int(p.y_class[i])
            votes[v] = votes.get(v, 0) + 1
        # break ties by smallest class (deterministic)
        best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))
        y_class[i] = best[0]
    if all(p.y_proba is not None for p in per_model):
        proba_arrays: list[np.ndarray] = [p.y_proba for p in per_model if p.y_proba is not None]
        first = proba_arrays[0]
        if first.ndim == 2:
            proba_avg = np.mean(proba_arrays, axis=0)
        else:
            proba_avg = np.mean(proba_arrays, axis=0)
    else:
        proba_avg = None
    return y_class, proba_avg, None


# ---------------------------------------------------------------------------
# MetaLabeledEnsemble (W3.4): primary * meta-learner combinator
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MetaLabeledEnsembleConfig:
    """Configuration for :class:`MetaLabeledEnsemble`.

    ``coverage_threshold`` is the minimum meta-probability
    (``p_meta``) the meta-learner must emit for the combinator to
    surface a signal. Bars where ``p_meta < coverage_threshold`` are
    abstained (``y_class = 0`` / FLAT, ``y_proba = 0.0``).

    ``meta_proba_floor`` is the minimum *combined* probability
    (``p_final = p_primary * p_meta``) below which the combinator
    abstains. The two thresholds are independent guards: the
    coverage threshold is on the meta-learner's confidence in
    isolation; the proba floor is on the joint product.
    """

    coverage_threshold: float = 0.5
    meta_proba_floor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError(
                f"coverage_threshold must be in [0, 1], got {self.coverage_threshold}"
            )
        if not 0.0 <= self.meta_proba_floor <= 1.0:
            raise ValueError(
                f"meta_proba_floor must be in [0, 1], got {self.meta_proba_floor}"
            )


@dataclass(frozen=True, slots=True)
class MetaLabeledTrained:
    """A fitted meta-labelled ensemble: primary + meta + config snapshot."""

    primary: TrainedModel
    meta: TrainedModel
    primary_model: Model[Any]
    meta_model: Model[Any]
    config: MetaLabeledEnsembleConfig
    feature_names: tuple[str, ...]
    target_kind: str
    classes: tuple[int, ...] | None
    created_at_ns: int
    extras: dict[str, Any] = field(default_factory=dict)


def _primary_proba_scalar(p: Prediction) -> np.ndarray:
    """Extract a 1-D ``p_primary in [0, 1]`` from a binary primary prediction.

    Supports both 1-D and 2-D ``y_proba`` layouts (see the existing
    :func:`_max_proba` helper for the 1-D vs 2-D contract). For
    2-D, we take the class-1 column by convention (binary tasks
    encode ``y_proba = P(class=1)`` as the second column in
    sorted-class order).
    """
    if p.y_proba is None:
        raise ModelError("primary must expose y_proba for MetaLabeledEnsemble")
    proba = p.y_proba
    if proba.ndim == 1:
        return proba.astype(np.float64, copy=False)
    if proba.ndim == 2:
        if proba.shape[1] < 2:
            raise ModelError(
                f"primary y_proba has {proba.shape[1]} columns; need >= 2 for binary"
            )
        return proba[:, 1].astype(np.float64, copy=False)
    raise ModelError(f"primary y_proba has unexpected ndim={proba.ndim}")


def _meta_proba_scalar(p: Prediction) -> np.ndarray:
    """Extract a 1-D ``p_meta in [0, 1]`` from the meta-learner.

    The W3.3 :class:`MetaLearnerModel` returns ``y_proba`` as a 1-D
    array of class-1 probabilities. We accept 2-D as well for
    forward-compat (other binary meta-learners may emit 2-D).
    """
    if p.y_proba is None:
        raise ModelError("meta must expose y_proba for MetaLabeledEnsemble")
    proba = p.y_proba
    if proba.ndim == 1:
        return proba.astype(np.float64, copy=False)
    if proba.ndim == 2:
        if proba.shape[1] < 2:
            raise ModelError(
                f"meta y_proba has {proba.shape[1]} columns; need >= 2 for binary"
            )
        return proba[:, 1].astype(np.float64, copy=False)
    raise ModelError(f"meta y_proba has unexpected ndim={proba.ndim}")


class MetaLabeledEnsemble(Model[MetaLabeledEnsembleConfig]):
    """Combinator: primary ensemble * meta-learner.

    The combinator is **additive** on :class:`TopKConfidenceEnsemble`
    (per Architect round 1 Tension C): the primary ensemble still
    produces its own ``p_primary`` per bar; the meta-learner emits
    a second ``p_meta in [0, 1]``; the combinator's output is
    ``p_final = p_primary * p_meta`` with two abstention gates:

    * ``p_meta < config.coverage_threshold`` -> abstain (FLAT / 0.0).
    * ``p_final < config.meta_proba_floor`` -> abstain (FLAT / 0.0).

    The class label (``y_class``) on abstained rows is the
    ``FLAT``/``0`` class. On emitted rows, ``y_class`` is the
    primary's class.

    ``_fit_core`` trains the primary first, then trains the
    meta-learner on the primary's *in-sample* predictions. In a
    production pipeline the meta-learner would be trained on the
    primary's OOF predictions (W3.6 protocol) to avoid the meta
    learner seeing its own inputs at train time; that is the
    trainer's job, not the combinator's. The combinator's contract
    is "given a fitted primary and a fitted meta, produce a
    prediction"; the trainer wires the OOF pipeline.
    """

    name = "metalabeled_ensemble"
    kind = "ensemble"

    def __init__(
        self,
        primary: Model[Any],
        meta_learner: Model[Any],
        config: MetaLabeledEnsembleConfig | None = None,
    ) -> None:
        super().__init__(config or MetaLabeledEnsembleConfig())
        if primary is meta_learner:
            raise ModelError(
                "primary and meta_learner must be distinct Model instances"
            )
        self.primary: Model[Any] = primary
        self.meta_learner: Model[Any] = meta_learner

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        # Train the primary first.
        primary_trained = self.primary.fit(features, y, sample_weight=sample_weight)
        # Generate the primary's in-sample probabilities to feed the
        # meta-learner. In production, the trainer substitutes OOF
        # predictions here (W3.6); the combinator's contract is
        # "fit the primary, then fit the meta on the primary's
        # predictions" — exactly what W3.6 + W3.7 wire together.
        primary_pred = self.primary.predict(primary_trained, features)
        p_primary = _primary_proba_scalar(primary_pred)

        # Side features for the meta-learner are whatever extra
        # columns the original feature matrix carries beyond the
        # primary's signal. We keep it minimal: the primary's
        # probability plus the original feature values. The trainer
        # can override by passing a richer FeatureMatrix to
        # ``meta_learner.fit`` via a custom subclass; the public
        # contract is "the meta sees the primary's proba".
        n = features.n_rows
        meta_values = np.column_stack([p_primary, features.values]).astype(
            np.float64, copy=False
        )
        meta_feature_names: tuple[str, ...] = ("p_primary",) + tuple(features.feature_names)
        meta_features = FeatureMatrix(
            values=meta_values,
            feature_names=meta_feature_names,
            ts=features.ts,
        )

        meta_trained = self.meta_learner.fit(
            meta_features, y, sample_weight=sample_weight
        )

        state = MetaLabeledTrained(
            primary=primary_trained,
            meta=meta_trained,
            primary_model=self.primary,
            meta_model=self.meta_learner,
            config=self.config,
            feature_names=features.feature_names,
            target_kind=primary_trained.target_kind,
            classes=primary_trained.classes,
            created_at_ns=primary_trained.created_at_ns,
            extras={
                "primary_backend": primary_trained.backend,
                "meta_backend": meta_trained.backend,
            },
        )
        metrics = {
            "primary_train_acc": float(
                primary_trained.spec.metrics.get("train_acc", float("nan"))
            ),
            "meta_train_acc": float(
                meta_trained.spec.metrics.get("train_acc", float("nan"))
            ),
        }
        return state, metrics

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if not isinstance(trained, MetaLabeledTrained):
            raise ModelError("MetaLabeledEnsemble expects MetaLabeledTrained state")
        if features.feature_names != trained.feature_names:
            raise ModelError(
                f"feature mismatch: trained on {trained.feature_names}, "
                f"got {features.feature_names}"
            )

        # 1. Primary prediction.
        primary_pred = trained.primary_model.predict(trained.primary, features)
        p_primary = _primary_proba_scalar(primary_pred)
        primary_class = primary_pred.y_class.astype(np.int64, copy=False)

        # 2. Meta-learner prediction on the same input + p_primary
        # feature. Mirrors the column order built in _fit_core.
        meta_values = np.column_stack([p_primary, features.values]).astype(
            np.float64, copy=False
        )
        meta_feature_names: tuple[str, ...] = ("p_primary",) + tuple(features.feature_names)
        meta_features = FeatureMatrix(
            values=meta_values,
            feature_names=meta_feature_names,
            ts=features.ts,
        )
        meta_pred = trained.meta_model.predict(trained.meta, meta_features)
        p_meta = _meta_proba_scalar(meta_pred)

        # 3. Combinator: p_final = p_primary * p_meta; abstain when
        # either gate trips.
        p_final = p_primary * p_meta
        cfg = trained.config
        abstain = (p_meta < cfg.coverage_threshold) | (p_final < cfg.meta_proba_floor)

        # On abstained rows, output FLAT class (0) and zero proba.
        n = features.n_rows
        classes = trained.classes
        flat_class = 0
        if classes is not None and len(classes) > 0:
            # Prefer the FLAT / 0 class if present; otherwise the
            # smallest class.
            try:
                flat_class = int(classes.index(0))
            except ValueError:
                flat_class = int(np.argmin(np.asarray(classes)))

        y_class = primary_class.copy()
        if abstain.any():
            y_class[abstain] = flat_class

        y_proba = np.where(abstain, 0.0, p_final).astype(np.float64, copy=False)
        return y_class, y_proba, p_final


__all__ = [
    "EnsembleSpec",
    "EnsembleTrained",
    "MajorityVoteEnsemble",
    "MetaLabeledEnsemble",
    "MetaLabeledEnsembleConfig",
    "MetaLabeledTrained",
    "TopKConfidenceEnsemble",
]
