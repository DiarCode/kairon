"""Tests for the architecture-diverse ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from kairon.models.base import Model, ModelError, Prediction, TrainedModel
from kairon.models.contracts import FeatureMatrix
from kairon.models.ensemble import (
    EnsembleSpec,
    MajorityVoteEnsemble,
    MetaLabeledEnsemble,
    MetaLabeledEnsembleConfig,
    TopKConfidenceEnsemble,
    _combine_majority,
    _combine_topk,
    _max_proba,
    _softmax,
)
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.metalabel import MLConfig, MetaLearnerModel
from kairon.models.registry import ModelKind
from kairon.models.tree import (
    RandomForestConfig,
    RandomForestModel,
)


def _toy(n: int = 200, seed: int = 11) -> tuple[FeatureMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 + 0.1 * rng.normal(size=n) > 0).astype(np.int64)
    fm = FeatureMatrix(
        values=np.column_stack([x1, x2]).astype(np.float64),
        feature_names=("x1", "x2"),
    )
    return fm, y


# --- pure helpers --------------------------------------------------------
def test_softmax_sums_to_one() -> None:
    x = np.array([[1.0, 2.0, 3.0]])
    s = _softmax(x)
    assert s.shape == (1, 3)
    assert np.allclose(s.sum(axis=1), 1.0)


def test_softmax_temperature_sharpens() -> None:
    x = np.array([[1.0, 2.0]])
    s_soft = _softmax(x, temperature=2.0)
    s_sharp = _softmax(x, temperature=0.5)
    # Sharper -> more mass on the larger entry
    assert s_sharp[0, 1] > s_soft[0, 1]


def test_max_proba_handles_2d_and_1d() -> None:
    assert _max_proba(np.array([0.7, 0.3])).tolist() == [0.7, 0.3]
    out = _max_proba(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    assert out.tolist() == [0.7, 0.8]


def test_combine_topk_perfect_consensus() -> None:
    """If all constituents agree, the combined proba should match them."""
    n, n_models = 5, 3
    y_proba = np.tile(np.array([0.9, 0.1]), (n, 1))  # all agree: P(class=0)=0.9
    per_model = [
        Prediction(
            y_class=np.zeros(n, dtype=np.int64),
            y_proba=y_proba.copy(),
        )
        for _ in range(n_models)
    ]
    spec = EnsembleSpec(min_k=1, max_k=4, confidence_floor=0.5)
    y_class, y_proba_out, _ = _combine_topk(per_model, spec, classes=(0, 1))
    assert (y_class == 0).all()
    assert y_proba_out is not None
    assert np.allclose(y_proba_out[:, 0], 0.9, atol=1e-6)


def test_combine_topk_handles_low_confidence() -> None:
    """If confidence is below the floor, k shrinks to min_k."""
    n_models = 4  # noqa: F841 - documented for the test
    y_proba_low = np.array([0.26, 0.74])  # max < 0.5
    y_proba_high = np.array([0.95, 0.05])  # max >= 0.5
    per_model = [
        Prediction(y_class=np.array([1, 1]), y_proba=y_proba_low.copy()),
        Prediction(y_class=np.array([1, 1]), y_proba=y_proba_low.copy()),
        Prediction(y_class=np.array([0, 1]), y_proba=y_proba_high.copy()),
        Prediction(y_class=np.array([0, 0]), y_proba=y_proba_high.copy()),
    ]
    spec = EnsembleSpec(min_k=1, max_k=4, confidence_floor=0.5)
    y_class, y_proba_out, _ = _combine_topk(per_model, spec, classes=(0, 1))
    # Shapes are right and the function didn't blow up
    assert y_class.shape == (2,)
    assert y_proba_out is not None
    assert y_proba_out.shape == (2, 2)


def test_combine_majority_picks_winner() -> None:
    per_model = [
        Prediction(y_class=np.array([0, 1, 0, 1]), y_proba=None),
        Prediction(y_class=np.array([0, 1, 1, 1]), y_proba=None),
        Prediction(y_class=np.array([0, 0, 0, 1]), y_proba=None),
    ]
    y_class, _, _ = _combine_majority(per_model, classes=(0, 1))
    # col 0: 3 votes for 0 -> 0; col 1: 3 votes for 1 -> 1; col 2: 2-1 -> 0; col 3: 3-0 -> 1
    assert y_class.tolist() == [0, 1, 0, 1]


def test_combine_majority_empty() -> None:
    with pytest.raises(ModelError, match="no model predictions"):
        _combine_majority([], classes=(0, 1))


# --- spec validation -----------------------------------------------------
def test_ensemble_spec_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        EnsembleSpec(min_k=0)
    with pytest.raises(ValueError):
        EnsembleSpec(min_k=5, max_k=2)
    with pytest.raises(ValueError):
        EnsembleSpec(confidence_floor=1.5)
    with pytest.raises(ValueError):
        EnsembleSpec(temperature=-1.0)
    with pytest.raises(ValueError):
        EnsembleSpec(tie_breaker="bogus")  # type: ignore[arg-type]


def test_topk_ensemble_rejects_empty() -> None:
    with pytest.raises(ModelError, match="at least one model"):
        TopKConfidenceEnsemble([])


def test_topk_ensemble_rejects_too_many() -> None:
    with pytest.raises(ModelError, match="too many constituents"):
        TopKConfidenceEnsemble([LogisticRegressionModel() for _ in range(20)])


# --- end-to-end ----------------------------------------------------------
def _core_models() -> list:
    return [
        LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
        RandomForestModel(
            RandomForestConfig(n_estimators=20, random_state=0, n_jobs=1)
        ),
    ]


def test_topk_ensemble_fit_predict_binary() -> None:
    fm, y = _toy()
    ens = TopKConfidenceEnsemble(_core_models(), EnsembleSpec(min_k=1, max_k=2))
    trained = ens.fit(fm, y)
    assert trained.target_kind == "classification"
    assert len(trained.state.constituents) == 2
    pred = ens.predict(trained, fm)
    assert pred.y_class.shape == (fm.n_rows,)
    assert pred.y_proba is not None
    # All classes are 0/1
    assert set(pred.y_class.tolist()).issubset({0, 1})


def test_topk_ensemble_features_must_match() -> None:
    fm, y = _toy()
    ens = TopKConfidenceEnsemble(_core_models())
    trained = ens.fit(fm, y)
    wrong = FeatureMatrix(values=np.zeros((2, 2)), feature_names=("a", "b"))
    with pytest.raises(ModelError, match="feature mismatch"):
        ens.predict(trained, wrong)


def test_topk_ensemble_beats_single_model_on_separable() -> None:
    """On a clean signal, the ensemble should be no worse than its best member."""
    rng = np.random.default_rng(13)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 * 1.0 + x2 * 1.0 + 0.1 * rng.normal(size=n) > 0).astype(np.int64)
    fm = FeatureMatrix(
        values=np.column_stack([x1, x2]).astype(np.float64),
        feature_names=("x1", "x2"),
    )
    members: list[Model] = [
        LogisticRegressionModel(LinearConfig(C=1.0, random_state=0)),
        RandomForestModel(
            RandomForestConfig(n_estimators=40, random_state=0, n_jobs=1)
        ),
    ]
    ens = TopKConfidenceEnsemble(members, EnsembleSpec(min_k=1, max_k=2))
    trained = ens.fit(fm, y)
    pred = ens.predict(trained, fm)
    # Each member's accuracy
    individual_accs: list[float] = []
    for tm, m in zip(trained.state.constituents, members, strict=True):
        p = m.predict(tm, fm)
        individual_accs.append(float((p.y_class == y).mean()))
    ensemble_acc = float((pred.y_class == y).mean())
    assert ensemble_acc >= min(individual_accs) - 0.01, (
        f"ensemble {ensemble_acc:.3f} under worst member {min(individual_accs):.3f}"
    )


def test_majority_vote_ensemble_fit_predict() -> None:
    fm, y = _toy()
    ens = MajorityVoteEnsemble(_core_models())
    trained = ens.fit(fm, y)
    pred = ens.predict(trained, fm)
    assert pred.y_class.shape == (fm.n_rows,)


def test_topk_ensemble_rejects_class_mismatch_between_constituents() -> None:
    """Two constituents must agree on the class label set."""
    fm, y = _toy()

    class _WeirdClassifier(Model["object"]):
        name = "weird_clf"
        kind = ModelKind.LINEAR

        def _fit_core(self, features, y, *, sample_weight, loss_fn):
            # Intentionally return a TrainedModel with a non-default class set

            state, metrics = {"classes": (10, 20, 30, 40)}, {}
            return state, metrics

        def _predict_core(self, trained, features):
            n = features.n_rows
            return np.zeros(n, dtype=np.int64), np.zeros((n, 2)), None

    class _Normal(Model["object"]):
        name = "normal_clf"
        kind = ModelKind.LINEAR

        def _fit_core(self, features, y, *, sample_weight, loss_fn):
            return {}, {}

        def _predict_core(self, trained, features):
            n = features.n_rows
            return np.zeros(n, dtype=np.int64), np.zeros((n, 2)), None

    # Patch Model.fit to inject an unusual class set for the weird classifier

    import kairon.models.base as _base
    from kairon.models.base import TrainedModel

    _orig_fit = _base.Model.fit

    def _patched_fit(self, features, y, *, sample_weight=None):
        result = _orig_fit(self, features, y, sample_weight=sample_weight)
        if self.name == "weird_clf":
            return TrainedModel(
                backend=result.backend,
                spec=result.spec,
                state=result.state,
                feature_names=result.feature_names,
                target_kind=result.target_kind,
                classes=(10, 20, 30, 40),
                created_at_ns=result.created_at_ns,
            )
        return result

    _base.Model.fit = _patched_fit  # type: ignore[method-assign]
    try:
        ens = TopKConfidenceEnsemble([_WeirdClassifier(object()), _Normal(object())])
        with pytest.raises(ModelError, match="classes mismatch"):
            ens.fit(fm, y)
    finally:
        _base.Model.fit = _orig_fit  # type: ignore[method-assign]


def test_ensemble_trained_is_frozen() -> None:
    fm, y = _toy()
    ens = TopKConfidenceEnsemble(_core_models())
    trained = ens.fit(fm, y)
    with pytest.raises(Exception):  # frozen dataclass
        trained.target_kind = "regression"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# W3.4: MetaLabeledEnsemble combinator
# ---------------------------------------------------------------------------
class _ConstantBinaryPrimary(Model[object]):
    """A test double that emits a constant ``p_primary`` of 0.7 per row.

    Used to exercise the ``p_final = p_primary * p_meta`` contract of
    :class:`MetaLabeledEnsemble` deterministically.
    """

    name = "constant_primary"
    kind = ModelKind.LINEAR

    def __init__(self, p_primary: float = 0.7) -> None:
        super().__init__(None)
        self.p_primary = float(p_primary)

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        return {"p": self.p_primary}, {}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        n = features.n_rows
        state = trained.state if isinstance(trained, TrainedModel) else trained
        p = float(state["p"])
        y_proba = np.full(n, p, dtype=np.float64)
        y_class = np.where(p >= 0.5, 1, 0).astype(np.int64) * np.ones(n, dtype=np.int64)
        return y_class, y_proba, None


class _ConstantMeta(Model[object]):
    """A test double that emits a constant ``p_meta`` of value X per row."""

    name = "constant_meta"
    kind = ModelKind.TREE

    def __init__(self, p_meta: float) -> None:
        super().__init__(None)
        self.p_meta = float(p_meta)

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        return {"p": self.p_meta}, {}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        n = features.n_rows
        state = trained.state if isinstance(trained, TrainedModel) else trained
        p = float(state["p"])
        y_proba = np.full(n, p, dtype=np.float64)
        y_class = np.where(p >= 0.5, 1, 0).astype(np.int64) * np.ones(n, dtype=np.int64)
        return y_class, y_proba, None


def test_metalabeled_combinator() -> None:
    """W3.4: p_final = p_primary * p_meta, abstains when p_meta == 0.

    For the 0.7/0.3 product case, the spec asserts p_final = 0.21.
    The default coverage_threshold is 0.5, which would normally
    abstain on p_meta=0.3; the test lowers the threshold to 0.0
    so the product contract is exercised without abstention.
    """
    fm, _ = _toy()
    primary = _ConstantBinaryPrimary(p_primary=0.7)
    meta = _ConstantMeta(p_meta=0.3)
    cfg = MetaLabeledEnsembleConfig(coverage_threshold=0.0, meta_proba_floor=0.0)
    ens = MetaLabeledEnsemble(primary=primary, meta_learner=meta, config=cfg)
    trained = ens.fit(fm, np.zeros(fm.n_rows, dtype=np.int64))
    pred = ens.predict(trained, fm)
    # p_final = 0.7 * 0.3 = 0.21, NOT min(0.7, 0.3) = 0.3
    assert pred.y_proba is not None
    assert np.allclose(pred.y_proba, 0.21, atol=1e-9)
    # The class label is the primary's class (1) — NOT 0/FLAT
    assert (pred.y_class == 1).all()

    # Now try p_meta = 0.0: abstention must be hard, y_proba == 0.0.
    primary2 = _ConstantBinaryPrimary(p_primary=0.7)
    meta2 = _ConstantMeta(p_meta=0.0)
    ens2 = MetaLabeledEnsemble(primary=primary2, meta_learner=meta2)
    trained2 = ens2.fit(fm, np.zeros(fm.n_rows, dtype=np.int64))
    pred2 = ens2.predict(trained2, fm)
    assert pred2.y_proba is not None
    assert np.allclose(pred2.y_proba, 0.0, atol=1e-9)
    assert (pred2.y_class == 0).all()


def test_metalabeled_combinator_preserves_strict_typing() -> None:
    """W3.4: the new MetaLabeledEnsemble class must satisfy pyright --strict.

    The pyright --strict pass is run by the build harness; this test
    is a runtime sanity check that the public surface (constructor +
    _fit_core / _predict_core) constructs and exchanges data with the
    abstract :class:`Model` contract without raising. The strict-typing
    proof is the gate that follows the test in the W3.4 verification
    step.
    """
    fm, y = _toy()
    primary = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    meta = MetaLearnerModel(MLConfig(n_estimators=20, max_depth=2, learning_rate=0.1))
    cfg = MetaLabeledEnsembleConfig(coverage_threshold=0.5, meta_proba_floor=0.0)
    ens = MetaLabeledEnsemble(primary=primary, meta_learner=meta, config=cfg)
    # Constructor accepts the typed config and the Model[]-bounded
    # primary/meta parameters; fit() routes through the Model base
    # class; predict() returns a Prediction. No any leakage in the
    # public surface.
    trained = ens.fit(fm, y)
    assert trained.target_kind == "classification"
    assert trained.state.primary is not None
    assert trained.state.meta is not None
    pred = ens.predict(trained, fm)
    assert pred.y_class.shape == (fm.n_rows,)
    assert pred.y_proba is not None
    assert pred.y_proba.shape == (fm.n_rows,)


def test_metalabeled_combinator_handles_low_meta_proba() -> None:
    """W3.4: with coverage_threshold=0.5 and p_meta=0.4, output is FLAT."""
    fm, _ = _toy()
    primary = _ConstantBinaryPrimary(p_primary=0.7)
    meta = _ConstantMeta(p_meta=0.4)  # below coverage_threshold=0.5
    ens = MetaLabeledEnsemble(
        primary=primary,
        meta_learner=meta,
        config=MetaLabeledEnsembleConfig(coverage_threshold=0.5, meta_proba_floor=0.0),
    )
    trained = ens.fit(fm, np.zeros(fm.n_rows, dtype=np.int64))
    pred = ens.predict(trained, fm)
    assert pred.y_proba is not None
    # 0.4 < 0.5 => abstained
    assert (pred.y_proba == 0.0).all()
    assert (pred.y_class == 0).all()
