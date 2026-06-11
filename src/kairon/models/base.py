"""Base model contracts.

A ``Model`` is a stateless recipe. Calling ``.fit`` returns an immutable
``TrainedModel`` that callers can persist, version, and load. Predictions
are always returned as a ``Prediction`` so the trainer/evaluator policy
layer can stay backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np
import pyarrow as pa

# Public, project-wide Literal alias for the loss-function choice.
# The Literal set is the load-bearing contract; the runtime check
# (see ``_validate_loss_fn``) is a defence-in-depth measure for callers
# that bypass static type checking.
LossFnName = Literal["cross_entropy", "sharpe", "cost_focal"]
_ALLOWED_LOSS_FN: frozenset[str] = frozenset(("cross_entropy", "sharpe", "cost_focal"))


def _validate_loss_fn(loss_fn: str) -> LossFnName:
    """Validate the loss_fn name against the Literal set.

    The Literal type is the static contract; this runtime check is a
    defence-in-depth measure for callers (CLI / YAML / mlflow) that may
    not be statically type-checked.
    """
    if loss_fn not in _ALLOWED_LOSS_FN:
        raise ValueError(
            f"unknown loss_fn: {loss_fn!r} "
            f"(allowed: {sorted(_ALLOWED_LOSS_FN)})"
        )
    return loss_fn  # type: ignore[return-value]

if TYPE_CHECKING:
    from kairon.models.contracts import FeatureMatrix


@dataclass(frozen=True, slots=True)
class Prediction:
    """Result of a model inference call.

    ``y_class`` and ``y_proba`` have the same row count, in the same order
    as the input features. ``y_proba`` is the class-1 probability for
    binary classification; for multi-class, columns are class indices in
    sorted order. ``y_proba`` may be ``None`` for regression targets.
    """

    y_class: np.ndarray  # shape (n,), int64
    y_proba: np.ndarray | None  # shape (n,) for binary, (n, k) for multiclass, or None
    y_score: np.ndarray | None = None  # raw continuous output (regression / volatility)
    # W6.4 additive multi-head outputs. Both default to None so
    # every existing v1 backend (LogisticRegression, XGBoost, LSTM,
    # NBEATS, the W3 ensembles, the W3.4 MetaLabeledEnsemble) is
    # unaffected. Only the W6.4 :class:`MultiHeadModel` populates
    # these; downstream consumers (sizer, regime detector, paper
    # trader) read them when present and fall back to the v1
    # ``y_proba``/``y_score`` paths when None.
    y_magnitude: np.ndarray | None = None  # shape (n,); magnitude head's regression output (e.g. log-return forecast)
    y_vol: np.ndarray | None = None  # shape (n,); vol head's quantile (pinball) prediction
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    backend: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.y_class.ndim != 1:
            raise ValueError(f"y_class must be 1-D, got shape {self.y_class.shape}")
        if self.y_proba is not None and self.y_proba.shape[0] != self.y_class.shape[0]:
            raise ValueError(
                f"y_proba has {self.y_proba.shape[0]} rows, "
                f"y_class has {self.y_class.shape[0]}"
            )
        if self.y_score is not None and self.y_score.shape[0] != self.y_class.shape[0]:
            raise ValueError(
                f"y_score has {self.y_score.shape[0]} rows, "
                f"y_class has {self.y_class.shape[0]}"
            )
        # W6.4 multi-head shape validation. y_magnitude and y_vol
        # are optional; when present they MUST be 1-D and aligned
        # with y_class. The validation is intentionally lenient
        # (None is always allowed) so the existing v1 callers
        # don't need to know about the new fields.
        if self.y_magnitude is not None:
            if self.y_magnitude.ndim != 1:
                raise ValueError(
                    f"y_magnitude must be 1-D, got shape {self.y_magnitude.shape}"
                )
            if self.y_magnitude.shape[0] != self.y_class.shape[0]:
                raise ValueError(
                    f"y_magnitude has {self.y_magnitude.shape[0]} rows, "
                    f"y_class has {self.y_class.shape[0]}"
                )
        if self.y_vol is not None:
            if self.y_vol.ndim != 1:
                raise ValueError(
                    f"y_vol must be 1-D, got shape {self.y_vol.shape}"
                )
            if self.y_vol.shape[0] != self.y_class.shape[0]:
                raise ValueError(
                    f"y_vol has {self.y_vol.shape[0]} rows, "
                    f"y_class has {self.y_class.shape[0]}"
                )


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Summary of a ``fit`` call — what the trainer logs to mlflow."""

    backend: str
    n_train_rows: int
    n_features: int
    feature_names: tuple[str, ...]
    train_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """An immutable, backend-agnostic container for a fitted model.

    ``state`` is whatever the backend needs to ``predict`` later — a
    fitted sklearn estimator, a torch module's state dict, a path to a
    serialized file, etc. ``feature_names`` records the column order used
    at fit time so predictions can re-order or reject inputs correctly.
    """

    backend: str
    spec: TrainResult
    state: Any  # backend-specific fitted artifact
    feature_names: tuple[str, ...]
    target_kind: str  # "classification" or "regression"
    classes: tuple[int, ...] | None = None  # for classification
    created_at_ns: int = 0  # wall clock, ns since epoch
    artifact_uri: str | None = None  # set by trainer after persistence


class ModelError(RuntimeError):
    """Raised when a model fails to fit or predict."""


class NotFitError(ModelError):
    """Raised when ``predict`` is called before ``fit``."""


ConfigT = TypeVar("ConfigT")


class Model(ABC, Generic[ConfigT]):
    """Abstract base class for every model backend.

    Concrete subclasses set ``name`` and ``kind``, accept a frozen config
    in their ``__init__``, and implement ``_fit_core`` / ``_predict_core``.
    The public ``fit`` / ``predict`` methods add logging, validation, and
    the unified ``TrainedModel`` / ``Prediction`` return types.
    """

    name: str = "abstract"
    kind: str = "abstract"  # "linear" / "tree" / "deep" / "stat"

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    # -- public API --------------------------------------------------------
    def fit(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        loss_fn: LossFnName = "cross_entropy",
    ) -> TrainedModel:
        """Fit the model and return a serializable ``TrainedModel``.

        ``loss_fn`` selects the training objective. The default
        ``"cross_entropy"`` preserves the v1 behaviour; the W5.2 / W5.3
        release wires the ``"sharpe"`` and ``"cost_focal"`` paths to
        their torch implementations. The runtime validation in
        :func:`_validate_loss_fn` rejects unknown values immediately.
        """
        import time

        if features.n_rows == 0:
            raise ModelError("cannot fit on zero rows")
        if y.shape[0] != features.n_rows:
            raise ModelError(
                f"y has {y.shape[0]} rows, features have {features.n_rows}"
            )
        if sample_weight is not None and sample_weight.shape[0] != features.n_rows:
            raise ModelError(
                f"sample_weight has {sample_weight.shape[0]} rows, "
                f"features have {features.n_rows}"
            )
        loss_fn = _validate_loss_fn(loss_fn)

        t0 = time.perf_counter()
        state, metrics = self._fit_core(
            features, y, sample_weight=sample_weight, loss_fn=loss_fn
        )
        elapsed = time.perf_counter() - t0

        target_kind, classes = _infer_target_kind(y)
        spec = TrainResult(
            backend=self.name,
            n_train_rows=features.n_rows,
            n_features=features.n_features,
            feature_names=features.feature_names,
            train_seconds=elapsed,
            metrics=metrics,
            extras={"loss_fn": loss_fn},
        )
        return TrainedModel(
            backend=self.name,
            spec=spec,
            state=state,
            feature_names=features.feature_names,
            target_kind=target_kind,
            classes=classes,
            created_at_ns=time.time_ns(),
        )

    def predict(
        self,
        trained: TrainedModel,
        features: FeatureMatrix,
    ) -> Prediction:
        if trained.backend != self.name:
            raise ModelError(
                f"trained model is {trained.backend!r}, "
                f"this backend is {self.name!r}"
            )
        if features.feature_names != trained.feature_names:
            raise ModelError(
                f"feature mismatch: trained on {trained.feature_names}, "
                f"got {features.feature_names}"
            )
        y_class, y_proba, y_score = self._predict_core(trained.state, features)
        return Prediction(
            y_class=y_class,
            y_proba=y_proba,
            y_score=y_score,
            feature_names=features.feature_names,
            backend=trained.backend,
        )

    # -- abstract hooks ---------------------------------------------------
    @abstractmethod
    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        """Backend-specific fit. Returns ``(state, metrics)``.

        ``loss_fn`` is one of ``"cross_entropy"`` (the v1 default),
        ``"sharpe"``, or ``"cost_focal"``. The :class:`Model.fit` shim
        has already validated the name against the Literal set before
        this hook is called; backends may treat the string as
        advisory metadata (the v1 backends ignore it and continue to
        fit cross-entropy). The W5.2 / W5.3 release will add the
        actual loss-aware branches for the deep-learning backends.
        """

    @abstractmethod
    def _predict_core(
        self,
        trained: TrainedModel,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Backend-specific predict. Returns ``(y_class, y_proba, y_score)``."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _infer_target_kind(y: np.ndarray) -> tuple[str, tuple[int, ...] | None]:
    """Decide whether ``y`` looks like classification or regression.

    Heuristic: if every element is a whole-number int that fits in int64
    and the unique count is small, treat as classification. Otherwise
    regression. The trainer can override via ``target_kind`` if needed.
    """
    arr = np.asarray(y)
    finite_int = np.isfinite(arr) & (arr == np.floor(arr))
    is_int_like = bool(finite_int.all())
    n_unique = int(np.unique(arr[finite_int]).size) if is_int_like else -1
    if is_int_like and 0 < n_unique <= 32:
        classes = tuple(int(v) for v in np.unique(arr[finite_int]))
        return "classification", classes
    return "regression", None


def prediction_to_arrays(pred: Prediction) -> pa.RecordBatch:
    """Serialize a ``Prediction`` to a ``pa.RecordBatch`` for parquet logging."""
    arrays: list[pa.Array] = [pa.array(pred.y_class)]
    fields: list[pa.Field] = [pa.field("y_class", pa.int64())]
    if pred.y_proba is not None:
        if pred.y_proba.ndim == 1:
            arrays.append(pa.array(pred.y_proba))
            fields.append(pa.field("y_proba", pa.float64()))
        else:
            for k in range(pred.y_proba.shape[1]):
                col = pred.y_proba[:, k]
                arrays.append(pa.array(col))
                fields.append(pa.field(f"y_proba_{k}", pa.float64()))
    if pred.y_score is not None:
        arrays.append(pa.array(pred.y_score))
        fields.append(pa.field("y_score", pa.float64()))
    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))
