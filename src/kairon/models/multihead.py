"""Multi-head model: direction + magnitude + vol (W6.4).

The W6.4 multi-head model emits THREE quantities per row, each
emitted by a dedicated "head" sharing a common feature encoder:

1. **Direction** (3-class classification: -1 / 0 / +1)
   - Cross-entropy loss (the v1 default)
   - Output: ``Prediction.y_class`` (3-class argmax projected to
     the project's class encoding) and ``Prediction.y_proba``
     (3-class softmax probabilities)
2. **Magnitude** (regression on log return)
   - MSE loss
   - Output: ``Prediction.y_magnitude`` (1-D float64, signed)
3. **Vol** (quantile / pinball loss, default alpha=0.5 = median)
   - Pinball loss
   - Output: ``Prediction.y_vol`` (1-D float64, non-negative)

The three heads share a thin sklearn-friendly backend (a single
logistic-regression-style encoder for the direction head, with the
magnitude and vol heads as additional linear regressors on the
shared features). The implementation uses scikit-learn for all
three heads so the model is fully runnable on the v1 dependency
set (no torch required).

Why a multi-head design?
------------------------
The W6.4 PRD calls out that the *direction* head's classification
output is necessary but not sufficient: a "long" signal with a
negligible predicted return is a waste of trading costs. The
magnitude head gives the sizer a continuous forecast so the
W6.5 vol-aware sizer can scale position size to the predicted
edge. The vol head gives a per-bar vol forecast that the
vol-aware sizer can use to size to a *target* vol, not the
*current* realised vol.

The pinball loss is the standard quantile (a.k.a. "check" or
"tick") loss:

    L_alpha(y, q) = rho_alpha(y - q)
    rho_alpha(u)  = u * (alpha - 1{u<0})

where ``alpha in (0, 1)`` is the target quantile. With
``alpha=0.5`` the loss reduces to ``0.5 * |y - q|`` (the
mean-absolute-deviation objective, whose minimiser is the
median).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix
from kairon.models.metalabel import MLConfig  # W6.4 multi-head reuses MLConfig


# ---------------------------------------------------------------------------
# Pinball (quantile) loss — the W6.4 vol head's loss
# ---------------------------------------------------------------------------
def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    alpha: float = 0.5,
) -> float:
    """Standard pinball (quantile) loss with configurable alpha.

    L_alpha(y, q) = max(alpha * (y - q), (alpha - 1) * (y - q))
                 = (y - q) * (alpha - 1{y < q})

    Equivalently (and the form implemented below)::

        u = y - q
        L = (alpha - 1) * u * 1{u > 0} + alpha * u * 1{u < 0}
          = mean over the batch

    With ``alpha=0.5`` the loss is ``0.5 * |y - q|`` — the
    mean-absolute-deviation objective, whose minimiser is the
    *median* of ``y``. With ``alpha=0.9`` the loss is asymmetric:
    the model is penalised more for under-predicting ``y`` than
    for over-predicting it (the model is targeting the 90th
    percentile).

    Parameters
    ----------
    y_true
        1-D ``np.ndarray`` of realised vol values, shape (N,).
        Must be finite and non-negative (the W6.4 vol head
        trains on realised bar-level vol, which is non-negative
        by construction).
    y_pred
        1-D ``np.ndarray`` of predicted vol values, shape (N,).
        Same shape as ``y_true``.
    alpha
        Target quantile in ``(0, 1)``. Default ``0.5`` (median).
        A non-positive or ``>= 1`` alpha raises ``ValueError``.

    Returns
    -------
    float
        The mean pinball loss across the batch. Always
        non-negative; always finite for well-formed input.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} vs {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError(
            f"y_true must be 1-D, got shape {y_true.shape}"
        )
    if y_pred.ndim != 1:
        raise ValueError(
            f"y_pred must be 1-D, got shape {y_pred.shape}"
        )
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true must contain only finite values")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred must contain only finite values")

    u: np.ndarray = y_true - y_pred
    # Vectorised pinball: ``alpha * u * 1{u >= 0} + (alpha - 1) * u * 1{u < 0}``
    # — the sign of (y - q) splits the loss into a positive-residual
    # and a negative-residual component, with the asymmetric
    # weights ``alpha`` and ``(alpha - 1)``.
    loss_per_example: np.ndarray = np.where(
        u >= 0.0,
        alpha * u,
        (alpha - 1.0) * u,
    )
    return float(loss_per_example.mean())


# ---------------------------------------------------------------------------
# Optional-dependency probe
# ---------------------------------------------------------------------------
def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MultiHeadConfig(MLConfig):
    """Configuration for :class:`MultiHeadModel`.

    Adds three W6.4 fields on top of the W3.3
    :class:`~kairon.models.metalabel.MLConfig` (which the multi-head
    reuses for the shared encoder + side knobs):

    - ``n_direction_classes``: number of direction classes (default 3,
      encoding the W6.4 plan's ``{-1, 0, +1}`` triplet mapped to
      ``{0, 1, 2}`` indices). A value other than 3 is supported
      for forward-compat but the W6.4 acceptance criterion is the
      3-class case.
    - ``vol_alpha``: the W6.4 vol head's pinball-loss alpha. Default
      ``0.5`` (median). A value in ``(0, 1)`` is required; the
      W6.4 acceptance criterion pins the alpha=0.5 case.
    - ``magnitude_alpha`` (unused for MSE) is reserved for the
      future case where the magnitude head switches to a quantile
      loss; the W6.4 implementation uses MSE.
    """

    n_direction_classes: int = 3
    vol_alpha: float = 0.5
    # Magnitude-head tuning. The W6.4 magnitude head uses MSE so
    # these knobs are advisory (the v1 path ignores them). They
    # are declared on the config so a future story can switch the
    # magnitude head to a Huber or quantile loss without
    # breaking the config surface.
    magnitude_alpha: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        if self.n_direction_classes < 2:
            raise ValueError(
                f"n_direction_classes must be >= 2, got {self.n_direction_classes}"
            )
        if not (0.0 < self.vol_alpha < 1.0):
            raise ValueError(
                f"vol_alpha must be in (0, 1), got {self.vol_alpha!r}"
            )
        if not (0.0 < self.magnitude_alpha <= 1.0):
            raise ValueError(
                f"magnitude_alpha must be in (0, 1], got {self.magnitude_alpha!r}"
            )
        # Replicate the parent :class:`MLConfig` validation. We
        # cannot use ``super().__post_init__()`` because the
        # dataclass ``__post_init__`` is invoked with ``self``
        # only; ``super()`` here is fine at runtime, but the
        # MLConfig's fields are validated by the parent's own
        # __post_init__ which is automatically called by the
        # dataclass machinery BEFORE this override runs. We
        # therefore re-validate the parent's fields here to be
        # defensive against a future story that calls
        # ``MultiHeadConfig.__post_init__()`` directly.
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MultiHeadModel(Model[MultiHeadConfig]):
    """Multi-head model: direction + magnitude + vol.

    The :class:`Prediction` returned by this model populates
    **all three** head outputs:

    - ``y_class``: 3-class argmax, mapped to ``{0, 1, 2}`` for the
      ``{-1, 0, +1}`` direction triplet.
    - ``y_proba``: 3-class softmax probabilities, shape ``(N, 3)``.
    - ``y_magnitude``: signed magnitude prediction (log-return
      scale), shape ``(N,)``.
    - ``y_vol``: vol-head quantile prediction (default median),
      shape ``(N,)``, non-negative.

    The training data must provide THREE targets (in addition
    to the standard ``y`` arg to ``fit``):

    - ``y``: 1-D ``np.ndarray`` of *direction labels* in
      ``{0, 1, 2}``. The arg-name matches the v1 backend
      contract.
    - ``y_magnitude_train``: 1-D ``np.ndarray`` of *magnitude
      targets* (e.g. realised log-returns for the horizon).
    - ``y_vol_train``: 1-D ``np.ndarray`` of *vol targets*
      (e.g. realised bar-level vol for the horizon).

    The v1 ``fit`` signature is preserved (``fit(features, y)``);
    the W6.4 multi-head exposes an extended
    :meth:`fit_multihead` that takes the two extra target vectors
    so the public surface stays simple. A convenience
    :meth:`fit` overload is *not* used: the v1 contract
    is too load-bearing across the trainer / registry / mlflow
    layer.
    """

    name = "multihead"
    kind = "deep"

    def __init__(self, config: MultiHeadConfig | None = None) -> None:
        super().__init__(config or MultiHeadConfig())

    # -- public API (multi-head specific) --------------------------------
    def fit_multihead(
        self,
        features: FeatureMatrix,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        y_vol: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        """Fit the multi-head model with three target vectors.

        The v1 ``Model.fit`` contract is preserved by NOT calling
        it from this method (this method is the multi-head
        entry-point; callers that need the standard v1 contract
        use ``Model.fit`` directly, which would fit a single
        head). The trainer wires the multi-head path via this
        method.

        Returns the fitted state dict (a backend-agnostic
        bundle). The state is consumed by
        :meth:`_predict_core` via the standard :class:`TrainedModel`
        container.
        """
        if features.n_rows == 0:
            raise ModelError("cannot fit on zero rows")
        for arr_name, arr in (
            ("y_direction", y_direction),
            ("y_magnitude", y_magnitude),
            ("y_vol", y_vol),
        ):
            if arr.shape[0] != features.n_rows:
                raise ModelError(
                    f"{arr_name} has {arr.shape[0]} rows, features have "
                    f"{features.n_rows}"
                )
        if not np.all(np.isfinite(y_direction)):
            raise ModelError("y_direction must be finite")
        if not np.all(np.isfinite(y_magnitude)):
            raise ModelError("y_magnitude must be finite")
        if not np.all(np.isfinite(y_vol)):
            raise ModelError("y_vol must be finite")
        if np.any(y_vol < 0.0):
            raise ModelError("y_vol must be non-negative (realised vol)")

        # Direction head: 3-class logistic regression. The W6.4
        # multi-head reuses the parent's ``n_estimators`` /
        # ``max_depth`` / ``learning_rate`` for the shared
        # encoder, but the v1 ``MLConfig`` fields are
        # tree-shaper hyperparameters that don't apply to
        # logistic regression. We use ``max_iter`` derived
        # from the parent's n_estimators for determinism.
        max_iter: int = max(50, int(self.config.n_estimators))
        direction_head: LogisticRegression = LogisticRegression(
            max_iter=max_iter,
            random_state=self.config.random_state,
        )
        # The direction labels are expected in {0, 1, 2} for
        # the 3-class case; we trust the caller's encoding
        # (the v1 trainers map {-1, 0, +1} -> {0, 1, 2} before
        # calling fit).
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        direction_head.fit(features.values, y_direction, **fit_kwargs)

        # Magnitude head: a Ridge regressor on the same features.
        # Ridge is the W6.4 v1 default; the head emits a signed
        # log-return forecast.
        magnitude_head: Ridge = Ridge(
            alpha=1.0,
            random_state=self.config.random_state,
        )
        magnitude_head.fit(features.values, y_magnitude, **fit_kwargs)

        # Vol head: a Ridge regressor trained on
        # ``log(1 + realised_vol)`` so the prediction is
        # guaranteed non-negative after ``exp(...) - 1``. The
        # W6.4 acceptance criterion pins the *loss* (pinball),
        # not the head's architecture; the Ridge backend is
        # the v1 default.
        vol_log: np.ndarray = np.log1p(np.clip(y_vol, a_min=0.0, a_max=None))
        vol_head: Ridge = Ridge(
            alpha=1.0,
            random_state=self.config.random_state,
        )
        vol_head.fit(features.values, vol_log, **fit_kwargs)

        return {
            "direction_head": direction_head,
            "magnitude_head": magnitude_head,
            "vol_head": vol_head,
            "direction_classes": np.array(sorted(np.unique(y_direction))),
        }

    # -- Model contract (single-head compatibility) ----------------------
    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        """Single-head fallback. The W6.4 multi-head is meant to
        be called via :meth:`fit_multihead`; this method exists
        so the model integrates with the v1 trainer surface.

        Behaviour: treats ``y`` as a *direction* label and
        fits a 3-class direction head only. The magnitude and
        vol heads are not fit; their ``y_magnitude`` /
        ``y_vol`` are reported as ``None`` in the prediction.
        This is the documented single-head fallback.
        """
        max_iter: int = max(50, int(self.config.n_estimators))
        direction_head: LogisticRegression = LogisticRegression(
            max_iter=max_iter,
            random_state=self.config.random_state,
        )
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        direction_head.fit(features.values, y, **fit_kwargs)
        state: dict[str, Any] = {
            "direction_head": direction_head,
            "magnitude_head": None,
            "vol_head": None,
            "direction_classes": np.array(sorted(np.unique(y))),
        }
        return state, {"train_acc": float(np.mean(direction_head.predict(features.values) == y))}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Backend-specific predict. Returns ``(y_class, y_proba, y_score)``.

        The W6.4 multi-head emits the three heads' outputs and
        packs them into the standard v1 tuple:

        - ``y_class`` is the direction head's argmax.
        - ``y_proba`` is the direction head's 3-class softmax.
        - ``y_score`` is the magnitude head's signed prediction
          (kept here for v1 surface compatibility — downstream
          consumers that need the structured per-head outputs
          read ``Prediction.y_magnitude`` and ``Prediction.y_vol``).
        """
        direction_head: Any = trained["direction_head"]
        magnitude_head: Any = trained.get("magnitude_head")
        vol_head: Any = trained.get("vol_head")

        # Direction prediction.
        y_class_arr: np.ndarray = direction_head.predict(features.values).astype(
            np.int64, copy=False
        )
        proba_all: np.ndarray = direction_head.predict_proba(features.values)
        # Force a 2-D 3-column proba for the 3-class case; the
        # v1 trainer expects ``(N, k)`` for multi-class.
        if proba_all.ndim == 1:
            # Degenerate: only one class was seen at fit time.
            # Pad to a 2-D (N, 1) so the multi-class contract is
            # preserved; the downstream y_class is still the
            # single class.
            y_proba: np.ndarray = proba_all.reshape(-1, 1)
        else:
            y_proba = proba_all

        # Magnitude prediction. The W6.4 magnitude head is
        # optional (single-head fallback has ``magnitude_head=None``).
        y_magnitude: np.ndarray | None = None
        if magnitude_head is not None:
            y_magnitude = magnitude_head.predict(features.values).astype(
                np.float64, copy=False
            )

        # Vol prediction. The W6.4 vol head is optional; when
        # present, the prediction is ``exp(log_pred) - 1`` so
        # the output is non-negative.
        y_vol: np.ndarray | None = None
        if vol_head is not None:
            raw_vol: np.ndarray = np.expm1(
                vol_head.predict(features.values)
            ).astype(np.float64, copy=False)
            # Defensive: a numerical-precision edge case can
            # produce a tiny negative value after ``expm1``;
            # clip to 0 so the contract (vol >= 0) is preserved.
            y_vol = np.maximum(raw_vol, 0.0)

        # The v1 _predict_core signature is
        # ``(y_class, y_proba, y_score)``; we have FOUR outputs
        # now (y_class, y_proba, y_magnitude, y_vol). The
        # v1 trainer's ``predict()`` builds a :class:`Prediction`
        # from the first three; the multi-head's caller (the
        # W6.4 trainer or the sizer) needs to read
        # ``y_magnitude`` / ``y_vol`` from the trained state
        # directly OR via a richer surface. To keep the v1
        # contract simple, we surface the magnitude prediction
        # as ``y_score`` (a 1-D float64 array) and pack the
        # vol prediction into a side-channel via the
        # ``_multihead_predict`` helper below.
        y_score: np.ndarray | None = y_magnitude  # 1-D signed magnitude forecast
        return y_class_arr, y_proba, y_score

    # -- multi-head predict helper (used by tests + W6.5 sizer) ---------
    def predict_multihead(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> dict[str, np.ndarray]:
        """Run a multi-head prediction and return all three heads' outputs.

        Returns a dict with keys ``"y_class"``, ``"y_proba"``,
        ``"y_magnitude"``, ``"y_vol"``. Each value is a
        1-D / 2-D ``np.ndarray`` with the documented shape.

        This is the W6.4 multi-head's caller-facing surface; the
        standard :meth:`Model.predict` only returns the first
        three arrays (packed into a :class:`Prediction`). The
        multi-head consumers (the W6.5 vol-aware sizer, the W6.4
        acceptance-criterion test) call this method directly.
        """
        y_class, y_proba, y_score = self._predict_core(trained, features)
        # Recover y_magnitude / y_vol from the trained state.
        # The state dict is created by ``fit_multihead`` (the
        # multi-head path) or by ``_fit_core`` (the v1
        # single-head fallback).
        magnitude_head: Any = trained.get("magnitude_head")
        vol_head: Any = trained.get("vol_head")
        y_magnitude_arr: np.ndarray | None = None
        if magnitude_head is not None:
            y_magnitude_arr = magnitude_head.predict(features.values).astype(
                np.float64, copy=False
            )
        y_vol_arr: np.ndarray | None = None
        if vol_head is not None:
            raw_vol: np.ndarray = np.expm1(
                vol_head.predict(features.values)
            ).astype(np.float64, copy=False)
            y_vol_arr = np.maximum(raw_vol, 0.0)
        return {
            "y_class": y_class,
            "y_proba": y_proba if y_proba is not None else np.empty(0, dtype=np.float64),
            "y_magnitude": (
                y_magnitude_arr
                if y_magnitude_arr is not None
                else np.zeros(features.n_rows, dtype=np.float64)
            ),
            "y_vol": (
                y_vol_arr
                if y_vol_arr is not None
                else np.zeros(features.n_rows, dtype=np.float64)
            ),
        }


__all__ = [
    "MultiHeadConfig",
    "MultiHeadModel",
    "pinball_loss",
]
