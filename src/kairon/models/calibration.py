"""Probability calibration.

After a model produces ``y_proba``, we often want to map those
probabilities to the true observed frequencies. This is most useful for
*boosting* models (LightGBM, XGBoost), which are known to be
over-confident on direction predictions.

We provide two calibrators:

- :class:`IsotonicCalibrator` — non-parametric, monotone, robust to
  arbitrary miscalibration. Best when the calibration set is medium
  (n ≥ 1000). State is a fitted :class:`sklearn.isotonic.IsotonicRegression`.

- :class:`PlattCalibrator` — parametric, fits a 1-D logistic on the
  model's logit. Best when the calibration set is small (n < 1000).
  State is the ``(a, b)`` coefficients of ``sigmoid(a*logit + b)``.

Both calibrators are themselves :class:`Model` subclasses, so they fit
into the same ``fit`` / ``predict`` API and can be persisted the same
way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from kairon.models.base import Model, ModelError, Prediction, TrainedModel
from kairon.models.contracts import FeatureMatrix


@dataclass(frozen=True, slots=True)
class IsotonicConfig:
    """Hyperparameters for isotonic calibration."""

    out_of_bounds: str = "clip"  # "clip" or "nan"
    y_min: float = 1e-6
    y_max: float = 1.0 - 1e-6
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.out_of_bounds not in {"clip", "nan"}:
            raise ValueError(f"out_of_bounds must be 'clip' or 'nan', got {self.out_of_bounds!r}")
        if not 0.0 < self.y_min < self.y_max < 1.0:
            raise ValueError(
                f"need 0 < y_min ({self.y_min}) < y_max ({self.y_max}) < 1"
            )


class IsotonicCalibrator(Model[IsotonicConfig]):
    """Isotonic regression calibrator.

    Use this when the calibration set is reasonably large (n ≥ 1000).
    It assumes the score being calibrated is a 1-D probability (binary
    classification) — the ``y_proba[:, 1]`` of a binary model, or the
    max-proba of a multi-class model collapsed to one dimension.
    """

    name = "isotonic"
    kind = "calibration"

    def __init__(self, config: IsotonicConfig | None = None) -> None:
        super().__init__(config or IsotonicConfig())

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        if features.n_features != 1:
            raise ModelError(
                f"IsotonicCalibrator expects 1 score column, got {features.n_features}"
            )
        scores = features.values[:, 0]
        # Squeeze scores into (0, 1) for the regression to behave.
        scores_c = np.clip(scores, self.config.y_min, self.config.y_max)
        # y must be 0/1 — calibrators operate on binary correctness,
        # not the original multi-class label.
        y_bin = (y == np.unique(y)[-1]).astype(np.int64)
        iso = IsotonicRegression(
            out_of_bounds=self.config.out_of_bounds,  # type: ignore[arg-type]
            y_min=self.config.y_min,
            y_max=self.config.y_max,
        )
        if sample_weight is not None:
            iso.fit(scores_c, y_bin, sample_weight=sample_weight)
        else:
            iso.fit(scores_c, y_bin)
        # In-sample Brier score
        p_cal = iso.predict(scores_c)
        brier = float(np.mean((p_cal - y_bin) ** 2))
        state = {"iso": iso, "y_max_class": int(np.unique(y)[-1])}
        return state, {"train_brier": brier}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        state = trained
        scores = features.values[:, 0]
        scores_c = np.clip(scores, self.config.y_min, self.config.y_max)
        p_cal = np.asarray(state["iso"].predict(scores_c), dtype=np.float64)
        y_class = np.where(p_cal >= 0.5, state["y_max_class"], 0).astype(np.int64)
        # Map binary correctness back to the original class scheme
        return y_class, p_cal, None


@dataclass(frozen=True, slots=True)
class PlattConfig:
    """Hyperparameters for Platt (sigmoid) calibration."""

    C: float = 1.0
    max_iter: int = 200
    y_min: float = 1e-6
    y_max: float = 1.0 - 1e-6
    extras: dict[str, Any] = field(default_factory=dict)


class PlattCalibrator(Model[PlattConfig]):
    """Platt scaling calibrator.

    Use this when the calibration set is small (n < 1000) or you need a
    smooth, parametric calibration. As with :class:`IsotonicCalibrator`,
    the input is a single 1-D score column.
    """

    name = "platt"
    kind = "calibration"

    def __init__(self, config: PlattConfig | None = None) -> None:
        super().__init__(config or PlattConfig())

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        if features.n_features != 1:
            raise ModelError(
                f"PlattCalibrator expects 1 score column, got {features.n_features}"
            )
        scores = features.values[:, 0]
        scores_c = np.clip(scores, self.config.y_min, self.config.y_max)
        # Use the logit of the score
        logit = np.log(scores_c / (1.0 - scores_c)).reshape(-1, 1)
        y_bin = (y == np.unique(y)[-1]).astype(np.int64)
        clf = LogisticRegression(C=self.config.C, max_iter=self.config.max_iter)
        if sample_weight is not None:
            clf.fit(logit, y_bin, sample_weight=sample_weight)
        else:
            clf.fit(logit, y_bin)
        p_cal = clf.predict_proba(logit)[:, 1]
        brier = float(np.mean((p_cal - y_bin) ** 2))
        state = {"clf": clf, "y_max_class": int(np.unique(y)[-1])}
        return state, {"train_brier": brier}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        state = trained
        scores = features.values[:, 0]
        scores_c = np.clip(scores, self.config.y_min, self.config.y_max)
        logit = np.log(scores_c / (1.0 - scores_c)).reshape(-1, 1)
        p_cal = state["clf"].predict_proba(logit)[:, 1]
        y_class = np.where(p_cal >= 0.5, state["y_max_class"], 0).astype(np.int64)
        return y_class, p_cal, None


def calibrated_proba(
    pred: Prediction,
    calibrator: Model[IsotonicConfig] | Model[PlattConfig],
    calibrator_state: TrainedModel,
) -> np.ndarray:
    """Apply a fitted calibrator to a fresh prediction.

    This is a convenience for the trainer: build a feature matrix from
    the model's max-proba, call the calibrator, and return the
    recalibrated 1-D probability vector.
    """
    if pred.y_proba is None:
        raise ModelError("cannot calibrate a prediction without y_proba")
    if pred.y_proba.ndim == 1:
        scores = pred.y_proba
    else:
        scores = _max_proba(pred.y_proba)
    fm = FeatureMatrix(
        values=scores.reshape(-1, 1),
        feature_names=("score",),
    )
    out = calibrator.predict(calibrator_state, fm)
    if out.y_proba is None:
        raise ModelError("calibrator did not produce y_proba")
    return out.y_proba


def _max_proba(y_proba: np.ndarray) -> np.ndarray:
    if y_proba.ndim == 1:
        return y_proba
    return y_proba.max(axis=-1)


__all__ = [
    "IsotonicCalibrator",
    "IsotonicConfig",
    "PlattCalibrator",
    "PlattConfig",
    "calibrated_proba",
]
