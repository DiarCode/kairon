"""Tree-based multi-head model: XGBoost direction + Ridge magnitude/vol.

The TreeMultiHeadModel upgrades the direction head from LogisticRegression
to a gradient-boosted tree (XGBoost preferred, LightGBM fallback, then
RandomForest). The magnitude and vol heads remain Ridge regressors — these
are linear heads that work well for continuous targets and don't benefit
from the non-linearity that the direction head needs.

This model addresses the core limitation identified in the accuracy
analysis: LogisticRegression is a linear classifier that cannot capture
the nonlinear feature interactions that tree models exploit. With 80+
features (many of which are nonlinear — EW position, regime probabilities,
FVG proximity), a tree model can achieve significantly higher accuracy.

Architecture:
    Direction head: XGBClassifier (or LGBMClassifier, or RandomForestClassifier)
        - n_estimators=300, max_depth=4, learning_rate=0.05
        - subsample=0.8, colsample_bytree=0.8 (regularization)
        - reg_alpha=0.1, reg_lambda=1.0 (L1+L2 regularization)
    Magnitude head: Ridge(alpha=1.0) — same as MultiHeadModel
    Vol head: Ridge(alpha=1.0) on log1p(vol) — same as MultiHeadModel

Fallback chain: XGBoost -> LightGBM -> RandomForest (always available)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix
from kairon.models.metalabel import MLConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TreeMultiHeadConfig(MLConfig):
    """Configuration for :class:`TreeMultiHeadModel`.

    Extends :class:`MLConfig` with tree-specific hyperparameters and
    the multi-head fields from W6.4 (direction classes, vol alpha).

    Tree-specific fields:
        direction_backend: "xgboost" (default), "lightgbm", or "random_forest".
            If the preferred backend is not installed, falls back to
            the next in the chain: XGBoost -> LightGBM -> RandomForest.
    """

    n_direction_classes: int = 3
    vol_alpha: float = 0.5
    magnitude_alpha: float = 0.5
    direction_backend: str = "xgboost"
    extras: dict[str, Any] = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        if self.n_direction_classes < 2:
            raise ValueError(
                f"n_direction_classes must be >= 2, got {self.n_direction_classes}"
            )
        if not (0.0 < self.vol_alpha < 1.0):
            raise ValueError(f"vol_alpha must be in (0, 1), got {self.vol_alpha!r}")
        if not (0.0 < self.magnitude_alpha <= 1.0):
            raise ValueError(
                f"magnitude_alpha must be in (0, 1], got {self.magnitude_alpha!r}"
            )
        if self.direction_backend not in ("xgboost", "lightgbm", "random_forest"):
            raise ValueError(
                f"direction_backend must be 'xgboost', 'lightgbm', or 'random_forest', "
                f"got {self.direction_backend!r}"
            )
        # Parent field validation
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


def _has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None  # type: ignore[attr-defined]


def _has_lightgbm() -> bool:
    return importlib.util.find_spec("lightgbm") is not None  # type: ignore[attr-defined]


def _resolve_backend(preferred: str) -> str:
    """Resolve the direction backend, falling back if not installed."""
    if preferred == "xgboost" and _has_xgboost():
        return "xgboost"
    if preferred == "xgboost" and not _has_xgboost():
        logger.info("XGBoost not installed; falling back to LightGBM")
        if _has_lightgbm():
            return "lightgbm"
        logger.info("LightGBM not installed; falling back to RandomForest")
        return "random_forest"
    if preferred == "lightgbm" and _has_lightgbm():
        return "lightgbm"
    if preferred == "lightgbm" and not _has_lightgbm():
        logger.info("LightGBM not installed; falling back to RandomForest")
        return "random_forest"
    return "random_forest"


def _build_direction_classifier(config: TreeMultiHeadConfig, backend: str):
    """Build a direction classifier for the given backend."""
    if backend == "xgboost":
        xgb = importlib.import_module("xgboost")
        return xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=min(config.subsample, 0.8),
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.random_state,
            objective="multi:softprob" if config.n_direction_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if config.n_direction_classes > 2 else "logloss",
        )
    elif backend == "lightgbm":
        lgb = importlib.import_module("lightgbm")
        n_classes = config.n_direction_classes
        params = dict(
            n_estimators=config.n_estimators,
            num_leaves=min(2**config.max_depth, 31),
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            subsample_freq=1,
            colsample_bytree=min(config.subsample, 0.8),
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_samples=20,
            n_jobs=-1,
            random_state=config.random_state,
            verbose=-1,
        )
        if n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
        else:
            params["objective"] = "binary"
        return lgb.LGBMClassifier(**params)
    else:
        # RandomForest — always available
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_state,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TreeMultiHeadModel(Model[TreeMultiHeadConfig]):
    """Tree-based multi-head model: XGBoost/LGBM direction + Ridge magnitude/vol.

    The direction head uses a gradient-boosted tree classifier (XGBoost by
    default, with LightGBM and RandomForest fallbacks). The magnitude and
    vol heads remain Ridge regressors — linear models are sufficient for
    continuous regression targets.

    This model is designed to replace the LogisticRegression direction head
    in :class:`MultiHeadModel` when nonlinear feature interactions are
    important (e.g., with 80+ features including EW position, regime
    probabilities, FVG proximity, etc.).
    """

    name = "tree_multihead"
    kind = "tree"

    def __init__(self, config: TreeMultiHeadConfig | None = None) -> None:
        super().__init__(config or TreeMultiHeadConfig())

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
        """Fit the tree-based multi-head model with three target vectors.

        Returns a state dict with keys: "direction_head", "magnitude_head",
        "vol_head", "direction_classes", "backend".
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

        # Resolve backend (fall back if preferred not installed)
        backend = _resolve_backend(self.config.direction_backend)
        if backend != self.config.direction_backend:
            logger.info(
                "TreeMultiHead: resolved backend %r (preferred %r)",
                backend, self.config.direction_backend,
            )

        # Direction head: tree-based classifier
        direction_head = _build_direction_classifier(self.config, backend)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        direction_head.fit(features.values, y_direction, **fit_kwargs)

        # Magnitude head: Ridge regressor (same as MultiHeadModel)
        magnitude_head = Ridge(alpha=1.0, random_state=self.config.random_state)
        magnitude_head.fit(features.values, y_magnitude, **fit_kwargs)

        # Vol head: Ridge on log1p(vol) (same as MultiHeadModel)
        vol_log = np.log1p(np.clip(y_vol, a_min=0.0, a_max=None))
        vol_head = Ridge(alpha=1.0, random_state=self.config.random_state)
        vol_head.fit(features.values, vol_log, **fit_kwargs)

        return {
            "direction_head": direction_head,
            "magnitude_head": magnitude_head,
            "vol_head": vol_head,
            "direction_classes": np.array(sorted(np.unique(y_direction))),
            "backend": backend,
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
        """Single-head fallback: fits direction head only."""
        backend = _resolve_backend(self.config.direction_backend)
        direction_head = _build_direction_classifier(self.config, backend)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        direction_head.fit(features.values, y, **fit_kwargs)
        state: dict[str, Any] = {
            "direction_head": direction_head,
            "magnitude_head": None,
            "vol_head": None,
            "direction_classes": np.array(sorted(np.unique(y))),
            "backend": backend,
        }
        acc = float(np.mean(direction_head.predict(features.values) == y))
        return state, {"train_acc": acc}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Predict direction, probability, and magnitude."""
        direction_head: Any = trained["direction_head"]
        magnitude_head: Any = trained.get("magnitude_head")
        vol_head: Any = trained.get("vol_head")

        # Direction prediction
        y_class_arr = direction_head.predict(features.values).astype(np.int64, copy=False)
        proba_all = direction_head.predict_proba(features.values)
        if proba_all.ndim == 1:
            y_proba = proba_all.reshape(-1, 1)
        else:
            y_proba = proba_all

        # Magnitude prediction
        y_magnitude: np.ndarray | None = None
        if magnitude_head is not None:
            y_magnitude = magnitude_head.predict(features.values).astype(np.float64, copy=False)

        # Vol prediction
        y_vol: np.ndarray | None = None
        if vol_head is not None:
            raw_vol = np.expm1(vol_head.predict(features.values)).astype(np.float64, copy=False)
            y_vol = np.maximum(raw_vol, 0.0)

        y_score = y_magnitude  # v1 compatibility
        return y_class_arr, y_proba, y_score

    # -- multi-head predict helper ----------------------------------------
    def predict_multihead(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> dict[str, np.ndarray]:
        """Run a multi-head prediction and return all three heads' outputs."""
        y_class, y_proba, _ = self._predict_core(trained, features)

        magnitude_head: Any = trained.get("magnitude_head")
        vol_head: Any = trained.get("vol_head")

        y_magnitude_arr: np.ndarray | None = None
        if magnitude_head is not None:
            y_magnitude_arr = magnitude_head.predict(features.values).astype(np.float64, copy=False)

        y_vol_arr: np.ndarray | None = None
        if vol_head is not None:
            raw_vol = np.expm1(vol_head.predict(features.values)).astype(np.float64, copy=False)
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


__all__ = ["TreeMultiHeadConfig", "TreeMultiHeadModel"]