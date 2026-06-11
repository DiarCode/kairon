"""Tree ensembles: RandomForest, XGBoost, LightGBM.

RandomForest is in core deps (sklearn). XGBoost and LightGBM are optional
and imported lazily so the model layer can still construct and serialize
an XGBoost config in a Windows environment that lacks the wheel.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix


# ---------------------------------------------------------------------------
# RandomForest — always available
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 5
    max_features: str | float | int = "sqrt"
    class_weight: str | None = "balanced"
    n_jobs: int = -1
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {self.max_depth}")
        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}")


class RandomForestModel(Model[RandomForestConfig]):
    """sklearn RandomForestClassifier."""

    name = "random_forest"
    kind = "tree"

    def __init__(self, config: RandomForestConfig | None = None) -> None:
        super().__init__(config or RandomForestConfig())

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        clf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,  # type: ignore[arg-type]
            class_weight=self.config.class_weight,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        if sample_weight is not None:
            clf.fit(features.values, y, sample_weight=sample_weight)
        else:
            clf.fit(features.values, y)
        acc = float((clf.predict(features.values) == y).mean())
        return clf, {"train_acc": acc}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        clf = trained
        y_class = clf.predict(features.values).astype(np.int64)
        proba_all = clf.predict_proba(features.values)
        if proba_all.shape[1] == 2:
            y_proba = proba_all[:, 1]
        else:
            y_proba = proba_all
        return y_class, y_proba, None


# ---------------------------------------------------------------------------
# XGBoost — optional
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    objective: str = "multi:softprob"
    tree_method: str = "hist"
    n_jobs: int = -1
    random_state: int = 42
    early_stopping_rounds: int | None = 20
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample}")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError(f"colsample_bytree must be in (0, 1], got {self.colsample_bytree}")


def _has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None  # type: ignore[attr-defined]


class XGBoostModel(Model[XGBoostConfig]):
    name = "xgboost"
    kind = "tree"

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        super().__init__(config or XGBoostConfig())
        if not _has_xgboost():
            raise ModelError(
                "xgboost is not installed; install with `uv sync --extra ml`"
            )

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        xgb = importlib.import_module("xgboost")
        n_classes = int(np.unique(y).size)
        params: dict[str, Any] = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_lambda": self.config.reg_lambda,
            "reg_alpha": self.config.reg_alpha,
            "min_child_weight": self.config.min_child_weight,
            "tree_method": self.config.tree_method,
            "n_jobs": self.config.n_jobs,
            "random_state": self.config.random_state,
            "objective": self.config.objective,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        }
        if n_classes > 2:
            params["num_class"] = n_classes
        clf = xgb.XGBClassifier(**params)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        clf.fit(features.values, y, **fit_kwargs)
        acc = float((clf.predict(features.values) == y).mean())
        return clf, {"train_acc": acc}

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
# LightGBM — optional
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LightGBMConfig:
    n_estimators: int = 400
    num_leaves: int = 31
    learning_rate: float = 0.05
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_samples: int = 20
    objective: str = "multiclass"
    n_jobs: int = -1
    random_state: int = 42
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.num_leaves < 2:
            raise ValueError(f"num_leaves must be >= 2, got {self.num_leaves}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")


def _has_lightgbm() -> bool:
    return importlib.util.find_spec("lightgbm") is not None  # type: ignore[attr-defined]


class LightGBMModel(Model[LightGBMConfig]):
    name = "lightgbm"
    kind = "tree"

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        super().__init__(config or LightGBMConfig())
        if not _has_lightgbm():
            raise ModelError(
                "lightgbm is not installed; install with `uv sync --extra ml`"
            )

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        lgb = importlib.import_module("lightgbm")
        n_classes = int(np.unique(y).size)
        params: dict[str, Any] = {
            "n_estimators": self.config.n_estimators,
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "subsample_freq": self.config.subsample_freq,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_lambda": self.config.reg_lambda,
            "reg_alpha": self.config.reg_alpha,
            "min_child_samples": self.config.min_child_samples,
            "n_jobs": self.config.n_jobs,
            "random_state": self.config.random_state,
            "verbose": -1,
        }
        if n_classes > 2:
            params["objective"] = self.config.objective
            params["num_class"] = n_classes
        else:
            params["objective"] = "binary"
        clf = lgb.LGBMClassifier(**params)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        clf.fit(features.values, y, **fit_kwargs)
        acc = float((clf.predict(features.values) == y).mean())
        return clf, {"train_acc": acc}

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


# Type-only marker (avoid unused-import warning on the Callable import)
_ = Callable
