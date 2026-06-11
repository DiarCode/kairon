"""Linear / logistic regression baseline.

Always available — sklearn is in core deps. Used as a calibration anchor
and as the "what's the linear floor of this dataset?" reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix


@dataclass(frozen=True, slots=True)
class LinearConfig:
    """Hyperparameters for the linear baseline.

    ``C`` is sklearn's inverse regularization strength; ``1.0`` is a
    neutral starting point. ``class_weight="balanced"`` is the right
    default for direction labels, which are roughly balanced but
    skewed toward ``FLAT`` in low-volatility regimes.
    """

    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    penalty: str = "l2"
    solver: str = "lbfgs"
    standardize: bool = True
    random_state: int = 42
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.penalty not in {"l1", "l2", "elasticnet", "none"}:
            raise ValueError(f"unsupported penalty: {self.penalty!r}")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError(f"class_weight must be None or 'balanced', got {self.class_weight!r}")


class LogisticRegressionModel(Model[LinearConfig]):
    """Multinomial logistic regression."""

    name = "logreg"
    kind = "linear"

    def __init__(self, config: LinearConfig | None = None) -> None:
        super().__init__(config or LinearConfig())

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        x = features.values
        if self.config.standardize:
            scaler: StandardScaler | None = StandardScaler().fit(x)
            x = scaler.transform(x)
        else:
            scaler = None

        clf_kwargs: dict[str, Any] = {
            "C": self.config.C,
            "max_iter": self.config.max_iter,
            "class_weight": self.config.class_weight,
            "random_state": self.config.random_state,
        }
        # sklearn 1.8+ deprecates passing penalty='none' / 'l1' / 'l2' as
        # a kwarg. We translate our config to the new l1_ratio semantics.
        penalty = self.config.penalty
        if penalty == "l1":
            clf_kwargs["l1_ratio"] = 1.0
        elif penalty == "l2":
            clf_kwargs["l1_ratio"] = 0.0
        elif penalty == "elasticnet":
            clf_kwargs["l1_ratio"] = 0.5
        elif penalty == "none":
            clf_kwargs["C"] = float("inf")
            clf_kwargs["l1_ratio"] = None
        if penalty == "elasticnet":
            clf_kwargs["solver"] = "saga"
        elif self.config.solver:
            clf_kwargs["solver"] = self.config.solver

        try:
            clf = LogisticRegression(**clf_kwargs)
        except TypeError as e:  # pragma: no cover - defensive against sklearn drift
            raise ModelError(f"LogisticRegression init failed: {e}") from e

        if sample_weight is not None:
            clf.fit(x, y, sample_weight=sample_weight)
        else:
            clf.fit(x, y)

        # in-sample score for fast triage; real eval is in the trainer
        y_pred_in = np.asarray(clf.predict(x))
        acc = float((y_pred_in == y).mean())
        state = {"scaler": scaler, "clf": clf}
        return state, {"train_acc": acc}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        state: dict[str, Any] = trained if isinstance(trained, dict) else trained.state
        scaler = state.get("scaler")
        clf = state["clf"]
        x = features.values
        if scaler is not None:
            x = scaler.transform(x)
        y_class = clf.predict(x).astype(np.int64)
        proba_all = clf.predict_proba(x)
        # For binary classification, return P(class==+1) (assume classes sorted asc).
        if proba_all.shape[1] == 2:
            classes = clf.classes_
            pos_idx = int(np.where(classes == classes.max())[0][0])
            y_proba = proba_all[:, pos_idx]
        else:
            y_proba = proba_all
        return y_class, y_proba, None


# Type-narrowing helper to keep mypy happy with the cast above.
_ = cast
