"""Stacked multi-head ensemble combining LR and Tree direction heads.

The StackedMultiHeadModel combines a LogisticRegression multi-head and a
Tree multi-head model via confidence-gated averaging. The direction head
uses TopKConfidenceEnsemble (select the K most confident predictions from
the constituent models). The magnitude and vol heads are simple averages
of the Ridge predictions from both models.

Architecture:
    Direction: TopKConfidenceEnsemble(LR_head, Tree_head)
               - Select predictions where max(proba) is above threshold
               - When both models agree, use their average probability
               - When they disagree, use the more confident model
    Magnitude: Average of LR and Tree magnitude predictions
    Vol: Average of LR and Tree vol predictions

This ensemble is designed to be more robust than either constituent
alone: the LR head provides well-calibrated probabilities on linear
features, while the Tree head captures nonlinear interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix
from kairon.models.metalabel import MLConfig
from kairon.models.multihead import MultiHeadConfig, MultiHeadModel
from kairon.models.tree_multihead import TreeMultiHeadConfig, TreeMultiHeadModel


@dataclass(frozen=True, slots=True)
class StackedMultiHeadConfig(MLConfig):
    """Configuration for :class:`StackedMultiHeadModel`.

    Delegates to :class:`MultiHeadConfig` for the LR head and
    :class:`TreeMultiHeadConfig` for the Tree head, with an additional
    confidence threshold for direction gating.
    """

    # Confidence threshold for direction gating (0 = no gating, all
    # predictions included; higher = more selective)
    confidence_threshold: float = 0.6
    # How many models to include in TopK (1 or 2)
    top_k: int = 2
    # Weight for LR vs Tree in magnitude/vol averaging (0.5 = equal)
    lr_weight: float = 0.5
    n_direction_classes: int = 3
    vol_alpha: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.top_k < 1 or self.top_k > 2:
            raise ValueError(f"top_k must be 1 or 2, got {self.top_k}")
        if not (0.0 <= self.lr_weight <= 1.0):
            raise ValueError(f"lr_weight must be in [0, 1], got {self.lr_weight}")


class StackedMultiHeadModel(Model[StackedMultiHeadConfig]):
    """Stacked multi-head ensemble combining LR and Tree models.

    Direction head uses confidence-gated ensemble:
    - When both models have max probability above threshold, average their
      probabilities and take the argmax
    - When only one model exceeds threshold, use that model's prediction
    - When neither model exceeds threshold, use the average probability

    Magnitude and vol heads are weighted averages of the LR and Tree
    predictions (default 50/50).
    """

    name = "stacked_multihead"
    kind = "ensemble"

    def __init__(self, config: StackedMultiHeadConfig | None = None) -> None:
        super().__init__(config or StackedMultiHeadConfig())
        self._lr_model = MultiHeadModel(MultiHeadConfig(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            n_direction_classes=self.config.n_direction_classes,
            vol_alpha=self.config.vol_alpha,
        ))
        self._tree_model = TreeMultiHeadModel(TreeMultiHeadConfig(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
            n_direction_classes=self.config.n_direction_classes,
            vol_alpha=self.config.vol_alpha,
        ))

    def fit_multihead(
        self,
        features: FeatureMatrix,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        y_vol: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Any:
        """Fit both LR and Tree multi-head models."""
        lr_state = self._lr_model.fit_multihead(
            features, y_direction, y_magnitude, y_vol,
            sample_weight=sample_weight,
        )
        tree_state = self._tree_model.fit_multihead(
            features, y_direction, y_magnitude, y_vol,
            sample_weight=sample_weight,
        )
        return {
            "lr_state": lr_state,
            "tree_state": tree_state,
            "direction_classes": np.array(sorted(np.unique(y_direction))),
        }

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        """Single-head fallback: fits both models on direction only."""
        lr_state = self._lr_model._fit_core(features, y, sample_weight=sample_weight, loss_fn=loss_fn)
        tree_state = self._tree_model._fit_core(features, y, sample_weight=sample_weight, loss_fn=loss_fn)
        state = {
            "lr_state": lr_state[0],  # _fit_core returns (state, metrics)
            "tree_state": tree_state[0],
            "direction_classes": np.array(sorted(np.unique(y))),
        }
        lr_acc = lr_state[1].get("train_acc", 0.0)
        tree_acc = tree_state[1].get("train_acc", 0.0)
        return state, {"train_acc": max(lr_acc, tree_acc)}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Ensemble prediction: confidence-gated direction + averaged magnitude/vol."""
        lr_state = trained["lr_state"]
        tree_state = trained["tree_state"]
        threshold = self.config.confidence_threshold

        # Get predictions from both models
        lr_result = self._lr_model.predict_multihead(lr_state, features)
        tree_result = self._tree_model.predict_multihead(tree_state, features)

        lr_proba = lr_result["y_proba"]  # (N, 3) or (N, 2)
        tree_proba = tree_result["y_proba"]  # (N, 3) or (N, 2)

        # Confidence-gated ensemble for direction
        n_classes = lr_proba.shape[1] if lr_proba.ndim > 1 else 2
        ensemble_proba = np.zeros_like(lr_proba)

        if n_classes == 2 and lr_proba.ndim == 2 and lr_proba.shape[1] == 2:
            # Binary case: use the positive class probability
            lr_max = np.maximum(lr_proba[:, 1], 1 - lr_proba[:, 1])
            tree_max = np.maximum(tree_proba[:, 1], 1 - tree_proba[:, 1])
        elif lr_proba.ndim == 2:
            lr_max = np.max(lr_proba, axis=1)
            tree_max = np.max(tree_proba, axis=1)
        else:
            lr_max = np.abs(lr_proba - 0.5) + 0.5
            tree_max = np.abs(tree_proba - 0.5) + 0.5

        # Weight by confidence
        for i in range(len(ensemble_proba)):
            lr_conf = lr_max[i] >= threshold
            tree_conf = tree_max[i] >= threshold

            if lr_conf and tree_conf:
                # Both confident: average probabilities
                w = self.config.lr_weight
                if lr_proba.ndim == 2:
                    ensemble_proba[i] = w * lr_proba[i] + (1 - w) * tree_proba[i]
                else:
                    ensemble_proba[i] = w * lr_proba[i] + (1 - w) * tree_proba[i]
            elif lr_conf:
                ensemble_proba[i] = lr_proba[i] if lr_proba.ndim == 2 else lr_proba[i]
            elif tree_conf:
                ensemble_proba[i] = tree_proba[i] if tree_proba.ndim == 2 else tree_proba[i]
            else:
                # Neither confident: average anyway
                w = self.config.lr_weight
                if lr_proba.ndim == 2:
                    ensemble_proba[i] = w * lr_proba[i] + (1 - w) * tree_proba[i]
                else:
                    ensemble_proba[i] = w * lr_proba[i] + (1 - w) * tree_proba[i]

        # Direction class
        if ensemble_proba.ndim == 2 and ensemble_proba.shape[1] > 1:
            y_class = np.argmax(ensemble_proba, axis=1).astype(np.int64)
        else:
            y_class = (ensemble_proba > 0.5).astype(np.int64)

        # Magnitude and vol: weighted average
        y_magnitude = None
        if lr_result["y_magnitude"] is not None and tree_result["y_magnitude"] is not None:
            w = self.config.lr_weight
            y_magnitude = w * lr_result["y_magnitude"] + (1 - w) * tree_result["y_magnitude"]

        return y_class, ensemble_proba, y_magnitude

    def predict_multihead(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> dict[str, np.ndarray]:
        """Ensemble multi-head prediction."""
        y_class, y_proba, y_magnitude = self._predict_core(trained, features)

        lr_state = trained["lr_state"]
        tree_state = trained["tree_state"]
        lr_result = self._lr_model.predict_multihead(lr_state, features)
        tree_result = self._tree_model.predict_multihead(tree_state, features)

        # Vol: weighted average
        y_vol = None
        if lr_result["y_vol"] is not None and tree_result["y_vol"] is not None:
            w = self.config.lr_weight
            y_vol = w * lr_result["y_vol"] + (1 - w) * tree_result["y_vol"]

        return {
            "y_class": y_class,
            "y_proba": y_proba if y_proba is not None else np.empty(0, dtype=np.float64),
            "y_magnitude": y_magnitude if y_magnitude is not None else np.zeros(features.n_rows, dtype=np.float64),
            "y_vol": y_vol if y_vol is not None else np.zeros(features.n_rows, dtype=np.float64),
        }


__all__ = ["StackedMultiHeadConfig", "StackedMultiHeadModel"]