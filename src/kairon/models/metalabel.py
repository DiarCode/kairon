"""Meta-learner model.

The meta-learner is the *secondary* model in a stacked / meta-labelled
pipeline (Lopez de Prado, *Advances in Financial Machine Learning*,
ch. 5). The primary model emits a directional probability
``p_primary`` per bar; this class takes that probability plus a
small DataFrame of *side* features (volatility z-score, spread,
regime, time-of-day, recent rolling win rate, ...) and returns a
``p_meta in [0, 1]`` -- the probability that the primary's signal is
*worth executing* given the surrounding market state.

Per the plan's W3.3 spec, the preferred backend is
``xgboost.XGBClassifier`` when the optional ``xgboost`` extra is
installed; the implementation falls back to
``sklearn.ensemble.GradientBoostingClassifier`` otherwise. The
fallback is seamless and the public API is identical regardless of
which backend is active.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from kairon.models.base import Model
from kairon.models.contracts import FeatureMatrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MLConfig:
    """Frozen config for the meta-learner.

    Defaults are conservative: 100 trees, max_depth 3, learning_rate
    0.05. The small trees + low learning rate match the W3.6 OOF
    protocol -- the meta-learner sees a small N (one observation per
    primary signal) and shallow trees generalise better than deep
    ones in that regime.
    """

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 42
    use_xgboost_if_available: bool = True
    subsample: float = 1.0  # sklearn GradientBoostingClassifier parameter
    extras: dict[str, Any] = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
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
# Optional-dependency probe
# ---------------------------------------------------------------------------
def _has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MetaLearnerModel(Model[MLConfig]):
    """Secondary model that predicts whether a primary signal is worth taking.

    The model inherits the standard :class:`kairon.models.base.Model`
    contract (``fit`` / ``predict`` / ``_fit_core`` / ``_predict_core``)
    so it integrates with the existing trainer, registry, and mlflow
    layer. The backend is selected at fit time:

    * if ``config.use_xgboost_if_available=True`` AND xgboost is
      installed: ``xgboost.XGBClassifier`` with the config's
      ``n_estimators``, ``max_depth``, ``learning_rate``,
      ``random_state``;
    * otherwise: ``sklearn.ensemble.GradientBoostingClassifier`` with
      the same hyperparameters (plus ``subsample``).

    The predict call returns a :class:`Prediction` with
    ``y_class in {0, 1}`` (the meta-label) and
    ``y_proba = p_meta`` (the class-1 probability = the probability
    that the primary's signal is worth taking). The ``y_proba`` is
    always 1-D for binary targets, matching the
    :class:`kairon.models.base.Prediction` contract.
    """

    name = "meta_learner"
    kind = "tree"

    def __init__(self, config: MLConfig | None = None) -> None:
        super().__init__(config or MLConfig())

    # -- backend selection ------------------------------------------------
    def _build_classifier(self) -> Any:
        """Construct the underlying sklearn / xgboost classifier.

        The selection rule:

        * xgboost is preferred when ``config.use_xgboost_if_available``
          is True AND ``importlib.util.find_spec('xgboost')`` returns
          a ModuleSpec. The check is runtime, not import-time, so the
          sklearn path is taken cleanly when the xgboost wheel is
          missing.
        * otherwise: sklearn ``GradientBoostingClassifier`` with the
          config's hyperparameters.
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
        return GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )

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
        # Binary case: return the class-1 probability as a 1-D array
        # (this is the meta-probability p_meta). Non-binary is not a
        # supported target for the meta-learner; the trainers and
        # callers in W3.4/3.5/6.2 only ever feed a binary y_meta.
        if proba_all.ndim == 2 and proba_all.shape[1] == 2:
            y_proba = proba_all[:, 1]
        else:
            y_proba = proba_all
        return y_class, y_proba, None


__all__ = ["MLConfig", "MetaLearnerModel"]
