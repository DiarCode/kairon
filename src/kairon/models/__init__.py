"""Models public API."""

from kairon.models.base import (
    Model,
    ModelError,
    NotFitError,
    Prediction,
    TrainedModel,
)
from kairon.models.calibration import (
    IsotonicCalibrator,
    IsotonicConfig,
    PlattCalibrator,
    PlattConfig,
    calibrated_proba,
)
from kairon.models.contracts import (
    FeatureMatrix,
    ensure_feature_matrix,
    feature_diff,
    is_classification,
    is_regression,
)
from kairon.models.deep_ensemble import DeepEnsemble, DeepEnsembleConfig
from kairon.models.ensemble import (
    EnsembleSpec,
    EnsembleTrained,
    MajorityVoteEnsemble,
    TopKConfidenceEnsemble,
)
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.lstm import LSTMConfig, LSTMModel
from kairon.models.nbeats import NBEATSConfig, NBEATSModel
from kairon.models.registry import (
    ModelKind,
    available_models,
    build_model,
    model_kind,
    register_model,
)
from kairon.models.tracking import MlflowTracker, TrackingConfig
from kairon.models.trainer import (
    FoldMetrics,
    NoOpTracker,
    Tracker,
    Trainer,
    TrainResult,
)
from kairon.models.tree import (
    LightGBMConfig,
    LightGBMModel,
    RandomForestConfig,
    RandomForestModel,
    XGBoostConfig,
    XGBoostModel,
)
from kairon.models.tree_multihead import TreeMultiHeadConfig, TreeMultiHeadModel
from kairon.models.stacked_multihead import StackedMultiHeadConfig, StackedMultiHeadModel

__all__ = [
    "DeepEnsemble",
    "DeepEnsembleConfig",
    "EnsembleSpec",
    "EnsembleTrained",
    "FeatureMatrix",
    "FoldMetrics",
    "IsotonicCalibrator",
    "IsotonicConfig",
    "LSTMConfig",
    "LSTMModel",
    "LightGBMConfig",
    "LightGBMModel",
    "LinearConfig",
    "LogisticRegressionModel",
    "MajorityVoteEnsemble",
    "MlflowTracker",
    "Model",
    "ModelError",
    "ModelKind",
    "NBEATSConfig",
    "NBEATSModel",
    "NoOpTracker",
    "NotFitError",
    "PlattCalibrator",
    "PlattConfig",
    "Prediction",
    "RandomForestConfig",
    "RandomForestModel",
    "StackedMultiHeadConfig",
    "StackedMultiHeadModel",
    "TopKConfidenceEnsemble",
    "Tracker",
    "TrackingConfig",
    "TrainResult",
    "TrainedModel",
    "Trainer",
    "TreeMultiHeadConfig",
    "TreeMultiHeadModel",
    "XGBoostConfig",
    "XGBoostModel",
    "available_models",
    "build_model",
    "calibrated_proba",
    "ensure_feature_matrix",
    "feature_diff",
    "is_classification",
    "is_regression",
    "model_kind",
    "register_model",
]
