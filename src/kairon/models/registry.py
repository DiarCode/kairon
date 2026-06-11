"""Model registry: name → factory.

Backends are looked up by string ("logreg", "random_forest", ...) so the
trainer, evaluator, and policy layer can stay backend-agnostic. Optional
backends (XGBoost, LightGBM, LSTM) raise :class:`ModelError` at
construction time if their dependencies are not installed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.tree import (
    LightGBMConfig,
    LightGBMModel,
    RandomForestConfig,
    RandomForestModel,
    XGBoostConfig,
    XGBoostModel,
)


class ModelKind(str, Enum):
    """High-level grouping for a backend — used by the ensemble router."""

    LINEAR = "linear"
    TREE = "tree"
    DEEP = "deep"
    STAT = "stat"
    ENSEMBLE = "ensemble"
    CALIBRATION = "calibration"


@dataclass(frozen=True, slots=True)
class _Registration:
    name: str
    factory: Callable[[dict[str, Any]], Model[Any]]
    kind: ModelKind
    requires: tuple[str, ...] = ()  # optional pip extras


_REGISTRY: dict[str, _Registration] = {
    "logreg": _Registration(
        "logreg",
        lambda cfg: LogisticRegressionModel(LinearConfig(**cfg) if cfg else LinearConfig()),
        ModelKind.LINEAR,
    ),
    "random_forest": _Registration(
        "random_forest",
        lambda cfg: RandomForestModel(
            RandomForestConfig(**cfg) if cfg else RandomForestConfig()
        ),
        ModelKind.TREE,
    ),
    "xgboost": _Registration(
        "xgboost",
        lambda cfg: XGBoostModel(XGBoostConfig(**cfg) if cfg else XGBoostConfig()),
        ModelKind.TREE,
        requires=("xgboost",),
    ),
    "lightgbm": _Registration(
        "lightgbm",
        lambda cfg: LightGBMModel(LightGBMConfig(**cfg) if cfg else LightGBMConfig()),
        ModelKind.TREE,
        requires=("lightgbm",),
    ),
}


def register_model(
    name: str,
    factory: Callable[[dict[str, Any]], Model[Any]],
    *,
    kind: ModelKind,
    requires: tuple[str, ...] = (),
) -> None:
    """Register a new backend (e.g. for a custom PyTorch model)."""
    if not name:
        raise ValueError("model name must be a non-empty string")
    if name in _REGISTRY:
        raise ValueError(f"model {name!r} already registered")
    _REGISTRY[name] = _Registration(
        name=name, factory=factory, kind=kind, requires=requires
    )


def build_model(name: str, config: dict[str, Any] | None = None) -> Model[Any]:
    if name not in _REGISTRY:
        raise ModelError(
            f"unknown model {name!r}; available: {sorted(_REGISTRY)}"
        )
    reg = _REGISTRY[name]
    for pkg in reg.requires:
        if importlib.util.find_spec(pkg) is None:  # type: ignore[attr-defined]
            raise ModelError(
                f"model {name!r} requires optional package {pkg!r}; "
                f"install with `uv sync --extra ml`"
            )
    return reg.factory(config or {})


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def model_kind(name: str) -> ModelKind:
    if name not in _REGISTRY:
        raise ModelError(f"unknown model {name!r}")
    return _REGISTRY[name].kind


__all__ = [
    "ModelKind",
    "available_models",
    "build_model",
    "model_kind",
    "register_model",
]
