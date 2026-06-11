"""Tests for the model registry."""

from __future__ import annotations

import pytest

from kairon.models.base import Model, ModelError
from kairon.models.linear import LogisticRegressionModel
from kairon.models.registry import (
    ModelKind,
    available_models,
    build_model,
    model_kind,
    register_model,
)


def test_build_model_logreg() -> None:
    m = build_model("logreg")
    assert isinstance(m, LogisticRegressionModel)
    assert m.name == "logreg"


def test_build_model_with_config() -> None:
    m = build_model("logreg", {"C": 0.5})
    assert m.config.C == 0.5


def test_build_unknown_raises() -> None:
    with pytest.raises(ModelError, match="unknown model"):
        build_model("not_a_model")


def test_model_kind_known() -> None:
    assert model_kind("logreg") == ModelKind.LINEAR
    assert model_kind("random_forest") == ModelKind.TREE


def test_model_kind_unknown_raises() -> None:
    with pytest.raises(ModelError, match="unknown model"):
        model_kind("nope")


def test_available_models_includes_core() -> None:
    names = set(available_models())
    assert "logreg" in names
    assert "random_forest" in names


def test_register_model() -> None:
    class _Custom(Model["object"]):
        name = "custom_test"
        kind = ModelKind.LINEAR

        def _fit_core(self, features, y, *, sample_weight, loss_fn):
            return None, {}

        def _predict_core(self, trained, features):
            import numpy as np
            return np.zeros(features.n_rows, dtype=np.int64), None, None

    register_model("custom_test", lambda cfg: _Custom(object()), kind=ModelKind.LINEAR)  # type: ignore[arg-type]
    try:
        m = build_model("custom_test")
        assert isinstance(m, _Custom)
        assert model_kind("custom_test") == ModelKind.LINEAR
        assert "custom_test" in available_models()
    finally:
        # Cleanup: there's no public unregister; re-registering raises
        from kairon.models import registry as _r
        _r._REGISTRY.pop("custom_test", None)


def test_register_model_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        register_model("", lambda c: None, kind=ModelKind.LINEAR)  # type: ignore[arg-type]


def test_register_model_rejects_duplicate() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_model("logreg", lambda c: None, kind=ModelKind.LINEAR)  # type: ignore[arg-type]


def test_optional_backend_raises_when_missing(monkeypatch) -> None:
    """Simulate xgboost being absent by stubbing find_spec."""
    import importlib

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "xgboost" else importlib.import_module(name).__spec__)  # type: ignore[attr-defined]
    with pytest.raises(ModelError, match="requires optional package"):
        build_model("xgboost", {})
