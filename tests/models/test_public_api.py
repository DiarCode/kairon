"""Smoke test: import every public model class and verify the public API."""
from __future__ import annotations

import importlib.util

import pytest


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None  # type: ignore[attr-defined]


def test_public_api_imports() -> None:
    from kairon.models import (
        available_models,
        build_model,
        register_model,
    )

    assert callable(build_model)
    assert callable(register_model)
    assert "logreg" in available_models()


def test_deep_ensemble_public_api() -> None:
    from kairon.models import DeepEnsemble, DeepEnsembleConfig

    assert DeepEnsemble is not None
    assert DeepEnsembleConfig is not None


def test_nbeats_public_api() -> None:
    from kairon.models import NBEATSConfig, NBEATSModel

    assert NBEATSModel is not None
    assert NBEATSConfig is not None


def test_lstm_public_api() -> None:
    from kairon.models import LSTMConfig, LSTMModel

    assert LSTMModel is not None
    assert LSTMConfig is not None


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")  # type: ignore[misc]
def test_torch_backends_listed() -> None:
    """When torch is installed, the registry still only includes the
    core backends; deep-TS models are accessed via the package, not
    the registry.
    """
    from kairon.models import available_models

    names = available_models()
    assert "logreg" in names
    assert "random_forest" in names
