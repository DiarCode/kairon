"""Tests for the FastAPI app (skipped if fastapi is not installed)."""

from __future__ import annotations

import importlib

import pytest


def _has_fastapi() -> bool:
    return importlib.util.find_spec("fastapi") is not None  # type: ignore[attr-defined]


def test_create_app_requires_fastapi() -> None:
    """If fastapi is absent, create_app() should raise ImportError."""
    if _has_fastapi():
        pytest.skip("fastapi is installed; can't test missing-dep path")
    from kairon.api.app import create_app

    with pytest.raises(ImportError, match="fastapi is not installed"):
        create_app()


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
def test_create_app_returns_fastapi_app() -> None:
    from fastapi import FastAPI

    from kairon.api.app import create_app

    app = create_app()
    assert isinstance(app, FastAPI)


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
def test_healthz_endpoint() -> None:
    from fastapi.testclient import TestClient

    from kairon.api.app import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["uptime_seconds"] >= 0.0


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
def test_list_models_endpoint() -> None:
    from fastapi.testclient import TestClient

    from kairon.api.app import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert "logreg" in body["models"]


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
def test_train_endpoint_validates_request() -> None:
    from fastapi.testclient import TestClient

    from kairon.api.app import create_app

    app = create_app()
    client = TestClient(app)
    # Missing required field
    resp = client.post("/v1/models/train", json={"symbol": "BTC/USDT"})
    assert resp.status_code == 422


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
def test_train_endpoint_stub() -> None:
    from fastapi.testclient import TestClient

    from kairon.api.app import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/models/train",
        json={"symbol": "BTC/USDT", "model_backend": "logreg"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "logreg"


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
def test_predict_endpoint_stub() -> None:
    from fastapi.testclient import TestClient

    from kairon.api.app import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/models/predict",
        json={"run_name": "abc", "symbol": "BTC/USDT", "n_rows": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_name"] == "abc"
    assert len(body["y_class"]) == 2


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
def test_backtest_endpoint_stub() -> None:
    from fastapi.testclient import TestClient

    from kairon.api.app import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/backtest",
        json={"run_name": "abc", "symbol": "BTC/USDT"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTC/USDT"
    assert body["final_equity"] == 10_000.0
