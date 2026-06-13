"""Smoke tests for the 5 web screens (US-007)."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_upload_returns_200() -> None:
    from kairon.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.get("/upload")
        assert r.status_code == 200
        body = r.text
        assert "Upload" in body
        assert "kairon-glass-card" in body
        assert "kairon-drop-zone" in body


def test_get_configure_returns_200() -> None:
    from kairon.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.get("/configure")
        assert r.status_code == 200
        body = r.text
        # Configure now redirects to /upload via client-side JS
        assert "Redirecting" in body or "/upload" in body


def test_get_analyze_returns_200() -> None:
    from kairon.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.get("/analyze")
        assert r.status_code == 200
        body = r.text
        assert "Analyzing" in body
        assert "kairon-progress-ring" in body


def test_get_track_returns_200() -> None:
    from kairon.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.get("/track")
        assert r.status_code == 200
        body = r.text
        assert "Track" in body
        assert "kairon-track-table" in body


def test_get_result_unknown_returns_404() -> None:
    from kairon.api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.get("/result/does-not-exist")
        assert r.status_code == 404
