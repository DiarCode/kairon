"""Tests for the /live dashboard API: status, screen render, start guards."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def _has_fastapi() -> bool:
    return importlib.util.find_spec("fastapi") is not None  # type: ignore[attr-defined]


SNAPSHOT_KEYS = {
    "status", "is_running", "session_id", "mode", "equity", "initial_equity",
    "session_pnl", "realized_pnl", "unrealized_pnl", "n_positions", "halted",
    "halt_reason", "last_heartbeat", "last_signal_ts", "symbols", "positions",
    "decisions", "closed_trades", "orders", "events", "win_rate", "equity_series",
}


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
class TestLiveDashboardApi:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from kairon.api.app import create_app

        return TestClient(create_app())

    def test_status_returns_snapshot_keys(self, client) -> None:
        resp = client.get("/api/live/status")
        assert resp.status_code == 200
        assert set(resp.json().keys()) == SNAPSHOT_KEYS
        assert resp.json()["status"] == "stopped"

    def test_live_screen_renders(self, client) -> None:
        resp = client.get("/live")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "AI Decisions" in resp.text
        assert "/api/live/stream" in resp.text

    def test_start_mainnet_without_confirm_is_400(self, client) -> None:
        resp = client.post("/api/live/start", json={"mode": "live"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "confirmation_required"

    def test_start_invalid_timeframe_is_422(self, client) -> None:
        resp = client.post("/api/live/start", json={"timeframe": "1x"})
        assert resp.status_code == 422

    def test_start_paper_with_mocked_host_is_200(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hermetic settings + a mocked start_session so no network is touched.
        monkeypatch.setattr(
            "kairon.config.KaironSettings",
            lambda: SimpleNamespace(bybit_tld="com", bybit_api_key="", bybit_api_secret=""),
        )
        app = client.app

        async def fake_start(config, settings, *, cooldown_seconds: float = 300.0):
            return {
                "session_id": "deadbeef0000",
                "mode": "Paper",
                "initial_equity": 10_000.0,
                "symbols": list(config.symbols),
            }

        monkeypatch.setattr(app.state.session_host, "start_session", fake_start)
        resp = client.post(
            "/api/live/start",
            json={"mode": "paper", "symbols": ["BTC-USDT-PERP"]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == "deadbeef0000"
        assert body["mode"] == "Paper"
