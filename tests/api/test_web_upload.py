"""Tests for the /api/uploads endpoint (US-007)."""
from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient


def test_post_uploads_returns_run_id(tmp_path: Path) -> None:
    # Force a fresh RunStore under tmp_path so we don't pollute data/runs.db
    from kairon.api.app import create_app
    from kairon.store.runs import RunStore

    app = create_app()
    app.state.run_store = RunStore(tmp_path / "runs.db")

    csv_bytes = b"ts,open,high,low,close,volume\n2026-01-01,1,2,0.5,1.5,10\n2026-01-02,2,3,1.5,2.5,20\n2026-01-03,2,3,1.5,2.5,20\n"
    with TestClient(app) as c:
        r = c.post(
            "/api/uploads",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "run_id" in body
        assert body["row_count"] == 3
        # The file should have been written under runs/<run_id>/input.csv
        server_path = Path(body["csv_path"])
        assert server_path.exists()
        assert server_path.read_bytes() == csv_bytes


def test_post_uploads_no_file_returns_422() -> None:
    """FastAPI returns 422 (Unprocessable Entity) when a required file is missing."""
    from kairon.api.app import create_app
    from kairon.store.runs import RunStore

    app = create_app()
    app.state.run_store = RunStore(Path("data/runs.db"))

    with TestClient(app) as c:
        r = c.post("/api/uploads", data={})
        assert r.status_code in (400, 422)  # FastAPI validation = 422; our manual = 400
