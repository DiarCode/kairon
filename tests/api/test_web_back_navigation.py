"""Tests for the Back-Context header on /result/<run_id> (US-007)."""
from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pandas as pd

from fastapi.testclient import TestClient

from kairon.analysis.contracts import ProvenanceBlock
from kairon.analysis.engine import AnalysisResult, ModelPrediction
from kairon.api.app import create_app
from kairon.store.runs import RunStore


def _stub_cs() -> Any:
    from kairon.analysis.engine import CurrentState

    return CurrentState(
        timestamp=datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC),
        close=100.0,
        ew_wave_position=0.0, ew_wave_direction=0.0, ew_is_impulse=False,
        ew_completion_prob=0.5, ew_fib_confluence=0.5, regime="ranging",
        regime_prob_trending=0.25, regime_prob_ranging=0.5,
        regime_prob_volatile=0.2, regime_prob_stressed=0.05,
        hurst_exp=0.5, garch_vol=0.02, atr_14=1.5, rsi_14=50.0,
        fib_dist_236=0.01, fib_dist_382=0.02, fib_dist_500=0.03,
        fib_dist_618=0.04, fib_dist_786=0.05,
        fvg_bullish=False, fvg_bearish=False, fvg_fill_pct=0.0,
        fvg_nearest_distance=0.0, ob_in_bullish_zone=False,
        ob_in_bearish_zone=False, ob_bullish_near=False, ob_bearish_near=False,
        bos_direction=0, choch_direction=0, bb_upper=101.0, bb_mid=100.0, bb_lower=99.0,
        ema_50=100.0, ema_200=100.0,
    )


def _seed_run(app: Any, run_id: str) -> None:
    """Insert a RunResult directly via the store (no need to run the engine here)."""
    from kairon.analysis.contracts import ModelTile

    run = AnalysisResult.__class__  # type: ignore[attr-defined]
    # Use the new RunResult contract directly
    from kairon.analysis.contracts import RunResult as RR

    rr = RR(
        run_id=run_id,
        asset="BTC",
        horizon="day",
        created_at_utc=datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC),
        models=(
            ModelTile(name="trend", chart_png_path=f"runs/{run_id}/charts/trend.png",
                      predicted_pct=0.01, stop_loss=99.0, ideal_entry=100.0,
                      ideal_exit=101.0, confidence=0.6),
        ),
        provenance=ProvenanceBlock(config_hash="c", data_hash="d", model_version="kairon-0.1.0", seed=42),
        base_price=100.0,
    )
    csv_path = Path("data") / run_id / "input.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("ts,open,high,low,close,volume\n2026-01-01,1,2,0.5,1.5,10\n")
    app.state.run_store.create(rr, csv_path)


def test_back_context_default_is_analyze(tmp_path: Path) -> None:
    app = create_app()
    app.state.run_store = RunStore(tmp_path / "runs.db")
    with TestClient(app) as c:
        _seed_run(app, "r1")
        r = c.get("/result/r1")
        assert r.status_code == 200
        assert r.headers.get("Back-Context") == "analyze"


def test_back_context_from_query_is_track(tmp_path: Path) -> None:
    app = create_app()
    app.state.run_store = RunStore(tmp_path / "runs.db")
    with TestClient(app) as c:
        _seed_run(app, "r2")
        r = c.get("/result/r2?from=track")
        assert r.status_code == 200
        assert r.headers.get("Back-Context") == "track"


def test_back_context_from_referer_track(tmp_path: Path) -> None:
    app = create_app()
    app.state.run_store = RunStore(tmp_path / "runs.db")
    with TestClient(app) as c:
        _seed_run(app, "r3")
        r = c.get("/result/r3", headers={"Referer": "http://testserver/track"})
        assert r.status_code == 200
        assert r.headers.get("Back-Context") == "track"
