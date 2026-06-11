"""End-to-end round trip: Upload -> Configure -> Analyze -> Result -> Track (US-008).

Exercises the full happy path through the FastAPI app, monkeypatching
``run_analysis`` and ``fetch_current_price`` so the test is hermetic.
"""
from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import pyarrow as pa
import pandas as pd
from fastapi.testclient import TestClient

from kairon.analysis.engine import AnalysisResult, CurrentState, ModelPrediction
from kairon.api.app import create_app
from kairon.store.runs import RunStore
from kairon.store.verifier import run_once


def _stub_cs() -> CurrentState:
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
        fvg_nearest_distance=0.0,
        ob_in_bullish_zone=False, ob_in_bearish_zone=False,
        ob_bullish_near=False, ob_bearish_near=False,
        bos_direction=0, choch_direction=0, bb_upper=101.0, bb_mid=100.0, bb_lower=99.0,
        ema_50=100.0, ema_200=100.0,
    )


def _fake_run_analysis(table: Any, **_: Any) -> AnalysisResult:
    return AnalysisResult(
        table=pa.table({"ts": [datetime(2026, 1, 1, tzinfo=UTC)], "open": [1.0],
                       "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]}),
        df=pd.DataFrame({"close": [1.5]}),
        feature_names=("rsi_14",), symbol="BTC", timeframe="1h", has_volume=True,
        current_state=_stub_cs(), sweet_spots=(),
        risk_levels=__import__("kairon.analysis.risk", fromlist=["RiskLevels"]).RiskLevels(
            stop_loss_long=99.0, stop_loss_short=101.0, stop_loss_long_tight=99.5,
            stop_loss_short_tight=100.5, take_profit_long_1=102.0, take_profit_long_2=103.0,
            take_profit_short_1=98.0, take_profit_short_2=97.0, fib_tp_long=105.0,
            fib_tp_short=95.0, position_size_pct=0.01, atr=1.5,
        ),
        model_predictions=(
            ModelPrediction(model_name="lr", direction="up", direction_class=2,
                           confidence=0.6, proba=(0.2, 0.2, 0.6), magnitude=0.01, vol_forecast=0.02),
            ModelPrediction(model_name="tree", direction="up", direction_class=2,
                           confidence=0.55, proba=(0.25, 0.2, 0.55), magnitude=0.008, vol_forecast=0.02),
        ),
        pivots=(),
    )


def test_full_round_trip_with_lifespan_and_verifier(tmp_path: Path) -> None:
    """Upload -> Run -> Result -> Track. Then the verifier thread (started
    by the lifespan) ticks once and writes back the actual_pct."""
    rows = "\n".join(
        f"2026-01-01T{i // 24:02d}:{i % 60:02d}:00,100,101,99,100,10" for i in range(100)
    )
    csv = f"ts,open,high,low,close,volume\n{rows}\n".encode("utf-8")

    with mock.patch("kairon.analysis.engine.run_analysis", _fake_run_analysis), \
         mock.patch("kairon.live.feed.fetch_current_price", return_value=101.0):
        app = create_app()
        app.state.run_store = RunStore(tmp_path / "runs.db")
        with TestClient(app) as c:
            # 1) Upload
            r1 = c.post("/api/uploads", files={"file": ("x.csv", io.BytesIO(csv), "text/csv")})
            assert r1.status_code == 200, r1.text
            run_id = r1.json()["run_id"]
            csv_path = r1.json()["csv_path"]
            assert r1.json()["row_count"] == 100

            # 2) Start the run
            r2 = c.post(
                "/api/runs",
                json={"run_id": run_id, "horizon": "day", "csv_path": csv_path},
            )
            assert r2.status_code == 200, r2.text
            assert r2.json()["status"] == "done"

            # 3) Result page
            r3 = c.get(f"/result/{run_id}")
            assert r3.status_code == 200
            for name in ("trend", "mean_reversion", "volatility", "ensemble"):
                assert f'data-name="{name}"' in r3.text

            # 4) The lifespan started the verifier thread; nudge the run's
            #    created_at_utc into the past so the verifier considers it
            #    due, then poke run_once once with the mocked fetch price.
            from kairon.analysis.contracts import ModelTile, ProvenanceBlock, RunResult

            run = app.state.run_store.get(run_id)
            assert run is not None
            # rewrite the row with created_at_utc 25h ago
            older = run.model_copy(update={"created_at_utc": datetime.now(UTC) - timedelta(hours=25)})
            app.state.run_store.create(older, Path(csv_path))

            n = run_once(
                app.state.run_store,
                fetch_price_fn=lambda asset, venue: 101.0,
                now_utc=datetime.now(UTC),
                base_price_reader=lambda r, p: 100.0,
            )
            assert n == 1

            # 5) Track page shows the run with actual_pct populated
            r4 = c.get("/track")
            assert r4.status_code == 200
            assert run_id in r4.text
            assert "hit" in r4.text or "missed" in r4.text
            # predicted_pct=0.01, actual_pct=0.01 -> both render as "1.00%"
            assert "1.00%" in r4.text
