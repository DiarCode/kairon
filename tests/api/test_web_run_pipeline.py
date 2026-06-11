"""Tests for the /api/runs pipeline + /api/runs/<id>/save (US-007)."""
from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pyarrow as pa

from kairon.analysis.contracts import ProvenanceBlock
from kairon.analysis.engine import AnalysisResult
from kairon.api.app import create_app
from kairon.store.runs import RunStore


def _stub_current_state() -> Any:
    from kairon.analysis.engine import CurrentState

    return CurrentState(
        timestamp=datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC),
        close=100.0,
        ew_wave_position=0.0,
        ew_wave_direction=0.0,
        ew_is_impulse=False,
        ew_completion_prob=0.5,
        ew_fib_confluence=0.5,
        regime="ranging",
        regime_prob_trending=0.25,
        regime_prob_ranging=0.5,
        regime_prob_volatile=0.2,
        regime_prob_stressed=0.05,
        hurst_exp=0.5,
        garch_vol=0.02,
        atr_14=1.5,
        rsi_14=50.0,
        fib_dist_236=0.01,
        fib_dist_382=0.02,
        fib_dist_500=0.03,
        fib_dist_618=0.04,
        fib_dist_786=0.05,
        fvg_bullish=False,
        fvg_bearish=False,
        fvg_fill_pct=0.0,
        fvg_nearest_distance=0.0,
        ob_in_bullish_zone=False,
        ob_in_bearish_zone=False,
        ob_bullish_near=False,
        ob_bearish_near=False,
        bos_direction=0,
        choch_direction=0,
        bb_upper=101.0,
        bb_mid=100.0,
        bb_lower=99.0,
        ema_50=100.0,
        ema_200=100.0,
    )


def test_post_runs_persists_a_run(tmp_path: Path, monkeypatch: Any) -> None:
    """Upload a CSV, then start a run, monkeypatching run_analysis to return a canned result."""
    import pandas as pd
    from kairon.analysis.engine import ModelPrediction

    app = create_app()
    app.state.run_store = RunStore(tmp_path / "runs.db")

    csv = b"ts,open,high,low,close,volume\n2026-01-01,1,2,0.5,1.5,10\n2026-01-02,2,3,1.5,2.5,20\n"
    stub_table = pa.table(
        {
            "ts": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        }
    )

    def fake_run_analysis(table: Any, **_: Any) -> AnalysisResult:
        return AnalysisResult(
            table=stub_table,
            df=pd.DataFrame({"close": [1.5]}),
            feature_names=("rsi_14",),
            symbol="BTC",
            timeframe="1h",
            has_volume=True,
            current_state=_stub_current_state(),
            sweet_spots=(),
            risk_levels=__import__("kairon.analysis.risk", fromlist=["RiskLevels"]).RiskLevels(
                stop_loss_long=99.0, stop_loss_short=101.0, stop_loss_long_tight=99.5,
                stop_loss_short_tight=100.5, take_profit_long_1=102.0, take_profit_long_2=103.0,
                take_profit_short_1=98.0, take_profit_short_2=97.0, fib_tp_long=105.0,
                fib_tp_short=95.0, position_size_pct=0.01, atr=1.5,
            ),
            model_predictions=(
                ModelPrediction(
                    model_name="lr", direction="up", direction_class=2,
                    confidence=0.6, proba=(0.2, 0.2, 0.6), magnitude=0.01, vol_forecast=0.02,
                ),
                ModelPrediction(
                    model_name="tree", direction="up", direction_class=2,
                    confidence=0.55, proba=(0.25, 0.2, 0.55), magnitude=0.008, vol_forecast=0.02,
                ),
            ),
            pivots=(),
        )

    with mock.patch("kairon.analysis.engine.run_analysis", fake_run_analysis):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            # 1) Upload
            r1 = c.post("/api/uploads", files={"file": ("x.csv", io.BytesIO(csv), "text/csv")})
            assert r1.status_code == 200, r1.text
            run_id = r1.json()["run_id"]
            csv_path = r1.json()["csv_path"]

            # 2) Start the run
            print("DEBUG csv_path:", csv_path, "exists:", Path(csv_path).exists(), "size:", Path(csv_path).stat().st_size if Path(csv_path).exists() else 0)
            r2 = c.post(
                "/api/runs",
                json={"run_id": run_id, "horizon": "day", "csv_path": csv_path},
            )
            assert r2.status_code == 200, r2.text
            body = r2.json()
            assert body["run_id"] == run_id
            assert body["status"] == "done"

            # 3) Status
            r3 = c.get(f"/api/runs/{run_id}")
            assert r3.status_code == 200
            assert r3.json()["status"] == "done"

            # 4) Result page renders with 4 model tiles
            r4 = c.get(f"/result/{run_id}")
            assert r4.status_code == 200
            for name in ("trend", "mean_reversion", "volatility", "ensemble"):
                assert f'data-name="{name}"' in r4.text


def test_post_runs_save_pins_a_run(tmp_path: Path, monkeypatch: Any) -> None:
    """The [Save] button wires to a real set_pinned() action."""
    import pandas as pd
    from kairon.analysis.engine import ModelPrediction

    app = create_app()
    app.state.run_store = RunStore(tmp_path / "runs.db")

    rows = "\n".join(
        f"2026-01-01T{i // 24:02d}:{i % 60:02d}:00,100,101,99,100,10" for i in range(100)
    )
    csv = f"ts,open,high,low,close,volume\n{rows}\n".encode("utf-8")
    stub_table = pa.table(
        {
            "ts": [datetime(2026, 1, 1, tzinfo=UTC)],
            "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0],
        }
    )

    def fake_run_analysis(table: Any, **_: Any) -> AnalysisResult:
        return AnalysisResult(
            table=stub_table, df=pd.DataFrame({"close": [1.5]}),
            feature_names=("rsi_14",), symbol="BTC", timeframe="1h", has_volume=True,
            current_state=_stub_current_state(), sweet_spots=(),
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

    with mock.patch("kairon.analysis.engine.run_analysis", fake_run_analysis):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            r1 = c.post("/api/uploads", files={"file": ("x.csv", io.BytesIO(csv), "text/csv")})
            run_id = r1.json()["run_id"]
            csv_path = r1.json()["csv_path"]
            c.post("/api/runs", json={"run_id": run_id, "horizon": "day", "csv_path": csv_path})

            # Save (pin)
            r2 = c.post(f"/api/runs/{run_id}/save", json={"pinned": True})
            assert r2.status_code == 200
            assert r2.json()["pinned"] is True

            # Unsave
            r3 = c.post(f"/api/runs/{run_id}/save", json={"pinned": False})
            assert r3.status_code == 200
            assert r3.json()["pinned"] is False
