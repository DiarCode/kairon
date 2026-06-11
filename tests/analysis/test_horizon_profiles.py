"""Tests for the horizon profiles + build_run_result helper (US-003).

Engine-shape work: the engine gains a ``HORIZON_PROFILES`` registry and
``build_run_result`` rebadges the 2 underlying model heads (``lr``/``tree``)
into the 4 spec-facing model names. The ``run_analysis`` signature is
FROZEN and these tests assert it remains so.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from kairon.analysis.contracts import ProvenanceBlock
from kairon.analysis.engine import (
    AnalysisResult,
    CurrentState,
    ModelPrediction,
    build_run_result,
    HORIZON_PROFILES,
    HorizonProfile,
    run_analysis,
)
from kairon.analysis.risk import RiskLevels


def _provenance() -> ProvenanceBlock:
    return ProvenanceBlock(
        config_hash="cfg-1",
        data_hash="data-1",
        model_version="kairon-0.1.0",
        seed=42,
    )


def _stub_current_state(close: float = 100.0) -> CurrentState:
    """Build a fully-populated CurrentState for tests."""
    return CurrentState(
        timestamp=datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC),
        close=close,
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
        bb_mid=close,
        bb_lower=close - 1.0,
        ema_50=close,
        ema_200=close,
    )


def _stub_table(close: float = 100.0) -> pa.Table:
    return pa.table(
        {
            "ts": [datetime(2026, 6, 9, 11, 0, 0, tzinfo=UTC)],
            "open": [close - 0.5],
            "high": [close + 0.5],
            "low": [close - 0.5],
            "close": [close],
            "volume": [10.0],
        }
    )


def _stub_analysis(
    *, close: float = 100.0, model_names: tuple[str, ...] = ("lr", "tree")
) -> AnalysisResult:
    preds = tuple(
        ModelPrediction(
            model_name=mn,
            direction="up",
            direction_class=2,
            confidence=0.6,
            proba=(0.2, 0.2, 0.6),
            magnitude=0.01,
            vol_forecast=0.02,
        )
        for mn in model_names
    )
    return AnalysisResult(
        table=_stub_table(close=close),
        df=pd.DataFrame({"close": [close]}),
        feature_names=("rsi_14",),
        symbol="BTC",
        timeframe="1h",
        has_volume=True,
        current_state=_stub_current_state(close=close),
        sweet_spots=(),
        risk_levels=RiskLevels(
            stop_loss_long=close * 0.99,
            stop_loss_short=close * 1.01,
            stop_loss_long_tight=close * 0.995,
            stop_loss_short_tight=close * 1.005,
            take_profit_long_1=close * 1.02,
            take_profit_long_2=close * 1.03,
            take_profit_short_1=close * 0.98,
            take_profit_short_2=close * 0.97,
            fib_tp_long=close * 1.05,
            fib_tp_short=close * 0.95,
            position_size_pct=0.01,
            atr=1.5,
        ),
        model_predictions=preds,
        pivots=(),
    )


# ---------- HORIZON_PROFILES ------------------------------------------------


def test_horizon_profiles_has_three_keys() -> None:
    assert set(HORIZON_PROFILES.keys()) == {"day", "swing", "long"}


def test_day_profile_duration_is_24h() -> None:
    p = HORIZON_PROFILES["day"]
    assert p.duration_hours == 24
    assert 0.005 <= p.stop_loss_distance_pct <= 0.05


def test_swing_profile_duration_in_5_to_7_days() -> None:
    p = HORIZON_PROFILES["swing"]
    assert 5 * 24 <= p.duration_hours <= 7 * 24
    assert 0.02 <= p.stop_loss_distance_pct <= 0.10


def test_long_profile_duration_in_30_to_90_days() -> None:
    p = HORIZON_PROFILES["long"]
    assert 30 * 24 <= p.duration_hours <= 90 * 24
    assert 0.05 <= p.stop_loss_distance_pct <= 0.25


def test_weights_sum_to_one_for_each_horizon() -> None:
    for name, profile in HORIZON_PROFILES.items():
        s = sum(profile.model_weights.values())
        assert abs(s - 1.0) < 1e-9, f"{name} weights sum to {s}"


def test_horizon_profile_is_frozen() -> None:
    p: HorizonProfile = HORIZON_PROFILES["day"]
    with pytest.raises(Exception):
        p.duration_hours = 25  # type: ignore[misc]


# ---------- build_run_result ------------------------------------------------


def test_build_run_result_emits_four_named_tiles() -> None:
    ar = _stub_analysis()
    rr = build_run_result(
        ar, horizon="day", run_id="r1", csv_path=Path("dummy"), provenance=_provenance()
    )
    names = tuple(t.name for t in rr.models)
    assert names == ("trend", "mean_reversion", "volatility", "ensemble")


def test_build_run_result_provenance_round_trips() -> None:
    ar = _stub_analysis()
    prov = _provenance()
    rr = build_run_result(
        ar, horizon="day", run_id="r1", csv_path=Path("dummy"), provenance=prov
    )
    assert rr.provenance == prov
    assert rr.provenance.model_dump_json() == prov.model_dump_json()


def test_build_run_result_horizon_propagates() -> None:
    ar = _stub_analysis()
    rr = build_run_result(
        ar, horizon="swing", run_id="r1", csv_path=Path("dummy"), provenance=_provenance()
    )
    assert rr.horizon == "swing"


def test_build_run_result_base_price_override() -> None:
    ar = _stub_analysis(close=100.0)
    rr = build_run_result(
        ar,
        horizon="day",
        run_id="r1",
        csv_path=Path("dummy"),
        provenance=_provenance(),
        base_price_override=42.0,
    )
    assert rr.base_price == 42.0
    # All tiles' ideal_entry should be 42.0
    for t in rr.models:
        assert t.ideal_entry == 42.0


def test_build_run_result_confidence_in_unit_interval() -> None:
    ar = _stub_analysis()
    rr = build_run_result(
        ar, horizon="day", run_id="r1", csv_path=Path("dummy"), provenance=_provenance()
    )
    for t in rr.models:
        assert 0.0 <= t.confidence <= 1.0, f"{t.name} confidence {t.confidence} out of range"


def test_build_run_result_no_l_head_or_t_head_in_output() -> None:
    """The 4 spec names must be present; 'lr'/'tree' must not leak through."""
    ar = _stub_analysis()
    rr = build_run_result(
        ar, horizon="day", run_id="r1", csv_path=Path("dummy"), provenance=_provenance()
    )
    names = {t.name for t in rr.models}
    assert names == {"trend", "mean_reversion", "volatility", "ensemble"}
    assert "lr" not in names
    assert "tree" not in names


def test_build_run_result_rejects_missing_engine_heads() -> None:
    ar = _stub_analysis(model_names=("lr",))  # no tree
    with pytest.raises(ValueError, match="expected engine to emit 'lr' and 'tree'"):
        build_run_result(
            ar, horizon="day", run_id="r1", csv_path=Path("dummy"), provenance=_provenance()
        )


# ---------- run_analysis signature is FROZEN --------------------------------


def test_run_analysis_signature_frozen() -> None:
    """Per US-003 AC: run_analysis signature is unchanged."""
    import inspect

    sig = inspect.signature(run_analysis)
    params = list(sig.parameters.keys())
    assert params == [
        "table",
        "symbol",
        "timeframe",
        "has_volume",
        "feature_set",
        "pivot_scale",
        "run_model",
        "horizon",
        "equity",
        "threshold",
    ]
