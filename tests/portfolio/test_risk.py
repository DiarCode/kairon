"""Tests for the risk & portfolio layer."""

from __future__ import annotations

import pytest

from kairon.portfolio import (
    ExposureLimits,
    PortfolioSignal,
    SizingConfig,
    aggregate_signals,
    check_exposure,
    fixed_fraction_size,
    kelly_size,
    size_position,
    vol_target_size,
)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def test_sizing_config_validation() -> None:
    with pytest.raises(ValueError):
        SizingConfig(method="sideways")
    with pytest.raises(ValueError):
        SizingConfig(fraction=0)
    with pytest.raises(ValueError):
        SizingConfig(fraction=1.5)
    with pytest.raises(ValueError):
        SizingConfig(kelly_cap=0)
    with pytest.raises(ValueError):
        SizingConfig(vol_target_annual=0)


def test_fixed_fraction_size() -> None:
    assert fixed_fraction_size(equity=10_000, price=100, fraction=0.1) == pytest.approx(10.0)


def test_fixed_fraction_size_validates() -> None:
    with pytest.raises(ValueError):
        fixed_fraction_size(equity=0, price=100, fraction=0.1)
    with pytest.raises(ValueError):
        fixed_fraction_size(equity=10_000, price=0, fraction=0.1)
    with pytest.raises(ValueError):
        fixed_fraction_size(equity=10_000, price=100, fraction=0)


def test_kelly_size_positive_edge() -> None:
    # p=0.6, b=1 (1:1 pay-off) → kelly = 0.2
    size = kelly_size(equity=10_000, price=100, win_rate=0.6, avg_win=100, avg_loss=100, cap=0.25)
    assert size == pytest.approx(20.0)  # 0.2 * 10_000 / 100 = 20


def test_kelly_size_clipped_to_cap() -> None:
    # Build a case where kelly > cap: small b
    size = kelly_size(equity=10_000, price=100, win_rate=0.55, avg_win=100, avg_loss=100, cap=0.05)
    # b = 1; kelly = 0.1 / 1 = 0.1 → clipped to 0.05
    assert size == pytest.approx(5.0)


def test_kelly_size_zero_edge_returns_zero() -> None:
    # p=0.5 → kelly = 0
    size = kelly_size(equity=10_000, price=100, win_rate=0.5, avg_win=100, avg_loss=100, cap=0.25)
    assert size == 0.0


def test_kelly_size_validates() -> None:
    with pytest.raises(ValueError):
        kelly_size(equity=10_000, price=100, win_rate=0, avg_win=1, avg_loss=1)
    with pytest.raises(ValueError):
        kelly_size(equity=10_000, price=100, win_rate=1, avg_win=1, avg_loss=1)
    with pytest.raises(ValueError):
        kelly_size(equity=10_000, price=100, win_rate=0.5, avg_win=0, avg_loss=1)
    with pytest.raises(ValueError):
        kelly_size(equity=10_000, price=100, win_rate=0.5, avg_win=1, avg_loss=0)
    with pytest.raises(ValueError):
        kelly_size(equity=10_000, price=100, win_rate=0.5, avg_win=1, avg_loss=1, cap=0)


def test_vol_target_size() -> None:
    # target 10%, forecast 20% → notional 50% of equity
    size = vol_target_size(equity=10_000, price=100, vol_forecast_annual=0.20, vol_target_annual=0.10)
    assert size == pytest.approx(50.0)  # 5_000 / 100


def test_vol_target_size_validates() -> None:
    with pytest.raises(ValueError):
        vol_target_size(equity=10_000, price=100, vol_forecast_annual=0)
    with pytest.raises(ValueError):
        vol_target_size(equity=10_000, price=100, vol_forecast_annual=0.1, vol_target_annual=0)


def test_size_position_dispatch() -> None:
    cfg = SizingConfig(method="fixed_fraction", fraction=0.1)
    assert size_position(equity=10_000, price=100, config=cfg) == pytest.approx(10.0)
    cfg = SizingConfig(method="kelly", kelly_cap=0.25)
    assert size_position(
        equity=10_000, price=100, config=cfg,
        win_rate=0.6, avg_win=100, avg_loss=100,
    ) == pytest.approx(20.0)
    cfg = SizingConfig(method="vol_target", vol_target_annual=0.10)
    assert size_position(
        equity=10_000, price=100, config=cfg, vol_forecast_annual=0.20,
    ) == pytest.approx(50.0)


def test_size_position_dispatch_rejects_missing_args() -> None:
    cfg = SizingConfig(method="kelly")
    with pytest.raises(ValueError):
        size_position(equity=10_000, price=100, config=cfg)
    cfg = SizingConfig(method="vol_target")
    with pytest.raises(ValueError):
        size_position(equity=10_000, price=100, config=cfg)


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------
def test_exposure_limits_validates() -> None:
    with pytest.raises(ValueError):
        ExposureLimits(max_position_equity_fraction=0)
    with pytest.raises(ValueError):
        ExposureLimits(max_position_equity_fraction=1.5)
    with pytest.raises(ValueError):
        ExposureLimits(max_total_leverage=0)
    with pytest.raises(ValueError):
        ExposureLimits(max_positions=0)


def test_exposure_allows_within_limits() -> None:
    limits = ExposureLimits(max_position_equity_fraction=0.5, max_total_leverage=1.0)
    ok, reason = check_exposure(
        candidate_symbol="BTC/USDT",
        candidate_size=10,
        candidate_price=100,
        existing={"ETH/USDT": (5, 100)},
        equity=10_000,
        limits=limits,
    )
    assert ok
    assert reason == "ok"


def test_exposure_blocks_per_position_cap() -> None:
    limits = ExposureLimits(max_position_equity_fraction=0.10)
    ok, reason = check_exposure(
        candidate_symbol="BTC/USDT",
        candidate_size=10,
        candidate_price=200,  # 2000 / 10000 = 20% > 10%
        existing={},
        equity=10_000,
        limits=limits,
    )
    assert not ok
    assert "exceeds" in reason


def test_exposure_blocks_total_leverage() -> None:
    # Per-position cap 80%, total-leverage 1.0x equity
    # Existing position: 50 lots * 50 = 2500 (25% of equity)
    # Use 80% per-position cap so the candidate passes the per-position test
    # but together with existing exceeds 100% leverage.
    limits = ExposureLimits(max_position_equity_fraction=0.8, max_total_leverage=1.0)
    # existing: 50 * 50 = 2500 (25% of equity)
    # candidate: 100 * 80 = 8000 (80% of equity) — passes per-position check
    # total: 10_500 / 10_000 = 1.05 > 1.0
    ok, reason = check_exposure(
        candidate_symbol="BTC/USDT",
        candidate_size=100,
        candidate_price=80,
        existing={"ETH/USDT": (50, 50)},
        equity=10_000,
        limits=limits,
    )
    assert not ok
    assert "leverage" in reason


def test_exposure_blocks_max_positions() -> None:
    limits = ExposureLimits(max_positions=1)
    ok, reason = check_exposure(
        candidate_symbol="BTC/USDT",
        candidate_size=1,
        candidate_price=100,
        existing={"ETH/USDT": (1, 100)},
        equity=10_000,
        limits=limits,
    )
    assert not ok
    assert "max_positions" in reason


def test_exposure_allows_adding_to_existing() -> None:
    """Adding to an existing position should not bump n_positions."""
    limits = ExposureLimits(max_positions=1)
    ok, _ = check_exposure(
        candidate_symbol="BTC/USDT",
        candidate_size=1,
        candidate_price=100,
        existing={"BTC/USDT": (1, 100)},
        equity=10_000,
        limits=limits,
    )
    assert ok


def test_exposure_validates_equity() -> None:
    limits = ExposureLimits()
    ok, _ = check_exposure(
        candidate_symbol="X", candidate_size=1, candidate_price=100,
        existing={}, equity=0, limits=limits,
    )
    assert not ok


def test_exposure_validates_candidate() -> None:
    limits = ExposureLimits()
    ok, _ = check_exposure(
        candidate_symbol="X", candidate_size=0, candidate_price=100,
        existing={}, equity=10_000, limits=limits,
    )
    assert not ok
    ok, _ = check_exposure(
        candidate_symbol="X", candidate_size=1, candidate_price=0,
        existing={}, equity=10_000, limits=limits,
    )
    assert not ok


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------
def test_aggregate_signals_basic() -> None:
    sigs = {"A": 0.8, "B": -0.5, "C": 0.0}
    ps = aggregate_signals(sigs)
    assert isinstance(ps, PortfolioSignal)
    assert ps.weights["A"] == pytest.approx(0.8)
    assert ps.weights["B"] == pytest.approx(-0.5)
    assert ps.weights["C"] == pytest.approx(0.0)
    assert ps.gross == pytest.approx(1.3)
    assert ps.net == pytest.approx(0.3)
    assert ps.n_long == 1
    assert ps.n_short == 1
    assert ps.n_flat == 1


def test_aggregate_signals_clipped_to_minus_one_one() -> None:
    sigs = {"A": 5.0, "B": -10.0}
    ps = aggregate_signals(sigs)
    assert ps.weights["A"] == 1.0
    assert ps.weights["B"] == -1.0


def test_aggregate_signals_confidence_floor() -> None:
    sigs = {"A": 0.05, "B": 0.5, "C": -0.5}
    ps = aggregate_signals(sigs, confidence_floor=0.1)
    assert ps.weights["A"] == 0.0
    assert ps.weights["B"] == pytest.approx(0.5)
    assert ps.n_flat == 1
    assert ps.n_long == 1
    assert ps.n_short == 1


def test_aggregate_signals_validates() -> None:
    with pytest.raises(ValueError):
        aggregate_signals({}, confidence_floor=-0.1)
    with pytest.raises(ValueError):
        aggregate_signals({}, confidence_floor=2.0)


def test_aggregate_signals_empty() -> None:
    ps = aggregate_signals({})
    assert ps.gross == 0.0
    assert ps.net == 0.0
    assert ps.n_long == 0
