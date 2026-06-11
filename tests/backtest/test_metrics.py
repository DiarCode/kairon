"""Tests for performance metrics."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.backtest.metrics import (
    BARS_PER_YEAR_1D,
    BARS_PER_YEAR_5M,
    PerformanceReport,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    win_rate,
)


def test_sharpe_ratio_constant_returns() -> None:
    rets = np.zeros(10)
    assert sharpe_ratio(rets) != sharpe_ratio(rets)  # NaN


def test_sharpe_ratio_simple() -> None:
    rets = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
    s = sharpe_ratio(rets, bars_per_year=BARS_PER_YEAR_1D)
    # Sanity: positive mean positive excess return → positive Sharpe
    assert s > 0


def test_sortino_ratio_no_downside() -> None:
    rets = np.array([0.01, 0.02, 0.01, 0.005, 0.015])
    assert sortino_ratio(rets) != sortino_ratio(rets)  # NaN


def test_sortino_ratio_simple() -> None:
    rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    s = sortino_ratio(rets, bars_per_year=BARS_PER_YEAR_1D)
    assert isinstance(s, float)


def test_max_drawdown_basic() -> None:
    equity = np.array([100.0, 110.0, 105.0, 90.0, 95.0, 95.0, 100.0])
    # peak at 110, trough at 90 → -0.1818
    assert max_drawdown(equity) == pytest.approx(-0.1818, abs=1e-3)


def test_max_drawdown_monotone_up() -> None:
    equity = np.array([100.0, 110.0, 120.0, 130.0])
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_empty() -> None:
    assert max_drawdown(np.array([])) != max_drawdown(np.array([]))  # NaN


def test_win_rate_basic() -> None:
    pnl = np.array([1.0, -1.0, 2.0, 0.5, -0.5])
    assert win_rate(pnl) == 0.6


def test_win_rate_empty() -> None:
    assert win_rate(np.array([])) != win_rate(np.array([]))


def test_profit_factor_basic() -> None:
    pnl = np.array([2.0, -1.0, 3.0, -0.5])
    # gross_wins = 5, gross_losses = 1.5 → 3.333
    assert profit_factor(pnl) == pytest.approx(10.0 / 3.0)


def test_profit_factor_no_losses() -> None:
    pnl = np.array([1.0, 2.0, 3.0])
    assert profit_factor(pnl) == float("inf")


def test_profit_factor_no_wins() -> None:
    pnl = np.array([-1.0, -2.0])
    assert profit_factor(pnl) == 0.0


def test_summarize_basic() -> None:
    equity = np.array([10_000.0, 10_100.0, 10_050.0, 10_200.0, 10_150.0, 10_500.0])
    report = summarize(equity, bars_per_year=BARS_PER_YEAR_1D)
    assert isinstance(report, PerformanceReport)
    assert report.total_return > 0
    assert report.annualized_return != 0
    assert report.sharpe != 0
    assert report.max_drawdown < 0  # there was a drawdown
    assert report.n_bars == 6
    assert report.n_trades == 0


def test_summarize_with_trade_pnl() -> None:
    equity = np.array([10_000.0, 10_100.0, 10_200.0, 10_300.0])
    pnl = np.array([100.0, 50.0, -25.0, 75.0])
    report = summarize(equity, trade_pnl=pnl, bars_per_year=BARS_PER_YEAR_1D)
    assert report.n_trades == 4
    assert report.win_rate == 0.75
    assert report.profit_factor > 0


def test_summarize_short_equity() -> None:
    equity = np.array([10_000.0, 9_500.0, 9_200.0])
    report = summarize(equity, bars_per_year=BARS_PER_YEAR_1D)
    assert report.total_return < 0
    assert report.max_drawdown < 0
    assert report.calmar != 0  # some value (could be NaN if no drawdown)


def test_summarize_to_dict() -> None:
    equity = np.array([10_000.0, 10_100.0, 10_200.0])
    d = summarize(equity, bars_per_year=BARS_PER_YEAR_5M).to_dict()
    assert "sharpe" in d
    assert "n_bars" in d
    assert d["n_bars"] == 3
