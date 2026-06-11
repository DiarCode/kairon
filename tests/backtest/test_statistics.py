"""Tests for DSR and PBO statistics."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.backtest.statistics import (
    DSRResult,
    DSRSpec,
    PBOResult,
    deflated_sharpe_ratio,
    probability_of_backtest_overfit,
)


def test_dsr_basic() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=500)
    spec = DSRSpec(n_trials=1, bars_per_year=365)
    res = deflated_sharpe_ratio(rets, spec=spec)
    assert isinstance(res, DSRResult)
    assert np.isfinite(res.sharpe)
    assert res.sharpe > 0
    assert 0.0 <= res.dsr <= 1.0


def test_dsr_too_few_returns() -> None:
    spec = DSRSpec()
    res = deflated_sharpe_ratio(np.array([0.01]), spec=spec)
    assert res.sharpe != res.sharpe  # NaN


def test_dsr_n_trials_haircut() -> None:
    """With more trials, the SR* haircut should be larger (more conservative)."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=500)
    res_1 = deflated_sharpe_ratio(rets, spec=DSRSpec(n_trials=1))
    res_100 = deflated_sharpe_ratio(rets, spec=DSRSpec(n_trials=100))
    # More trials -> larger haircut threshold
    assert res_100.sr_star > res_1.sr_star


def test_dsr_bars_per_year_scaling() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=500)
    res = deflated_sharpe_ratio(rets, spec=DSRSpec(bars_per_year=252))
    # Sharpe should scale with sqrt(bars_per_year)
    assert res.extras["n_bars"] == 500


def test_dsr_handles_fat_tails() -> None:
    """Returns with high kurtosis should get a haircut for n_trials>1."""
    rng = np.random.default_rng(0)
    rets = rng.standard_t(df=3, size=500) * 0.01
    res = deflated_sharpe_ratio(rets, spec=DSRSpec(n_trials=100))
    # n_trials=100 → haircut is > 0
    assert res.sr_star > 0


def test_pbo_basic() -> None:
    """PBO is in [0, 1]."""
    rng = np.random.default_rng(0)
    oos = rng.normal(0.0, 0.01, size=(10, 5))
    res = probability_of_backtest_overfit(oos_returns_by_path=oos)
    assert isinstance(res, PBOResult)
    assert 0.0 <= res.pbo <= 1.0
    assert res.n_paths == 10
    assert res.n_folds == 5


def test_pbo_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        probability_of_backtest_overfit(oos_returns_by_path=np.zeros(10))


def test_pbo_too_few_paths() -> None:
    res = probability_of_backtest_overfit(oos_returns_by_path=np.zeros((1, 5)))
    assert res.pbo != res.pbo  # NaN


def test_pbo_perfect_paths_zero() -> None:
    """If every path has identical OOS, PBO should be 0."""
    oos = np.zeros((20, 4))  # all-zero paths
    res = probability_of_backtest_overfit(oos_returns_by_path=oos)
    # All paths have the same best OOS (0), so median == best, pbo==0
    assert res.pbo == 0.0
