"""Tests for the backtest engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from kairon.backtest.cost import CostModel
from kairon.backtest.engine import (
    BacktestSpec,
    run_backtest,
    signals_to_target,
)
from kairon.backtest.position import Side


def _series(n: int = 100, *, start: float = 100.0, drift: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    rets = rng.normal(drift, 0.02, size=n)
    closes = start * np.cumprod(1.0 + rets)
    ts = np.array(
        [datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(n)],
        dtype=object,
    )
    return ts, closes


def test_signals_to_target_basic() -> None:
    s = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.2])
    t = signals_to_target(s)
    assert t.tolist() == [-1, -1, 0, 1, 1, 1, -1]


def test_signals_to_target_with_min_change() -> None:
    s = np.array([-1.0, -0.05, 0.05, 1.0, 0.1])
    t = signals_to_target(s, min_signal_change=0.2)
    # middle 3 are below 0.2 -> 0
    assert t.tolist() == [-1, 0, 0, 1, 0]


def test_signals_to_target_rejects_ndim() -> None:
    with pytest.raises(ValueError, match="1-D"):
        signals_to_target(np.zeros((2, 2)))


def test_signals_to_target_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="min_signal_change"):
        signals_to_target(np.zeros(5), min_signal_change=-0.1)


def test_run_backtest_validates_lengths() -> None:
    ts, c = _series(50)
    s = np.zeros(40)  # wrong length
    with pytest.raises(ValueError, match="rows"):
        run_backtest(symbol="X", timestamps=ts, close=c, signals=s)


def test_run_backtest_no_trades_stays_flat() -> None:
    ts, c = _series(50)
    s = np.zeros(50)
    spec = BacktestSpec(initial_equity=10_000.0)
    result = run_backtest(symbol="X", timestamps=ts, close=c, signals=s, spec=spec)
    assert result.n_trades == 0
    assert result.equity_curve[0] == pytest.approx(10_000.0)
    assert result.final_equity == pytest.approx(10_000.0)


def test_run_backtest_long_only_on_long_signal() -> None:
    ts, c = _series(50, drift=0.0)  # zero drift so price is roughly stable
    s = np.ones(50)  # always long
    spec = BacktestSpec(initial_equity=10_000.0, fraction=1.0, cost=CostModel(commission_bps=0, slippage_bps=0, half_spread_bps=0))
    result = run_backtest(symbol="X", timestamps=ts, close=c, signals=s, spec=spec)
    # One open at bar 0, still open at end -> 1 trade but closed at last bar
    assert result.n_trades == 1
    final = result.final_equity
    # We invested all equity, so final ≈ initial (costs are zero, drift is zero)
    # The exact value depends on the random walk, so allow a 30% band.
    assert 7_000.0 < final < 13_000.0, f"final {final} too far from initial"


def test_run_backtest_long_then_flat() -> None:
    ts, c = _series(20)
    s = np.array([1] * 10 + [0] * 10)
    spec = BacktestSpec(initial_equity=10_000.0, fraction=1.0, cost=CostModel(commission_bps=0, slippage_bps=0, half_spread_bps=0))
    result = run_backtest(symbol="X", timestamps=ts, close=c, signals=s, spec=spec)
    assert result.n_trades == 1
    assert result.trades[0].closed_at is not None
    assert result.trades[0].side == Side.LONG


def test_run_backtest_costs_reduce_pnl() -> None:
    ts, c = _series(20)
    s = np.array([1] * 10 + [0] * 10)
    spec = BacktestSpec(initial_equity=10_000.0, fraction=1.0, cost=CostModel(commission_bps=10, slippage_bps=0, half_spread_bps=0))
    result = run_backtest(symbol="X", timestamps=ts, close=c, signals=s, spec=spec)
    t = result.trades[0]
    # Entry + exit fees = 20 bps
    assert t.entry_costs > 0
    assert t.exit_costs > 0
    # Net PnL is gross PnL minus costs
    assert t.exit_price is not None
    assert t.pnl < (t.size * (t.exit_price - t.entry_price))


def test_run_backtest_thrashing_prevention() -> None:
    """A 1-0-1-0 signal sequence does open/close a position per bar.

    Without an explicit debounce, every 1→0 transition is a close and
    every 0→1 transition is an open. The mitigation in practice is to
    smooth the signal upstream (EMA, hysteresis) or use
    ``spec.min_signal_change``. This test pins the *current* behaviour
    so any future change to debounce is intentional.
    """
    ts, c = _series(10)
    s = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    spec = BacktestSpec(initial_equity=10_000.0, fraction=1.0, cost=CostModel(commission_bps=0, slippage_bps=0, half_spread_bps=0))
    result = run_backtest(symbol="X", timestamps=ts, close=c, signals=s, spec=spec)
    # 5 0→1 transitions → 5 opens; each followed by a 1→0 close on the
    # next bar. So 5 round-trip trades.
    assert result.n_trades == 5


def test_backtest_spec_validates() -> None:
    with pytest.raises(ValueError):
        BacktestSpec(initial_equity=0)
    with pytest.raises(ValueError):
        BacktestSpec(fraction=0)
    with pytest.raises(ValueError):
        BacktestSpec(fraction=1.1)
    with pytest.raises(ValueError):
        BacktestSpec(sizing="not_a_sizing")  # type: ignore[arg-type]


def test_backtest_result_to_table_round_trip() -> None:
    ts, c = _series(20)
    s = np.array([1] * 10 + [0] * 10)
    spec = BacktestSpec(initial_equity=10_000.0, fraction=1.0, cost=CostModel(commission_bps=0, slippage_bps=0, half_spread_bps=0))
    result = run_backtest(symbol="X", timestamps=ts, close=c, signals=s, spec=spec)
    table = result.to_table()
    assert table.num_rows == result.n_trades
    assert "pnl" in table.column_names
    assert "side" in table.column_names


def test_backtest_result_to_dict() -> None:
    ts, c = _series(20)
    s = np.array([1] * 10 + [0] * 10)
    result = run_backtest(symbol="X", timestamps=ts, close=c, signals=s, spec=BacktestSpec(initial_equity=10_000.0, fraction=1.0, cost=CostModel(commission_bps=0, slippage_bps=0, half_spread_bps=0)))
    d = result.to_table()
    assert d.num_rows == result.n_trades
