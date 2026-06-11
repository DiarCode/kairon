"""Tests for momentum indicators: RSI, Stochastic, Williams %R, CCI."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.technical.momentum import cci, rsi, stochastic, williams_r


def _mk(close: list[float], high: list[float] | None = None, low: list[float] | None = None) -> pa.Table:
    n = len(close)
    opn = [c - 0.5 for c in close]
    h = high or [c + 1.0 for c in close]
    lo = low or [c - 1.0 for c in close]
    return pa.table(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC) for _ in range(n)],
            "open": opn,
            "high": h,
            "low": lo,
            "close": close,
            "volume": [10.0] * n,
        },
        schema=OHLCV_SCHEMA,
    )


def test_rsi_constant_50() -> None:
    """A flat series produces NaN (division by zero in Wilder)."""
    t = _mk([100.0] * 20)
    out = rsi(t, period=14)
    vals = out.column("rsi_14").to_pylist()
    # No losses and no gains → RSI undefined for many rows
    assert any(math.isnan(v) for v in vals)


def test_rsi_uptrend_near_100() -> None:
    """A monotonically increasing series should have RSI close to 100."""
    t = _mk([100.0 + i for i in range(30)])
    out = rsi(t, period=14)
    vals = out.column("rsi_14").to_pylist()
    last = [v for v in vals if not math.isnan(v)][-1]
    assert last > 90.0


def test_rsi_downtrend_near_0() -> None:
    t = _mk([200.0 - i for i in range(30)])
    out = rsi(t, period=14)
    vals = out.column("rsi_14").to_pylist()
    last = [v for v in vals if not math.isnan(v)][-1]
    assert last < 10.0


def test_rsi_rejects_period_zero() -> None:
    t = _mk([100.0, 101.0])
    with pytest.raises(ValueError):
        rsi(t, period=0)


def test_stochastic_columns_present() -> None:
    t = _mk([100.0 + i for i in range(30)])
    out = stochastic(t)
    assert "stoch_k" in out.column_names
    assert "stoch_d" in out.column_names


def test_stochastic_k_in_range() -> None:
    t = _mk([100.0 + (i % 10) for i in range(30)])
    out = stochastic(t)
    vals = [v for v in out.column("stoch_k").to_pylist() if not math.isnan(v)]
    assert all(0.0 <= v <= 100.0 for v in vals)


def test_williams_r_in_range() -> None:
    t = _mk([100.0 + (i % 5) for i in range(20)])
    out = williams_r(t, period=14)
    vals = [v for v in out.column("williams_r").to_pylist() if not math.isnan(v)]
    assert all(-100.0 <= v <= 0.0 for v in vals)


def test_cci_basic() -> None:
    t = _mk([100.0 + i * 0.5 for i in range(40)])
    out = cci(t, period=20)
    vals = [v for v in out.column("cci").to_pylist() if not math.isnan(v)]
    assert len(vals) > 0
    # CCI on a strong uptrend should be positive
    assert vals[-1] > 0
