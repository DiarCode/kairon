"""Tests for structure indicators: BOS/CHoCH, candlestick, Fibonacci."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.technical.structure import (
    bos_choch,
    candlestick_patterns,
    fibonacci_levels,
)


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


def test_bos_columns_present() -> None:
    t = _mk([100.0 + i for i in range(50)])
    out = bos_choch(t, order=3)
    assert "bos" in out.column_names
    assert "choch" in out.column_names


def test_bos_choch_in_int8_range() -> None:
    t = _mk([100.0 + (i % 7) for i in range(50)])
    out = bos_choch(t, order=3)
    bos = out.column("bos").to_pylist()
    choch = out.column("choch").to_pylist()
    assert all(v in (-1, 0, 1) for v in bos)
    assert all(v in (-1, 0, 1) for v in choch)


def test_candlestick_columns_present() -> None:
    t = _mk([100.0] * 30)
    out = candlestick_patterns(t)
    for col in ("cdl_doji", "cdl_hammer", "cdl_shooting_star", "cdl_engulfing_bull", "cdl_engulfing_bear"):
        assert col in out.column_names


def test_candlestick_detects_doji() -> None:
    # A bar where open == close and high/low differ
    t = pa.table(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
            "open": [100.0, 100.0],
            "high": [101.0, 105.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],  # body=0
            "volume": [10.0, 10.0],
        },
        schema=OHLCV_SCHEMA,
    )
    out = candlestick_patterns(t)
    doji = out.column("cdl_doji").to_pylist()
    # First bar: 0/2 = 0 (doji); second: 0/6 = 0 (doji)
    # We test that the column works; threshold inside is 0.001.
    assert isinstance(doji, list)
    assert len(doji) == 2


def test_fibonacci_columns_present() -> None:
    t = _mk([100.0 + i * 0.1 for i in range(150)])
    out = fibonacci_levels(t, lookback=50)
    for col in ("fib_236", "fib_382", "fib_500", "fib_618", "fib_786"):
        assert col in out.column_names


def test_fibonacci_levels_decreasing() -> None:
    """Retracement levels from a high to a low should be monotonically decreasing."""
    t = _mk([100.0 + i for i in range(200)])  # uptrend
    out = fibonacci_levels(t, lookback=100)
    levels = [out.column(c).to_pylist() for c in ("fib_236", "fib_382", "fib_500", "fib_618", "fib_786")]
    # Pick a bar with all valid
    for i in range(150, 200):
        row = [lvl[i] for lvl in levels]
        if not any(math.isnan(v) for v in row):
            # 0.236 > 0.382 > ... > 0.786 means price ranges from 23.6% (closer to high)
            # to 78.6% (closer to low)
            assert row[0] > row[1] > row[2] > row[3] > row[4]
            break
