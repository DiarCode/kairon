"""Tests for volatility indicators: Bollinger Bands, ATR."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.technical.volatility import atr, bollinger


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


def test_bollinger_columns_present() -> None:
    t = _mk([100.0 + i for i in range(30)])
    out = bollinger(t, period=20, std_dev=2.0)
    for col in ("bb_mid", "bb_upper", "bb_lower"):
        assert col in out.column_names


def test_bollinger_mid_equals_sma() -> None:
    close = [100.0 + 0.1 * i for i in range(30)]
    t = _mk(close)
    out = bollinger(t, period=20, std_dev=2.0)
    mid = out.column("bb_mid").to_pylist()
    expected = sum(close[:20]) / 20
    assert abs(mid[19] - expected) < 1e-9


def test_bollinger_upper_above_lower() -> None:
    t = _mk([100.0 + (i % 5) for i in range(30)])
    out = bollinger(t, period=20, std_dev=2.0)
    upper = out.column("bb_upper").to_pylist()
    lower = out.column("bb_lower").to_pylist()
    for u, lo in zip(upper, lower, strict=True):
        if not (math.isnan(u) or math.isnan(lo)):
            assert u >= lo


def test_bollinger_rejects_invalid() -> None:
    t = _mk([100.0] * 30)
    with pytest.raises(ValueError):
        bollinger(t, period=0)
    with pytest.raises(ValueError):
        bollinger(t, period=20, std_dev=0.0)


def test_atr_columns_present() -> None:
    t = _mk([100.0 + i for i in range(30)])
    out = atr(t, period=14)
    assert "atr_14" in out.column_names


def test_atr_positive_for_range_bars() -> None:
    """ATR should be > 0 for bars with non-zero high-low range."""
    t = _mk(close=[100.0 + i for i in range(30)])
    out = atr(t, period=14)
    vals = [v for v in out.column("atr_14").to_pylist() if not math.isnan(v)]
    assert all(v > 0 for v in vals)
