"""Tests for trend indicators: EMA, SMA, MACD, ADX, Ichimoku."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.technical.trend import adx, ema, ichimoku, macd, sma


def _mk(
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
    n: int = 200,
) -> pa.Table:
    cols = [c for c in (open_, high, low, close) if c is not None]
    inferred_n = max((len(c) for c in cols), default=n)
    opn = open_ or [100.0 + 0.1 * i for i in range(inferred_n)]
    h = high or [opn[i] + 1.0 for i in range(inferred_n)]
    lo = low or [opn[i] - 1.0 for i in range(inferred_n)]
    cl = close or [opn[i] + 0.05 * i for i in range(inferred_n)]
    return pa.table(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC) for _ in range(inferred_n)],
            "open": opn,
            "high": h,
            "low": lo,
            "close": cl,
            "volume": [10.0] * inferred_n,
        },
        schema=OHLCV_SCHEMA,
    )


def test_ema_known_value() -> None:
    # On a constant series, EMA equals the constant.
    t = _mk(close=[100.0] * 20)
    out = ema(t, period=5)
    vals = out.column("ema_5_close").to_pylist()
    assert all(abs(v - 100.0) < 1e-9 for v in vals if not math.isnan(v))


def test_ema_first_nan_then_seed() -> None:
    t = _mk(close=[100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    out = ema(t, period=3)
    vals = out.column("ema_3_close").to_pylist()
    assert math.isnan(vals[0])
    assert math.isnan(vals[1])
    # The seed is the SMA of [100,110,120] = 110
    assert abs(vals[2] - 110.0) < 1e-9


def test_ema_rejects_invalid_period() -> None:
    t = _mk(n=10)
    with pytest.raises(ValueError):
        ema(t, period=0)


def test_sma_basic() -> None:
    t = _mk(close=[10.0, 20.0, 30.0, 40.0])
    out = sma(t, period=2, source="close", out="sma_x")
    vals = out.column("sma_x").to_pylist()
    assert math.isnan(vals[0])
    assert vals[1] == 15.0
    assert vals[2] == 25.0
    assert vals[3] == 35.0


def test_macd_columns_present() -> None:
    t = _mk(n=100)
    out = macd(t)
    assert "macd_line" in out.column_names
    assert "macd_signal" in out.column_names
    assert "macd_hist" in out.column_names


def test_macd_rejects_invalid_params() -> None:
    t = _mk(n=100)
    with pytest.raises(ValueError):
        macd(t, fast=30, slow=12, signal=9)


def test_adx_columns_present() -> None:
    t = _mk(n=100)
    out = adx(t)
    assert "adx" in out.column_names
    assert "plus_di" in out.column_names
    assert "minus_di" in out.column_names


def test_ichimoku_columns_present() -> None:
    t = _mk(n=120)
    out = ichimoku(t)
    for col in ("ichi_tenkan", "ichi_kijun", "ichi_senkou_a", "ichi_senkou_b", "ichi_chikou"):
        assert col in out.column_names
