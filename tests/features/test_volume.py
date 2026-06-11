"""Tests for volume indicators: OBV, VWAP, CVD."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.technical.volume import cvd, obv, vwap


def _mk(close: list[float], volume: list[float]) -> pa.Table:
    n = len(close)
    opn = [c - 0.5 for c in close]
    h = [c + 1.0 for c in close]
    lo = [c - 1.0 for c in close]
    return pa.table(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC) for _ in range(n)],
            "open": opn,
            "high": h,
            "low": lo,
            "close": close,
            "volume": volume,
        },
        schema=OHLCV_SCHEMA,
    )


def test_obv_rises_on_up_days() -> None:
    t = _mk([100.0, 101.0, 102.0], [10.0, 20.0, 30.0])
    out = obv(t)
    vals = out.column("obv").to_pylist()
    # 0 → +20 → +20+30 = 50
    assert vals == [0.0, 20.0, 50.0]


def test_obv_falls_on_down_days() -> None:
    t = _mk([102.0, 101.0, 100.0], [10.0, 20.0, 30.0])
    out = obv(t)
    vals = out.column("obv").to_pylist()
    assert vals == [0.0, -20.0, -50.0]


def test_vwap_columns_present() -> None:
    t = _mk([100.0, 101.0, 102.0], [10.0, 10.0, 10.0])
    out = vwap(t)
    assert "vwap" in out.column_names
    vals = out.column("vwap").to_pylist()
    # vwap should be a sensible average
    assert all(v > 99.0 and v < 105.0 for v in vals)


def test_cvd_positive_on_uptrend() -> None:
    t = _mk([100.0, 101.0, 102.0], [10.0, 10.0, 10.0])
    out = cvd(t)
    vals = out.column("cvd").to_pylist()
    # Up bars → positive delta → cumulative goes up
    assert vals[-1] > 0
    assert vals[-1] == sum(10.0 for _ in range(3))
