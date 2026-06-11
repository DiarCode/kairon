"""Tests for volatility and triple-barrier labels."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from kairon.data.io import OHLCV_SCHEMA
from kairon.labels.schema import DirectionClass, LabelKind, LabelSpec
from kairon.labels.triple_barrier import make_triple_barrier_labels
from kairon.labels.volatility import make_volatility_labels


def _mk(close: list[float], high: list[float] | None = None, low: list[float] | None = None, every_s: int = 60) -> pa.Table:
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=every_s * i) for i in range(len(close))]
    h = high or [c + 0.5 for c in close]
    lo = low or [c - 0.5 for c in close]
    return pa.table(
        {
            "ts": ts,
            "open": close,
            "high": h,
            "low": lo,
            "close": close,
            "volume": [10.0] * len(close),
        },
        schema=OHLCV_SCHEMA,
    )


def test_volatility_positive_for_oscillating_series() -> None:
    # oscillating close → realized vol > 0
    t = _mk([100.0 + (i % 5) for i in range(120)], every_s=60)
    spec = LabelSpec(kind=LabelKind.VOLATILITY, horizon="1h")
    frame = make_volatility_labels(t, spec=spec, symbol="BTC-USDT")
    assert len(frame.bars) > 0
    assert all(b.y > 0 for b in frame.bars)


def test_volatility_skips_short_windows() -> None:
    t = _mk([100.0] * 50, every_s=60)
    spec = LabelSpec(kind=LabelKind.VOLATILITY, horizon="30m")
    frame = make_volatility_labels(t, spec=spec, symbol="BTC-USDT")
    # With 50 1-min bars and 30-min horizon, 48 bars have a window of
    # >= 3 bars; the last 2 (bars 48, 49) have only 2 and 1 future bars
    # and are dropped.
    assert len(frame.bars) == 48


def test_triple_barrier_upper_hit() -> None:
    # 60 bars at 1m; 30-minute horizon; close jumps above the upper barrier
    t = _mk([100.0] * 5 + [110.0] * 55, every_s=60)
    spec = LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon="30m")
    frame = make_triple_barrier_labels(t, spec=spec, symbol="BTC-USDT", pt_pct=0.05, sl_pct=0.05)
    # First bar's barrier = 105; the high of bar 5 is 110.5 → hit upper
    assert frame.bars[0].y_class == int(DirectionClass.UP)


def test_triple_barrier_lower_hit() -> None:
    t = _mk([100.0] * 5 + [90.0] * 55, every_s=60)
    spec = LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon="30m")
    frame = make_triple_barrier_labels(t, spec=spec, symbol="BTC-USDT", pt_pct=0.05, sl_pct=0.05)
    assert frame.bars[0].y_class == int(DirectionClass.DOWN)


def test_triple_barrier_vertical_default_dropped() -> None:
    """If require_finite=True (default) and no barrier is hit, drop the bar."""
    t = _mk([100.0] * 120, every_s=60)
    spec = LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon="30m")
    frame = make_triple_barrier_labels(t, spec=spec, symbol="BTC-USDT", pt_pct=0.05, sl_pct=0.05)
    assert len(frame.bars) == 0  # all dropped because no barrier hit


def test_triple_barrier_vertical_kept_when_not_require_finite() -> None:
    t = _mk([100.0] * 120, every_s=60)
    spec = LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon="30m")
    frame = make_triple_barrier_labels(
        t, spec=spec, symbol="BTC-USDT", pt_pct=0.05, sl_pct=0.05, require_finite=False
    )
    assert all(b.y_class == int(DirectionClass.FLAT) for b in frame.bars)


def test_triple_barrier_rejects_invalid_params() -> None:
    t = _mk([100.0] * 10, every_s=60)
    spec = LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon="1h")
    with pytest.raises(ValueError):
        make_triple_barrier_labels(t, spec=spec, symbol="BTC-USDT", pt_pct=0.0, sl_pct=0.01)
    with pytest.raises(ValueError):
        make_triple_barrier_labels(t, spec=spec, symbol="BTC-USDT", pt_pct=0.01, sl_pct=-0.1)
