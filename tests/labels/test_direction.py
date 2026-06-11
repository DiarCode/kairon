"""Tests for direction labels + critical leakage invariant."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from kairon.data.io import OHLCV_SCHEMA
from kairon.labels.direction import make_direction_labels
from kairon.labels.schema import LabelKind, LabelSpec


def _mk(close: list[float], *, start: datetime | None = None, every_s: int = 60) -> pa.Table:
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
    n = len(close)
    ts = [start + timedelta(seconds=every_s * i) for i in range(n)]
    return pa.table(
        {
            "ts": ts,
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [10.0] * n,
        },
        schema=OHLCV_SCHEMA,
    )


def test_uptrend_labels_all_up() -> None:
    """Strictly increasing series should produce all UP labels."""
    # 120 bars at 1-minute intervals; 1h horizon = 60 bars; we have 60 future bars
    t = _mk([100.0 + i for i in range(120)], every_s=60)
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    frame = make_direction_labels(t, spec=spec, symbol="BTC-USDT")
    ys = [b.y_class for b in frame.bars]
    up_frac = sum(1 for y in ys if y == 1) / max(len(ys), 1)
    assert up_frac > 0.9


def test_downtrend_labels_all_down() -> None:
    t = _mk([200.0 - i for i in range(120)], every_s=60)
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    frame = make_direction_labels(t, spec=spec, symbol="BTC-USDT")
    ys = [b.y_class for b in frame.bars]
    down_frac = sum(1 for y in ys if y == -1) / max(len(ys), 1)
    assert down_frac > 0.9


def test_flat_thresholds() -> None:
    """Bars that move less than the flat threshold should be FLAT (0)."""
    # 50 bars of 100.0, then 50 bars of 100.001 — tiny move
    t = _mk([100.0] * 25 + [100.001] * 25, every_s=60)
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    frame = make_direction_labels(t, spec=spec, symbol="BTC-USDT", flat_threshold_pct=0.01)
    ys = [b.y_class for b in frame.bars]
    # All should be FLAT
    assert all(y == 0 for y in ys)


def test_no_future_close_label() -> None:
    """Bars whose horizon goes past the end of the table are dropped."""
    t = _mk([100.0] * 10, every_s=60)
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    frame = make_direction_labels(t, spec=spec, symbol="BTC-USDT")
    # 10 bars at 1-minute intervals; 1h = 60 bars ahead; none has a future close
    assert len(frame.bars) == 0


def test_rejects_unsorted_ts() -> None:
    """Make sure the function refuses unsorted timestamps (would let us
    accidentally use a future close as a feature)."""
    ts = [datetime(2024, 1, 1, 0, 0, tzinfo=UTC), datetime(2024, 1, 1, 0, 5, tzinfo=UTC)]
    ts = ts[::-1]  # reverse
    t = pa.table(
        {
            "ts": ts,
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [10.0, 10.0],
        },
        schema=OHLCV_SCHEMA,
    )
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    with pytest.raises(ValueError, match="sorted"):
        make_direction_labels(t, spec=spec, symbol="BTC-USDT")


def test_rejects_wrong_kind() -> None:
    t = _mk([100.0] * 5, every_s=60)
    spec = LabelSpec(kind=LabelKind.MAGNITUDE, horizon="1h")
    with pytest.raises(ValueError, match="DIRECTION"):
        make_direction_labels(t, spec=spec, symbol="BTC-USDT")
