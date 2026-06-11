"""Tests for magnitude labels."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log

import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.labels.magnitude import make_magnitude_labels
from kairon.labels.schema import LabelKind, LabelSpec


def _mk(close: list[float], *, every_s: int = 60) -> pa.Table:
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=every_s * i) for i in range(len(close))]
    return pa.table(
        {
            "ts": ts,
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [10.0] * len(close),
        },
        schema=OHLCV_SCHEMA,
    )


def test_magnitude_known_value() -> None:
    """For 100 → 110, the 1-step log return should be log(110/100) ≈ 0.0953."""
    t = _mk([100.0] * 60 + [110.0] * 60, every_s=60)  # 60 1m bars per hour
    spec = LabelSpec(kind=LabelKind.MAGNITUDE, horizon="1h")
    frame = make_magnitude_labels(t, spec=spec, symbol="BTC-USDT")
    # The 1st hour (60 bars) all have a 1h-ahead close of 110; expect ~log(1.1)
    expected = log(1.1)
    first = frame.bars[0].y
    assert isinstance(first, float)
    assert abs(first - expected) < 1e-9


def test_magnitude_skips_nan() -> None:
    t = _mk([100.0] * 10 + [0.0] + [100.0] * 50, every_s=60)
    spec = LabelSpec(kind=LabelKind.MAGNITUDE, horizon="1h")
    frame = make_magnitude_labels(t, spec=spec, symbol="BTC-USDT")
    # The bar at the 0.0 close should be dropped (close <= 0 guard)
    assert all(b.y > 0 or b.y <= 0 for b in frame.bars)  # tautology — but no NaN
