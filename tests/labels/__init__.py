"""Tests for the label module's public dispatch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.labels import make_labels
from kairon.labels.schema import LabelKind, LabelSpec


def _mk(n: int = 120) -> pa.Table:
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    close = [100.0 + 0.1 * i for i in range(n)]
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


def test_make_labels_dispatches_to_direction() -> None:
    t = _mk()
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    f = make_labels(t, spec=spec, symbol="BTC-USDT")
    assert f.spec is spec
    assert f.symbol == "BTC-USDT"
    assert len(f.bars) > 0


def test_make_labels_dispatches_to_magnitude() -> None:
    t = _mk()
    spec = LabelSpec(kind=LabelKind.MAGNITUDE, horizon="1h")
    f = make_labels(t, spec=spec, symbol="BTC-USDT")
    assert f.spec is spec
    assert len(f.bars) > 0


def test_make_labels_dispatches_to_volatility() -> None:
    t = _mk()
    spec = LabelSpec(kind=LabelKind.VOLATILITY, horizon="1h")
    f = make_labels(t, spec=spec, symbol="BTC-USDT")
    assert len(f.bars) > 0
