"""Tests for the regime classifier."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.regime import (
    Regime,
    RegimeModel,
    add_regime,
    regime_distribution,
)
from kairon.features.technical.trend import adx
from kairon.features.technical.volatility import atr


def _mk(close: list[float], n_atr_period: int = 14) -> pa.Table:
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
            "volume": [10.0] * n,
        },
        schema=OHLCV_SCHEMA,
    )


def test_regime_is_str_enum() -> None:
    assert Regime.TRENDING.value == "trending"
    assert Regime.RANGING.value == "ranging"
    assert Regime.VOLATILE.value == "volatile"
    assert Regime.STRESSED.value == "stressed"


def test_regime_distribution_counts_all_labels() -> None:
    labels = ["trending", "ranging", "volatile", "trending", "ranging"]
    out = regime_distribution(labels)
    assert out == {"trending": 2, "ranging": 2, "volatile": 1, "stressed": 0}


def test_regime_fit_requires_sufficient_rows() -> None:
    t = _mk([100.0 + i for i in range(20)])  # too few finite rows after adx/atr
    t = adx(atr(t, period=14))
    with pytest.raises(ValueError, match=r"only \d+ finite rows"):
        RegimeModel.fit(t)


def test_regime_fit_returns_model() -> None:
    # Build a synthetic series with regime-like variation: 30 bars up, 30 down, 40 ranging.
    close = (
        [100.0 + 0.5 * i for i in range(30)]
        + [115.0 - 0.5 * i for i in range(30)]
        + [100.0 + 0.05 * (i % 5) for i in range(40)]
    )
    t = _mk(close)
    t = adx(atr(t, period=14))
    m = RegimeModel.fit(t)
    assert len(m.adx_means) == 4
    assert len(m.atr_z_means) == 4
    assert all(w > 0 for w in m.weights)
    assert abs(sum(m.weights) - 1.0) < 1e-6


def test_add_regime_appends_column() -> None:
    close = (
        [100.0 + 0.5 * i for i in range(30)]
        + [115.0 - 0.5 * i for i in range(30)]
        + [100.0 + 0.05 * (i % 5) for i in range(40)]
    )
    t = _mk(close)
    out = add_regime(t)
    assert "regime" in out.column_names
    labels = out.column("regime").to_pylist()
    assert all(lbl in {r.value for r in Regime} for lbl in labels if lbl is not None)


def test_predict_one_stress_override() -> None:
    close = (
        [100.0 + 0.5 * i for i in range(30)]
        + [115.0 - 0.5 * i for i in range(30)]
        + [100.0 + 0.05 * (i % 5) for i in range(40)]
    )
    t = _mk(close)
    t = adx(atr(t, period=14))
    m = RegimeModel.fit(t)
    assert m.predict_one(20.0, 10.0) == Regime.STRESSED
