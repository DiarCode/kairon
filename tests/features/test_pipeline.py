"""Tests for the typed feature pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.pipeline import FeaturePipeline


def _mk(n: int = 200) -> pa.Table:
    return pa.table(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC) for _ in range(n)],
            "open": [100.0 + 0.1 * i for i in range(n)],
            "high": [101.0 + 0.1 * i for i in range(n)],
            "low": [99.0 + 0.1 * i for i in range(n)],
            "close": [100.0 + 0.2 * i for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
        },
        schema=OHLCV_SCHEMA,
    )


def test_default_pipeline_runs_and_is_deterministic() -> None:
    t = _mk()
    p = FeaturePipeline()
    r1 = p.run(t)
    r2 = p.run(t)
    assert r1.provenance.input_hash == r2.provenance.input_hash
    assert r1.provenance.pipeline_hash == r2.provenance.pipeline_hash


def test_pipeline_appends_expected_columns() -> None:
    t = _mk()
    p = FeaturePipeline()
    r = p.run(t)
    for col in ("ema_5_close", "ema_50", "macd_line", "adx", "rsi_14", "atr_14", "obv", "vwap"):
        assert col in r.table.column_names, f"missing {col}"


def test_pipeline_rejects_no_ts() -> None:
    t = pa.table({"a": [1.0, 2.0]})
    p = FeaturePipeline(features=["trend.ema_5"])
    import pytest

    with pytest.raises(ValueError, match="ts"):
        p.run(t)


def test_pipeline_with_explicit_features() -> None:
    t = _mk()
    p = FeaturePipeline(features=["trend.ema_5", "momentum.rsi_14"])
    r = p.run(t)
    assert set(r.feature_names) == {"trend.ema_5", "momentum.rsi_14"}
    assert "ema_5_close" in r.table.column_names
    assert "rsi_14" in r.table.column_names


def test_pipeline_hash_changes_with_features() -> None:
    a = FeaturePipeline(features=["trend.ema_5"])
    b = FeaturePipeline(features=["trend.ema_5", "momentum.rsi_14"])
    assert a.pipeline_hash != b.pipeline_hash
