"""Tests for data quality diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from kairon.data.diagnostics import (
    DEFAULT_CHECKS,
    DiagnosticReport,
    Severity,
    assert_quality,
    bar_starts_are_aligned,
    check_high_ge_low,
    check_monotonic_ts,
    check_no_duplicate_ts,
    check_no_negative_values,
    check_schema,
    check_timezone_is_utc,
    run_diagnostics,
    timeframe_to_timedelta,
)
from kairon.data.io import OHLCV_SCHEMA


def _mk(
    ts: list[datetime] | None = None,
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
    volume: list[float] | None = None,
) -> pa.Table:
    # Determine n: explicit ts wins; otherwise the longest list column;
    # otherwise default 1.
    if ts is not None:
        n = len(ts)
    else:
        cols = [c for c in (open_, high, low, close, volume) if c is not None]
        n = max((len(c) for c in cols), default=1)
    base_ts = ts or [datetime(2024, 1, 1, tzinfo=UTC)] * n
    return pa.table(
        {
            "ts": base_ts,
            "open": open_ or [100.0] * n,
            "high": high or [101.0] * n,
            "low": low or [99.0] * n,
            "close": close or [100.5] * n,
            "volume": volume or [10.0] * n,
        },
        schema=OHLCV_SCHEMA,
    )


def test_timeframe_to_timedelta() -> None:
    assert timeframe_to_timedelta("1m") == 60
    assert timeframe_to_timedelta("5m") == 300
    assert timeframe_to_timedelta("1h") == 3600
    assert timeframe_to_timedelta("1d") == 86400
    with pytest.raises(ValueError):
        timeframe_to_timedelta("bogus")


def test_check_schema_passes() -> None:
    t = _mk()
    r = check_schema(t)
    assert r.passed is True
    assert r.severity == Severity.INFO


def test_check_schema_fails_on_mismatch() -> None:
    bad = pa.table({"a": [1]})
    r = check_schema(bad)
    assert r.passed is False
    assert r.severity == Severity.ERROR


def test_check_monotonic_ts_passes() -> None:
    ts = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
    ]
    r = check_monotonic_ts(_mk(ts=ts))
    assert r.passed is True
    assert r.n_affected == 0


def test_check_monotonic_ts_detects_regression() -> None:
    ts = [
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
    ]
    r = check_monotonic_ts(_mk(ts=ts))
    assert r.passed is False
    assert r.severity == Severity.ERROR
    assert r.n_affected == 1


def test_check_no_duplicate_ts() -> None:
    ts = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
    ]
    r = check_no_duplicate_ts(_mk(ts=ts))
    assert r.passed is False
    assert r.n_affected == 1


def test_check_no_negative_values() -> None:
    t = _mk(volume=[-1.0, 10.0, 20.0])
    r = check_no_negative_values(t)
    assert r.passed is False
    assert r.n_affected == 1


def test_check_high_ge_low() -> None:
    t = _mk(high=[100.0, 95.0], low=[99.0, 99.0])
    r = check_high_ge_low(t)
    assert r.passed is False
    assert r.n_affected == 1
    assert r.severity == Severity.WARN


def test_check_timezone_is_utc() -> None:
    t = _mk()
    r = check_timezone_is_utc(t)
    assert r.passed is True
    assert r.severity == Severity.INFO


def test_run_diagnostics_aggregates() -> None:
    t = _mk()
    report = run_diagnostics(t, symbol="BTC-USDT", venue="ccxt", timeframe="5m")
    assert isinstance(report, DiagnosticReport)
    assert all(r.passed for r in report.results)
    assert not report.has_errors
    assert not report.has_warnings


def test_assert_quality_raises_on_error() -> None:
    t = _mk(volume=[-1.0])
    report = run_diagnostics(t, symbol="BTC-USDT", venue="ccxt", timeframe="5m")
    with pytest.raises(ValueError, match="data quality errors"):
        assert_quality(report)


def test_assert_quality_passes_on_clean() -> None:
    t = _mk()
    report = run_diagnostics(t, symbol="BTC-USDT", venue="ccxt", timeframe="5m")
    assert_quality(report)  # should not raise


def test_bar_alignment_pass() -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    ts = [base + timedelta(minutes=5 * i) for i in range(3)]
    r = bar_starts_are_aligned(ts, "5m")
    assert r.passed is True


def test_bar_alignment_fail() -> None:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    ts = [base, base + timedelta(minutes=3)]  # 3m is not 5m
    r = bar_starts_are_aligned(ts, "5m")
    assert r.passed is False
    assert r.n_affected == 1


def test_default_checks_is_tuple() -> None:
    assert isinstance(DEFAULT_CHECKS, tuple)
    assert len(DEFAULT_CHECKS) >= 5
