"""Data quality diagnostics.

These checks run on every ingestion and are the first line of defense
against silent data corruption, missing bars, and timezone drift.

Every check returns a ``DiagnosticResult`` so the caller can surface a
typed report. The orchestrator in ``ingestion.py`` aggregates results
and emits a single typed summary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pyarrow as pa
import pyarrow.compute as pc  # noqa: F401  — re-exported for callers

from kairon.data.io import OHLCV_SCHEMA


class Severity(str, Enum):
    """Severity of a data quality issue."""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """A single data quality check result."""

    name: str
    severity: Severity
    passed: bool
    message: str
    n_affected: int = 0
    sample: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """An aggregate report from one or more checks."""

    results: tuple[DiagnosticResult, ...]
    symbol: str
    venue: str
    timeframe: str

    @property
    def has_errors(self) -> bool:
        return any(r.severity == Severity.ERROR for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.severity == Severity.WARN for r in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "venue": self.venue,
            "timeframe": self.timeframe,
            "results": [
                {
                    "name": r.name,
                    "severity": r.severity.value,
                    "passed": r.passed,
                    "message": r.message,
                    "n_affected": r.n_affected,
                    "sample": r.sample,
                }
                for r in self.results
            ],
            "summary": {
                "errors": sum(1 for r in self.results if r.severity == Severity.ERROR),
                "warnings": sum(1 for r in self.results if r.severity == Severity.WARN),
                "passed": sum(1 for r in self.results if r.passed),
            },
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_schema(table: pa.Table) -> DiagnosticResult:
    """Verify that the table matches the OHLCV schema exactly."""
    passed = table.schema == OHLCV_SCHEMA
    return DiagnosticResult(
        name="schema_match",
        severity=Severity.ERROR if not passed else Severity.INFO,
        passed=passed,
        message="schema matches OHLCV" if passed else f"schema mismatch: {table.schema}",
    )


def check_monotonic_ts(table: pa.Table) -> DiagnosticResult:
    """Timestamps must be monotonically non-decreasing."""
    if len(table) < 2:
        return DiagnosticResult(
            name="monotonic_ts",
            severity=Severity.INFO,
            passed=True,
            message="< 2 rows; trivially monotonic",
        )
    ts = table.column("ts").to_pylist()
    n_bad = 0
    sample: list[str] = []
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1]:
            n_bad += 1
            if len(sample) < 5:
                sample.append(str(ts[i]))
    return DiagnosticResult(
        name="monotonic_ts",
        severity=Severity.ERROR if n_bad > 0 else Severity.INFO,
        passed=n_bad == 0,
        message=f"{n_bad} non-monotonic timestamps" if n_bad else "timestamps are monotonic",
        n_affected=n_bad,
        sample=sample,
    )


def check_no_duplicate_ts(table: pa.Table) -> DiagnosticResult:
    """Timestamps must be unique within the table."""
    ts = table.column("ts").to_pylist()
    counts: dict[object, int] = {}
    for t in ts:
        counts[t] = counts.get(t, 0) + 1
    dups = {k: v for k, v in counts.items() if v > 1}
    n_dups = sum(v - 1 for v in dups.values())
    sample: list[str] = []
    if dups:
        sample = [str(k) for k in list(dups.keys())[:5]]
    return DiagnosticResult(
        name="no_duplicate_ts",
        severity=Severity.ERROR if n_dups > 0 else Severity.INFO,
        passed=n_dups == 0,
        message=f"{n_dups} duplicate timestamps" if n_dups else "no duplicate timestamps",
        n_affected=n_dups,
        sample=sample,
    )


def check_no_negative_values(table: pa.Table) -> DiagnosticResult:
    """Open/high/low/close/volume must be non-negative."""
    issues: list[str] = []
    n_affected = 0
    for col_name in ("open", "high", "low", "close", "volume"):
        col = table.column(col_name).to_pylist()
        n_neg = sum(1 for v in col if v < 0)
        if n_neg > 0:
            issues.append(f"{col_name}={n_neg}")
            n_affected += n_neg
    passed = not issues
    return DiagnosticResult(
        name="no_negative_values",
        severity=Severity.ERROR if issues else Severity.INFO,
        passed=passed,
        message=("negative values: " + ", ".join(issues)) if issues else "all values non-negative",
        n_affected=n_affected,
    )


def check_high_ge_low(table: pa.Table) -> DiagnosticResult:
    """High must be ≥ Low for every bar."""
    if len(table) == 0:
        return DiagnosticResult(
            name="high_ge_low",
            severity=Severity.INFO,
            passed=True,
            message="empty table",
        )
    high = table.column("high").to_pylist()
    low = table.column("low").to_pylist()
    n_bad = sum(1 for h_val, l_val in zip(high, low, strict=True) if h_val < l_val)
    return DiagnosticResult(
        name="high_ge_low",
        severity=Severity.WARN if n_bad > 0 else Severity.INFO,
        passed=n_bad == 0,
        message=f"{n_bad} bars where high < low" if n_bad else "high >= low for all bars",
        n_affected=n_bad,
    )


def check_timezone_is_utc(table: pa.Table) -> DiagnosticResult:
    """Timestamp column must be UTC tz-aware."""
    ts_type = table.schema.field("ts").type
    tz = getattr(ts_type, "tz", None)
    return DiagnosticResult(
        name="tz_is_utc",
        severity=Severity.ERROR if tz != "UTC" else Severity.INFO,
        passed=tz == "UTC",
        message=f"timestamp tz is {tz!r}" if tz else "timestamp has no timezone",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
DEFAULT_CHECKS = (
    check_schema,
    check_monotonic_ts,
    check_no_duplicate_ts,
    check_no_negative_values,
    check_high_ge_low,
    check_timezone_is_utc,
)


def run_diagnostics(
    table: pa.Table,
    *,
    symbol: str,
    venue: str,
    timeframe: str,
    checks: Sequence = DEFAULT_CHECKS,
) -> DiagnosticReport:
    """Run all (or a subset of) checks and return a typed report."""
    results = tuple(check(table) for check in checks)
    return DiagnosticReport(
        results=results,
        symbol=symbol,
        venue=venue,
        timeframe=timeframe,
    )


def assert_quality(report: DiagnosticReport) -> None:
    """Raise ``ValueError`` if the report has any ERROR-severity results."""
    if report.has_errors:
        msgs = "; ".join(
            f"{r.name}: {r.message}" for r in report.results if r.severity == Severity.ERROR
        )
        raise ValueError(f"data quality errors for {report.symbol}/{report.timeframe}: {msgs}")


# ---------------------------------------------------------------------------
# Timeframe handling
# ---------------------------------------------------------------------------
def timeframe_to_timedelta(tf: str) -> int:
    """Convert a timeframe string (5m, 1h, 1d) to seconds (an int).

    We return seconds as an int to avoid datetime arithmetic inside hot paths.
    """
    table = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }
    if tf not in table:
        raise ValueError(f"unknown timeframe {tf!r}; expected one of {sorted(table)}")
    return table[tf]


def bar_starts_are_aligned(ts: Sequence[datetime], timeframe: str) -> DiagnosticResult:
    """Check that every timestamp is a clean multiple of the timeframe (UTC)."""
    n = len(ts)
    if n < 2:
        return DiagnosticResult(
            name="bar_alignment",
            severity=Severity.INFO,
            passed=True,
            message="< 2 rows",
        )
    seconds = timeframe_to_timedelta(timeframe)
    bad = 0
    sample: list[str] = []
    for t in ts:
        if t.tzinfo is None or t.utcoffset() is None:
            bad += 1
            if len(sample) < 5:
                sample.append(str(t))
            continue
        epoch = int(t.timestamp())
        if epoch % seconds != 0:
            bad += 1
            if len(sample) < 5:
                sample.append(str(t))
    return DiagnosticResult(
        name="bar_alignment",
        severity=Severity.WARN if bad else Severity.INFO,
        passed=bad == 0,
        message=(
            f"{bad} bars not aligned to {timeframe}"
            if bad
            else f"all bars aligned to {timeframe}"
        ),
        n_affected=bad,
        sample=sample,
    )
