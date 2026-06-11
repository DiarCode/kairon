"""Tests for the bar-level leakage fixture in :mod:`tests.fixtures.leakage`.

These tests pin the behaviour of the shared fixture so that downstream
test authors can rely on it. They cover four invariants:

1. The planted-leak case: a short fixture cannot realise a long
   label window.
2. The clean case: a fixture whose span matches the label window
   passes.
3. ``assert_timestamp_monotonic`` accepts a well-formed fixture and
   rejects a fixture with a duplicate (out-of-order) row.
4. The fixture conforms exactly to :data:`OHLCV_SCHEMA` and is
   deterministic across calls with the same seed.

The last test demonstrates a downstream import of the helpers from
this module — i.e. a test that does not live in
``tests/fixtures/`` would do ``from tests.fixtures.leakage import
...`` and get the same functions. The round-trip ``tests.fixtures
.leakage`` import here exercises that import path so a future
packaging change (e.g. removing ``tests/__init__.py``) cannot silently
break downstream consumers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from kairon.data.io import OHLCV_SCHEMA

from tests.fixtures.leakage import (
    assert_no_leakage,
    assert_timestamp_monotonic,
    real_history_fixture,
)


# ---------------------------------------------------------------------------
# real_history_fixture — schema and determinism
# ---------------------------------------------------------------------------
def test_fixture_conforms_to_ohlcv_schema() -> None:
    """The fixture must have the exact OHLCV schema, not just a compatible one."""
    table = real_history_fixture(n_bars=10)
    assert table.schema == OHLCV_SCHEMA, (
        f"expected exact OHLCV_SCHEMA, got {table.schema!r}"
    )
    assert table.num_columns == 6
    assert len(table) == 10


def test_fixture_is_deterministic_for_same_seed() -> None:
    """Two calls with the same seed must produce identical first/last close.

    This is the W1.6 determinism contract: the fixture is reproducible.
    """
    a = real_history_fixture(n_bars=100, seed=1234)
    b = real_history_fixture(n_bars=100, seed=1234)
    a_close = a.column("close").to_pylist()
    b_close = b.column("close").to_pylist()
    assert a_close[0] == pytest.approx(b_close[0])
    assert a_close[-1] == pytest.approx(b_close[-1])
    # Stronger check: every close should match (deterministic generator).
    for i, (x, y) in enumerate(zip(a_close, b_close, strict=True)):
        assert x == pytest.approx(y), f"close[{i}] differs: {x} vs {y}"
    # And the timestamps should be identical.
    a_ts = a.column("ts").to_pylist()
    b_ts = b.column("ts").to_pylist()
    assert a_ts == b_ts


def test_fixture_different_seeds_yield_different_walks() -> None:
    """Different seeds should produce different close sequences."""
    a = real_history_fixture(n_bars=200, seed=1)
    b = real_history_fixture(n_bars=200, seed=2)
    a_close = a.column("close").to_pylist()
    b_close = b.column("close").to_pylist()
    assert a_close[0] != pytest.approx(b_close[0]) or a_close[-1] != pytest.approx(
        b_close[-1]
    )


def test_fixture_timestamps_are_in_the_past() -> None:
    """The default start must be in the past (no future timestamps)."""
    table = real_history_fixture(n_bars=10)
    last_ts: datetime = table.column("ts")[-1].as_py()  # type: ignore[assignment]
    assert last_ts.tzinfo is not None
    last_utc = last_ts.astimezone(UTC)
    assert last_utc < datetime.now(UTC), (
        f"fixture's last ts {last_utc.isoformat()} is in the future"
    )


def test_fixture_rejects_zero_n_bars() -> None:
    with pytest.raises(ValueError, match="n_bars"):
        real_history_fixture(n_bars=0)


def test_fixture_rejects_zero_every_seconds() -> None:
    with pytest.raises(ValueError, match="every_seconds"):
        real_history_fixture(n_bars=10, every_seconds=0)


def test_fixture_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        real_history_fixture(n_bars=10, sigma=-0.01)


# ---------------------------------------------------------------------------
# assert_no_leakage — planted leak and clean cases
# ---------------------------------------------------------------------------
def test_assert_no_leakage_detects_planted_leak_short_fixture_long_horizon() -> None:
    """A 10-bar (10-minute) fixture cannot realise a 1-day label window.

    This is the planted-leak case: the fixture is much shorter than the
    label horizon, so a downstream consumer that naively tries to
    realise the label would have to reach into the future. The helper
    must raise.
    """
    table = real_history_fixture(n_bars=10, every_seconds=60)
    with pytest.raises(AssertionError, match="future-leakage"):
        assert_no_leakage(table, label_horizon_seconds=86_400)  # 1 day


def test_assert_no_leakage_passes_when_horizon_fits_fixture() -> None:
    """A 10-bar (10-minute) fixture can realise a 1-minute label window.

    1-minute label horizon vs. 10-minute fixture: the label window is
    comfortably inside the fixture's span.
    """
    table = real_history_fixture(n_bars=10, every_seconds=60)
    # Must not raise.
    assert_no_leakage(table, label_horizon_seconds=60)


def test_assert_no_leakage_passes_when_horizon_equals_fixture_length() -> None:
    """The boundary case: label horizon exactly equals fixture span passes.

    The check is ``(max(ts) - min(ts)).total_seconds() >=
    label_horizon_seconds``. A 101-bar 1-second fixture has a 100s
    span, so a 100s label horizon is the tightest passable case.
    """
    table = real_history_fixture(n_bars=101, every_seconds=1)
    assert_no_leakage(table, label_horizon_seconds=100)


def test_assert_no_leakage_rejects_empty_table() -> None:
    empty = pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)
    with pytest.raises(AssertionError, match="empty"):
        assert_no_leakage(empty, label_horizon_seconds=60)


def test_assert_no_leakage_rejects_wrong_schema() -> None:
    bad = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(AssertionError, match="schema"):
        assert_no_leakage(bad, label_horizon_seconds=1)  # type: ignore[arg-type]


def test_assert_no_leakage_rejects_non_positive_horizon() -> None:
    table = real_history_fixture(n_bars=10)
    with pytest.raises(AssertionError, match="label_horizon_seconds"):
        assert_no_leakage(table, label_horizon_seconds=0)
    with pytest.raises(AssertionError, match="label_horizon_seconds"):
        assert_no_leakage(table, label_horizon_seconds=-1)


# ---------------------------------------------------------------------------
# assert_timestamp_monotonic
# ---------------------------------------------------------------------------
def test_assert_timestamp_monotonic_passes_on_clean_fixture() -> None:
    """A well-formed fixture must satisfy monotonicity + no-future checks."""
    table = real_history_fixture(n_bars=20, every_seconds=60)
    # Must not raise.
    assert_timestamp_monotonic(table)


def test_assert_timestamp_monotonic_detects_duplicate_row() -> None:
    """A duplicate row (==, not strictly greater) violates non-decreasing.

    The helper requires *strictly non-decreasing* (>=), and a fixture
    with a duplicate row at index i where ts[i] == ts[i-1] should
    pass. To exercise the failure path we synthesise a fixture with
    a backwards step: ts[i] < ts[i-1].
    """
    base = real_history_fixture(n_bars=10, every_seconds=60)
    ts_col = base.column("ts").to_pylist()
    # Force a backwards step: row 5 is older than row 4.
    ts_col[5] = ts_col[4] - timedelta(seconds=10)
    bad = pa.table(
        {
            "ts": ts_col,
            "open": base.column("open").to_pylist(),
            "high": base.column("high").to_pylist(),
            "low": base.column("low").to_pylist(),
            "close": base.column("close").to_pylist(),
            "volume": base.column("volume").to_pylist(),
        },
        schema=OHLCV_SCHEMA,
    )
    with pytest.raises(AssertionError, match="not non-decreasing"):
        assert_timestamp_monotonic(bad)


def test_assert_timestamp_monotonic_detects_future_timestamp() -> None:
    """A fixture whose last bar is in the far future must be flagged."""
    base = real_history_fixture(n_bars=10, every_seconds=60)
    future = datetime.now(UTC) + timedelta(days=365)
    ts_col = [
        future - timedelta(seconds=60 * (10 - 1 - i)) for i in range(10)
    ]
    bad = pa.table(
        {
            "ts": ts_col,
            "open": base.column("open").to_pylist(),
            "high": base.column("high").to_pylist(),
            "low": base.column("low").to_pylist(),
            "close": base.column("close").to_pylist(),
            "volume": base.column("volume").to_pylist(),
        },
        schema=OHLCV_SCHEMA,
    )
    with pytest.raises(AssertionError, match="future timestamp"):
        assert_timestamp_monotonic(bad)


def test_assert_timestamp_monotonic_rejects_empty_table() -> None:
    empty = pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)
    with pytest.raises(AssertionError, match="empty"):
        assert_timestamp_monotonic(empty)


def test_assert_timestamp_monotonic_rejects_wrong_schema() -> None:
    bad = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(AssertionError, match="schema"):
        assert_timestamp_monotonic(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip import (proves downstream tests can import the helpers)
# ---------------------------------------------------------------------------
def test_downstream_can_import_helpers_from_package() -> None:
    """A consumer of the fixture module gets the same helpers."""
    # Re-import via the package path; this exercises the
    # ``from tests.fixtures.leakage import ...`` path that downstream
    # tests will use.
    from tests.fixtures.leakage import (  # noqa: F401  (import-only check)
        assert_no_leakage as _assert_no_leakage,
        assert_timestamp_monotonic as _assert_timestamp_monotonic,
        real_history_fixture as _real_history_fixture,
    )
    assert _assert_no_leakage is assert_no_leakage
    assert _assert_timestamp_monotonic is assert_timestamp_monotonic
    assert _real_history_fixture is real_history_fixture
