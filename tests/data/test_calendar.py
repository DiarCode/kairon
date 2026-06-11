"""Tests for the calendar helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kairon.data.calendar import bar_grid, expected_bar_count
from kairon.data.diagnostics import timeframe_to_timedelta


def test_bar_grid_yields_aligned_starts() -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 1, 0, 30, tzinfo=UTC)
    bars = list(bar_grid(start, end, "5m"))
    # 30 minutes / 5 min = 6 bars
    assert len(bars) == 6
    assert bars[0] == start
    assert bars[-1] == datetime(2024, 1, 1, 0, 25, tzinfo=UTC)


def test_bar_grid_rejects_naive() -> None:
    with pytest.raises(ValueError):
        list(bar_grid(  # type: ignore[arg-type]
            datetime(2024, 1, 1),
            datetime(2024, 1, 1, 0, 30),
            "5m",
        ))


def test_expected_bar_count() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, 1, tzinfo=UTC)
    assert expected_bar_count(start, end, "5m") == 12
    assert expected_bar_count(start, end, "1h") == 1
    assert expected_bar_count(start, start, "1h") == 0
    assert expected_bar_count(end, start, "1h") == 0  # end <= start


def test_timeframe_consistency() -> None:
    # every timeframe must be a positive int
    for tf in ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"):
        assert timeframe_to_timedelta(tf) > 0
