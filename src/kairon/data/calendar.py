"""Trading calendar helpers.

We only model the basics: a bar at the start of a timeframe. The exchanges
themselves define their trading hours; for crypto it's 24/7, for US stocks
it's roughly 09:30-16:00 ET on weekdays excluding holidays. Kairon does
not maintain a holiday calendar in v1; we compute expected bar counts
from the time delta and use them only for diagnostics, never for
filtering (the system always operates on observed bars).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from kairon.data.diagnostics import timeframe_to_timedelta


def bar_grid(
    start: datetime, end: datetime, timeframe: str
) -> Iterator[datetime]:
    """Yield bar-start timestamps in ``[start, end)`` at the given timeframe.

    All timestamps are UTC. ``start`` is included only if it is itself
    aligned to the timeframe.
    """
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware (UTC)")
    if end.tzinfo is None:
        raise ValueError("end must be timezone-aware (UTC)")
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    seconds = timeframe_to_timedelta(timeframe)
    step = timedelta(seconds=seconds)
    cur = start_utc
    while cur < end_utc:
        yield cur
        cur = cur + step


def expected_bar_count(start: datetime, end: datetime, timeframe: str) -> int:
    """Number of bars expected in ``[start, end)`` for a 24/7 market.

    For non-24/7 markets (US stocks), the actual count will be lower;
    callers should compute observed vs expected and surface as a warning.
    """
    if end <= start:
        return 0
    delta = (end - start).total_seconds()
    seconds = timeframe_to_timedelta(timeframe)
    return int(delta // seconds)
