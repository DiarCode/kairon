"""Tests for the research historical-bar parquet store (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from kairon.data.history_store import (
    history_path,
    max_stored_ts,
    merge_history,
    read_history,
    write_history,
)
from kairon.data.io import OHLCV_SCHEMA


def _bars(start: datetime, n: int, step_seconds: int = 60, price: float = 100.0) -> pa.Table:
    ts = [start + timedelta(seconds=step_seconds * i) for i in range(n)]
    return pa.table(
        {
            "ts": ts,
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "close": [price] * n,
            "volume": [1.0] * n,
        },
        schema=OHLCV_SCHEMA,
    )


class TestHistoryStore:
    def test_history_path_layout(self) -> None:
        p = history_path(Path("data"), "ETH-USDT-PERP", "1m")
        assert p == Path("data") / "history" / "ETH-USDT-PERP" / "1m.parquet"

    def test_read_empty_when_no_file(self, tmp_path: Path) -> None:
        table = read_history(tmp_path, "ETH-USDT-PERP", "1m")
        assert table.num_rows == 0
        assert table.schema == OHLCV_SCHEMA

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        bars = _bars(start, 5)
        path = write_history(bars, tmp_path, "ETH-USDT-PERP", "1m")
        assert path.exists()
        out = read_history(tmp_path, "ETH-USDT-PERP", "1m")
        assert out.num_rows == 5
        # Write sorts ascending by ts.
        ts = out.column("ts").to_pylist()
        assert ts == sorted(ts)

    def test_write_rejects_wrong_schema(self, tmp_path: Path) -> None:
        bad = pa.table({"ts": [datetime(2026, 1, 1, tzinfo=UTC)], "open": [1.0]})
        with pytest.raises(ValueError, match="schema mismatch"):
            write_history(bad, tmp_path, "ETH-USDT-PERP", "1m")

    def test_read_time_filter(self, tmp_path: Path) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        write_history(_bars(start, 10), tmp_path, "SOL-USDT-PERP", "1m")
        lo = start + timedelta(minutes=3)
        hi = start + timedelta(minutes=6)
        out = read_history(tmp_path, "SOL-USDT-PERP", "1m", start=lo, end=hi)
        # bars at minutes 3,4,5,6 inclusive
        assert out.num_rows == 4
        assert out.column("ts").to_pylist()[0] == lo
        assert out.column("ts").to_pylist()[-1] == hi

    def test_merge_dedupes_and_appends(self, tmp_path: Path) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        merge_history(_bars(start, 3), tmp_path, "XRP-USDT-PERP", "1m")
        # Re-fetch overlapping + new: bars 2..5 (2 overlaps, 4,5 are new).
        merge_history(
            _bars(start + timedelta(minutes=2), 4),
            tmp_path, "XRP-USDT-PERP", "1m",
        )
        out = read_history(tmp_path, "XRP-USDT-PERP", "1m")
        assert out.num_rows == 6  # minutes 0..5, no duplicates
        ts = out.column("ts").to_pylist()
        assert ts == sorted(ts)
        assert len(set(ts)) == 6

    def test_merge_refreshes_overlapping_bar_values(self, tmp_path: Path) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        merge_history(_bars(start, 1, price=100.0), tmp_path, "BTC-USDT-PERP", "1m")
        # Same timestamp, different price -> the merged row keeps the LAST value.
        merge_history(_bars(start, 1, price=200.0), tmp_path, "BTC-USDT-PERP", "1m")
        out = read_history(tmp_path, "BTC-USDT-PERP", "1m")
        assert out.num_rows == 1
        assert float(out.column("close")[0].as_py()) == 200.0

    def test_max_stored_ts(self, tmp_path: Path) -> None:
        assert max_stored_ts(tmp_path, "ETH-USDT-PERP", "1m") is None
        start = datetime(2026, 1, 1, tzinfo=UTC)
        merge_history(_bars(start, 3), tmp_path, "ETH-USDT-PERP", "1m")
        assert max_stored_ts(tmp_path, "ETH-USDT-PERP", "1m") == start + timedelta(minutes=2)
