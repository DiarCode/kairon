"""Parquet read/write for the research historical-bar store.

Layout (per the scalping-edge-enhancement plan):
    ``data/history/<symbol_canonical>/<timeframe>.parquet``

A single parquet file per (symbol, timeframe) holds the full 8-week research
window. This is deliberately simpler than the monthly-partitioned production
OHLCV store in :mod:`kairon.data.io` — the research store is small (8wk x 1m
≈ 80k rows/symbol) and is read whole by the backtest harness, so one file is
both simpler and faster to scan.

All tables use :data:`kairon.data.io.OHLCV_SCHEMA`. Writes are idempotent and
incremental: :func:`merge_history` reads any existing file, concatenates the new
bars, deduplicates on ``ts``, sorts ascending, and overwrites — so re-running a
fetch only appends bars that arrived since the last stored timestamp.

TESTNET-only label: bars fetched from the Bybit testnet endpoint are stored
here; mainnet microstructure differs, so do not mix mainnet bars into this
store.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kairon.data.io import OHLCV_SCHEMA

__all__ = [
    "dedup_last_per_ts",
    "history_path",
    "max_stored_ts",
    "merge_history",
    "read_history",
    "write_history",
]


def history_path(root: Path, symbol: str, timeframe: str) -> Path:
    """Parquet path for one (symbol, timeframe) research history file."""
    return Path(root) / "history" / symbol / f"{timeframe}.parquet"


def _empty_table() -> pa.Table:
    return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)


def dedup_last_per_ts(table: pa.Table) -> pa.Table:
    """Drop duplicate ``ts`` rows, keeping the LAST occurrence of each.

    Last-write-wins (a dict keyed by ts, final index recorded) is robust to
    sort instability: when ``table`` is the concatenation ``[existing, new]``,
    the new row's value supersedes the existing one for a shared timestamp. The
    result is sorted ascending by ``ts``. Pure-Python over the ts column is O(n)
    and only runs at fetch/merge time (not per tick), so 80k-row histories are
    fine. Returns the table unchanged (but sorted) when there are no duplicates.
    """
    if table.num_rows == 0:
        return table
    ts_py = table.column("ts").to_pylist()
    last_index: dict[object, int] = {}
    for i, t in enumerate(ts_py):
        last_index[t] = i
    if len(last_index) == table.num_rows:
        # No duplicates; still guarantee ascending order for callers.
        return table.sort_by([("ts", "ascending")])
    keep = sorted(last_index.values())
    return table.take(keep).sort_by([("ts", "ascending")])


def read_history(
    root: Path,
    symbol: str,
    timeframe: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pa.Table:
    """Read a symbol's history parquet, optionally filtered to ``[start, end]``.

    Returns an empty OHLCV-schema table when no file exists. The time filter is
    applied in-process (the single-file store is small enough that pushing the
    predicate into parquet is not worth the complexity).
    """
    path = history_path(root, symbol, timeframe)
    if not path.exists():
        return _empty_table()
    table = pq.read_table(path, schema=OHLCV_SCHEMA)
    if start is None and end is None:
        return table
    ts = table.column("ts")
    mask = None
    if start is not None:
        mask = pa.compute.greater_equal(ts, pa.scalar(start, type=ts.type))
    if end is not None:
        end_mask = pa.compute.less_equal(ts, pa.scalar(end, type=ts.type))
        mask = end_mask if mask is None else pa.compute.and_(mask, end_mask)
    return table.filter(mask)


def write_history(table: pa.Table, root: Path, symbol: str, timeframe: str) -> Path:
    """Overwrite a symbol's history parquet with ``table`` (full replace).

    The caller is responsible for merging with existing bars (see
    :func:`merge_history`). Validates the schema so a malformed write cannot
    corrupt the store.
    """
    if table.schema != OHLCV_SCHEMA:
        msg = f"OHLCV schema mismatch: expected {OHLCV_SCHEMA}, got {table.schema}"
        raise ValueError(msg)
    path = history_path(root, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by ts ascending so downstream readers (and the fidelity gate) can
    # assume chronological order without re-sorting.
    table = table.sort_by("ts")
    pq.write_table(table, path)
    return path


def merge_history(
    new_table: pa.Table, root: Path, symbol: str, timeframe: str
) -> pa.Table:
    """Merge ``new_table`` into the stored history (idempotent + incremental).

    Reads any existing file, concatenates, deduplicates on ``ts`` (keeping the
    last occurrence so a re-fetch refreshes a bar's values), sorts ascending,
    and writes the result back. Returns the merged table.
    """
    existing = read_history(root, symbol, timeframe)
    if existing.num_rows == 0 and new_table.num_rows == 0:
        return _empty_table()
    combined = pa.concat_tables([existing, new_table], promote_options="none") \
        if existing.num_rows and new_table.num_rows else (existing or new_table)
    # Deduplicate on ts, keeping the LAST value for each timestamp so a re-fetch
    # of an already-stored bar refreshes it rather than duplicating the row.
    deduped = dedup_last_per_ts(combined)
    write_history(deduped, root, symbol, timeframe)
    return deduped


def max_stored_ts(root: Path, symbol: str, timeframe: str) -> datetime | None:
    """The most recent bar timestamp in the stored history, or None if empty."""
    table = read_history(root, symbol, timeframe)
    if table.num_rows == 0:
        return None
    return pa.compute.max(table.column("ts")).as_py()
