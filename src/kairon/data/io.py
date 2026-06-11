"""Parquet and DuckDB IO with explicit schemas.

The IO layer is intentionally minimal: read/write typed frames with
explicit schemas, compute content hashes for reproducibility, and never
auto-coerce types. The goal is that ``hash(IO.read(path)) ==
hash(IO.read(path))`` and that any coercion error fires loudly.

The default directory layout is::

    data/
      raw/
        ohlcv/{venue}/{canonical}/{timeframe}/{YYYY}/{MM}.parquet
      processed/
        features/{canonical}/{timeframe}/{YYYY}.parquet
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from kairon.data.symbols import Symbol

# ---------------------------------------------------------------------------
# Default schemas (typed via pyarrow)
# ---------------------------------------------------------------------------
OHLCV_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        ("ts", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
    ]
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DataPaths:
    """Filesystem layout for raw and processed data.

    The class is frozen + slotted for immutability and to satisfy
    pyright strict mode (slots give better attribute types).
    """

    root: Path

    @classmethod
    def default(cls) -> DataPaths:
        """Return the default ``./data`` path under the cwd."""
        return cls(Path(os.environ.get("KAIRON_DATA_ROOT", "data")).resolve())

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    def ohlcv_path(
        self,
        symbol: Symbol,
        venue: str,
        timeframe: str,
        ts: datetime,
    ) -> Path:
        """Return the canonical parquet path for an OHLCV bar's month partition."""
        if ts.tzinfo is None:
            raise ValueError("ts must be timezone-aware (UTC)")
        ts_utc = ts.astimezone(UTC)
        return (
            self.raw
            / "ohlcv"
            / venue
            / symbol.canonical
            / timeframe
            / f"{ts_utc.year:04d}"
            / f"{ts_utc.month:02d}.parquet"
        )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def file_hash(path: Path, *, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Compute a content hash of a file."""
    h = hashlib.new(algo)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def table_hash(table: pa.Table, *, algo: str = "sha256") -> str:
    """Compute a content hash of a pyarrow Table via its IPC bytes."""
    h = hashlib.new(algo)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    h.update(sink.getvalue().to_pybytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------
def write_ohlcv(
    table: pa.Table,
    *,
    symbol: Symbol,
    venue: str,
    timeframe: str,
    paths: DataPaths | None = None,
) -> list[Path]:
    """Write an OHLCV table to per-month parquet files.

    Returns the list of written paths.
    """
    if table.schema != OHLCV_SCHEMA:
        raise ValueError(
            f"OHLCV schema mismatch: expected {OHLCV_SCHEMA}, got {table.schema}"
        )
    if len(table) == 0:
        return []
    p = paths or DataPaths.default()
    ts_col = table.column("ts").to_pylist()
    months: dict[tuple[int, int], pa.RecordBatch] = {}
    for batch in table.to_batches():
        for i in range(batch.num_rows):
            ts = batch.column("ts")[i].as_py()
            assert isinstance(ts, datetime)
            key = (ts.year, ts.month)
            months.setdefault(key, []).append(batch.slice(i, 1))
    written: list[Path] = []
    for (year, month), batches in months.items():
        out = p.ohlcv_path(
            symbol, venue, timeframe, datetime(year, month, 1, tzinfo=UTC)
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        if batches:
            new_table = pa.Table.from_batches(batches, schema=OHLCV_SCHEMA)
        else:
            new_table = pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)
        if out.exists():
            existing = pq.read_table(out)
            new_table = pa.concat_tables([existing, new_table], promote_options="default")
        pq.write_table(new_table, out, compression="snappy")
        written.append(out)
        _ = ts_col  # satisfy type checker
    return written


def read_ohlcv(
    *,
    symbol: Symbol,
    venue: str,
    timeframe: str,
    paths: DataPaths | None = None,
    year: int | None = None,
    month: int | None = None,
) -> pa.Table:
    """Read OHLCV parquet; optionally filter to a single month."""
    p = paths or DataPaths.default()
    base = p.raw / "ohlcv" / venue / symbol.canonical / timeframe
    if not base.exists():
        return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)
    if year is not None and month is not None:
        files = [base / f"{year:04d}" / f"{month:02d}.parquet"]
    elif year is not None:
        files = sorted((base / f"{year:04d}").glob("*.parquet"))
    else:
        files = sorted(base.glob("*/*.parquet"))
    files = [f for f in files if f.is_file()]
    if not files:
        return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)
    return pq.read_table(files, schema=OHLCV_SCHEMA)


# ---------------------------------------------------------------------------
# Plan-style partitioned writer (W1.5)
# ---------------------------------------------------------------------------
def _month_partition_path(
    *, venue: str, symbol: str, timeframe: str, year: int, month: int
) -> Path:
    """Return the per-month parquet file path for the plan layout.

    Layout (hive-style for DuckDB predicate pushdown)::

        {base}/{venue}/{symbol}/{timeframe}/year={yyyy}/month={mm}/data.parquet

    Each partition lives inside a directory whose name follows the hive
    partitioning convention (``key=value``), so that
    ``read_parquet(..., hive_partitioning=true)`` can prune partitions
    when filtered on ``year``/``month``.
    """
    return (
        Path(venue)
        / symbol
        / timeframe
        / f"year={year:04d}"
        / f"month={month:02d}"
        / "data.parquet"
    )


def write_partitioned(
    table: pa.Table,
    *,
    venue: str,
    symbol: str,
    timeframe: str,
    base_path: Path,
) -> list[Path]:
    """Write an OHLCV table to per-month parquet files using the plan layout.

    The plan layout is::

        {base_path}/{venue}/{symbol}/{timeframe}/year={yyyy}/month={mm}/data.parquet

    This is intentionally simpler than :func:`write_ohlcv`, which writes
    to ``data/raw/ohlcv/{venue}/{canonical}/{timeframe}/{yyyy}/{mm}.parquet``
    under a pydantic :class:`Symbol`. The plan layout is what the W1.5
    spec and the BTC-only fallback data path expect
    (``data/binance/BTCUSDT/1m/year=2025/month=06/data.parquet``).

    The ``year=``/``month=`` directory names follow the hive-partitioning
    convention so that DuckDB can prune partitions when filtered on
    ``year``/``month`` (``hive_partitioning=true``). ``year`` is bound
    as INTEGER and ``month`` as VARCHAR (DuckDB's default for hive
    string values); readers should cast with ``CAST(month AS INT)``
    when comparing numerically.

    Idempotent: if a partition file already exists, the new rows are
    concatenated to it. Parent directories are created as needed.

    Pure-Python, no I/O concurrency, no async.

    Parameters
    ----------
    table:
        PyArrow table that must exactly match :data:`OHLCV_SCHEMA`.
    venue:
        Venue slug (e.g. ``"binance"``).
    symbol:
        Symbol slug in the partition path (e.g. ``"BTCUSDT"``).
    timeframe:
        Timeframe slug (e.g. ``"1m"``).
    base_path:
        Root directory under which the partitioned tree is written.

    Returns
    -------
    list[Path]
        The sorted list of partition files written (one per (year, month)).
    """
    if table.schema != OHLCV_SCHEMA:
        raise ValueError(
            f"OHLCV schema mismatch: expected {OHLCV_SCHEMA}, got {table.schema}"
        )
    if len(table) == 0:
        return []
    base = Path(base_path)
    # Group row slices by (year, month); accumulate pyarrow RecordBatches
    # so we can concatenate per partition without converting to Python.
    months: dict[tuple[int, int], list[pa.RecordBatch]] = {}
    for batch in table.to_batches():
        ts_arr = batch.column("ts")
        # Vectorised extraction: pyarrow timestamps carry year/month in us.
        for i in range(batch.num_rows):
            ts = ts_arr[i].as_py()
            assert isinstance(ts, datetime)
            months.setdefault((ts.year, ts.month), []).append(batch.slice(i, 1))
    written: list[Path] = []
    for (year, month), batches in sorted(months.items()):
        out = base / _month_partition_path(
            venue=venue,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            month=month,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        new_table = pa.Table.from_batches(batches, schema=OHLCV_SCHEMA)
        if out.exists():
            existing = pq.read_table(out)
            new_table = pa.concat_tables([existing, new_table], promote_options="default")
        pq.write_table(new_table, out, compression="snappy")
        written.append(out)
    return written
