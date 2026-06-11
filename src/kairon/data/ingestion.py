"""Ingestion orchestrator.

The orchestrator coordinates the pipeline: fetch → diagnose → write →
hash → report. It is intentionally tiny: it does not know about
backpressure, scheduling, or retries; those are responsibilities of
the CLI / API / scheduler that calls it.

The orchestrator's only job is to make a single ingestion step
**reproducible**: given the same (adapter, symbol, timeframe, start,
end), it produces the same output files and the same hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from kairon.data.adapters import AdapterError, MarketDataAdapter
from kairon.data.diagnostics import (
    DiagnosticReport,
    assert_quality,
    run_diagnostics,
)
from kairon.data.io import (
    DataPaths,
    table_hash,
    write_ohlcv,
)

if TYPE_CHECKING:
    from datetime import datetime

    from kairon.data.symbols import Symbol


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """The result of a single ingestion step."""

    symbol: str
    venue: str
    timeframe: str
    start: datetime
    end: datetime
    table_hash: str
    n_rows: int
    written_paths: tuple[Path, ...]
    report: DiagnosticReport


def ingest(
    adapter: MarketDataAdapter,
    *,
    symbol: Symbol,
    timeframe: str,
    start: datetime,
    end: datetime,
    paths: DataPaths | None = None,
    raise_on_error: bool = True,
) -> IngestionResult:
    """Fetch → diagnose → write a single (symbol, timeframe) window.

    Parameters
    ----------
    adapter:
        A ``MarketDataAdapter`` implementation (CCXT, FRED, ...).
    symbol, timeframe, start, end:
        Fully-typed inputs; the orchestrator does not coerce them.
    paths:
        Optional ``DataPaths`` override (defaults to ``./data``).
    raise_on_error:
        If True, raise ``ValueError`` when diagnostics find errors;
        if False, return the result with the report intact.
    """
    logger.info(
        "ingest start symbol={} venue={} timeframe={} [{}, {})",
        symbol.canonical,
        adapter.name,
        timeframe,
        start.isoformat(),
        end.isoformat(),
    )
    if start >= end:
        raise ValueError(f"start ({start}) must be < end ({end})")
    try:
        table = adapter.fetch(symbol, timeframe, start, end)
    except AdapterError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise AdapterError(f"adapter.fetch failed: {exc}") from exc
    report = run_diagnostics(
        table, symbol=symbol.canonical, venue=adapter.name, timeframe=timeframe
    )
    if raise_on_error:
        assert_quality(report)
    p = paths or DataPaths.default()
    written_paths_list = write_ohlcv(
        table, symbol=symbol, venue=adapter.name, timeframe=timeframe, paths=p
    )
    h = table_hash(table)
    res = IngestionResult(
        symbol=symbol.canonical,
        venue=adapter.name,
        timeframe=timeframe,
        start=start,
        end=end,
        table_hash=h,
        n_rows=table.num_rows,
        written_paths=tuple(written_paths_list),
        report=report,
    )
    logger.info(
        "ingest done symbol={} venue={} n_rows={} hash={}",
        res.symbol,
        res.venue,
        res.n_rows,
        res.table_hash[:12],
    )
    return res
