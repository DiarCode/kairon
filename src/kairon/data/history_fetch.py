"""Bybit (TESTNET) historical-bar fetcher for the research store.

Pulls 1m / 5m / 15m bars back N weeks (default 8, per the scalping-edge-
enhancement plan's resolved decision #3) for the candidate scalping universe,
paginated at the exchange's 1000-bar-per-call limit, and stores them via
:mod:`kairon.data.history_store`.

Pagination rationale: :meth:`CCXTAdapter.afetch` chunks a range into
``chunk_ms`` windows (default 30 days) but issues a *single* ``limit=1000``
ccxt call per window — so a 30-day 1m window would return only the first 1000
of ~43 200 bars, silently dropping the rest. To fetch a long 1m range correctly
we drive our own pagination here: each window is exactly ``chunk_bars`` bars
wide (``chunk_bars * tf_seconds`` seconds), so the single ccxt call's 1000-row
limit exactly covers the window with no gaps. The cursor advances by the window
width regardless of how many rows returned (handles exchange gaps and the final
partial window), so the loop always terminates.

Incremental: :func:`sync_history` reads the last stored timestamp and only
fetches from there forward (or from ``now - weeks`` if nothing is stored yet),
then :func:`merge_history` deduplicates and appends. Re-running is cheap.

TESTNET-only: the adapter is constructed with ``testnet=True``. Mainnet
microstructure differs; do not point this at mainnet for the research store.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from kairon.data.adapters.ccxt_adapter import CCXTAdapter
from kairon.data.history_store import (
    dedup_last_per_ts,
    max_stored_ts,
    merge_history,
    read_history,
)
from kairon.data.io import OHLCV_SCHEMA
from kairon.data.symbols import CryptoVenue, Symbol

__all__ = [
    "DEFAULT_TIMEFRAMES",
    "DEFAULT_WEEKS",
    "TF_SECONDS",
    "fetch_history",
    "sync_all",
    "sync_history",
]

# Seconds per bar for the supported research timeframes.
TF_SECONDS: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}

DEFAULT_WEEKS: int = 8
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m")


def _tf_seconds(timeframe: str) -> int:
    try:
        return TF_SECONDS[timeframe]
    except KeyError as e:
        msg = (
            f"unsupported timeframe {timeframe!r}; supported: {sorted(TF_SECONDS)}"
        )
        raise ValueError(msg) from e


async def fetch_history(
    symbol: Symbol,
    timeframe: str,
    *,
    start: datetime,
    end: datetime,
    testnet: bool = True,
    chunk_bars: int = 1000,
) -> pa.Table:
    """Fetch ``[start, end]`` of bars for one symbol/timeframe, paginated.

    Returns an OHLCV-schema table sorted ascending by ``ts``, deduplicated.
    """
    tf_s = _tf_seconds(timeframe)
    window = timedelta(seconds=chunk_bars * tf_s)
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    tables: list[pa.Table] = []
    adapter = CCXTAdapter(venue=CryptoVenue.BYBIT, testnet=testnet)
    try:
        cursor = start
        while cursor < end:
            window_end = min(cursor + window, end)
            try:
                chunk = await adapter.afetch(symbol, timeframe, cursor, window_end)
            except Exception:
                # A single failed window (exchange hiccup, rate limit) must not
                # abort the whole fetch; advance and retry the next window.
                cursor = window_end
                continue
            if chunk.num_rows > 0:
                tables.append(chunk)
            cursor = window_end
    finally:
        await adapter.aclose()
    if not tables:
        return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)
    combined = pa.concat_tables(tables, promote_options="none")
    # Windows are contiguous/non-overlapping, but dedupe defensively (a resumed
    # fetch could overlap the prior window by one bar) and sort ascending.
    return dedup_last_per_ts(combined)


async def sync_history(
    symbol: Symbol,
    timeframe: str,
    *,
    root: Path,
    weeks: int = DEFAULT_WEEKS,
    testnet: bool = True,
    chunk_bars: int = 1000,
    now: datetime | None = None,
) -> pa.Table:
    """Incrementally fetch + merge one symbol/timeframe into the research store.

    Reads the last stored bar timestamp; fetches from ``max(last_ts + 1 bar,
    now - weeks)`` to ``now``. Returns the full merged history for the symbol.
    """
    now = (now or datetime.now(UTC)).astimezone(UTC)
    horizon_start = now - timedelta(weeks=weeks)
    last_ts = max_stored_ts(Path(root), symbol.canonical, timeframe)
    start = horizon_start
    if last_ts is not None:
        # Resume one bar after the last stored bar; never fetch the same bar
        # twice (the merge would dedupe it anyway, but this avoids the work).
        start = max(horizon_start, last_ts + timedelta(seconds=_tf_seconds(timeframe)))
    if start >= now:
        # Already up to date within the horizon.
        return read_history(Path(root), symbol.canonical, timeframe)
    fetched = await fetch_history(
        symbol, timeframe, start=start, end=now, testnet=testnet, chunk_bars=chunk_bars,
    )
    if fetched.num_rows == 0:
        return read_history(Path(root), symbol.canonical, timeframe)
    return merge_history(fetched, Path(root), symbol.canonical, timeframe)


async def sync_all(
    symbols: list[Symbol],
    timeframes: list[str],
    *,
    root: Path,
    weeks: int = DEFAULT_WEEKS,
    testnet: bool = True,
    chunk_bars: int = 1000,
) -> dict[tuple[str, str], int]:
    """Sync every (symbol, timeframe) and return a row-count report.

    Sequential (one venue connection at a time) to stay gentle on the testnet
    rate limit; a fetch of the full 8wk x 3tf universe is a one-time cost.
    """
    report: dict[tuple[str, str], int] = {}
    for sym in symbols:
        for tf in timeframes:
            table = await sync_history(
                sym, tf, root=root, weeks=weeks, testnet=testnet, chunk_bars=chunk_bars,
            )
            report[(sym.canonical, tf)] = table.num_rows
    return report


def sync_all_sync(
    symbols: list[Symbol],
    timeframes: list[str],
    *,
    root: Path,
    weeks: int = DEFAULT_WEEKS,
    testnet: bool = True,
    chunk_bars: int = 1000,
) -> dict[tuple[str, str], int]:
    """Synchronous wrapper around :func:`sync_all` for CLI use."""
    return asyncio.run(
        sync_all(
            symbols, timeframes, root=root, weeks=weeks, testnet=testnet,
            chunk_bars=chunk_bars,
        )
    )
