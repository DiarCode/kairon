"""Download 2 years of BTC/ETH/SOL OHLCV data from Binance via sync ccxt.

Bypasses the async CCXTAdapter (which has event-loop issues on this
platform) and writes parquet in the same layout that ``read_ohlcv()``
expects.

Usage::

    uv run python scripts/download_real_data.py

Data lands at::

    data/raw/ohlcv/binance/{BTC-USDT,ETH-USDT,SOL-USDT}/{1h,5m}/{YYYY}/{MM}.parquet

Exit code 0 on success, non-zero on fatal error.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pyarrow as pa
from loguru import logger

from kairon.data.io import OHLCV_SCHEMA, DataPaths, write_ohlcv
from kairon.data.symbols import CryptoVenue, Symbol, crypto_spot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH: Path = REPO_ROOT / "data" / "acquisition_report.json"

START: datetime = datetime(2024, 6, 1, tzinfo=UTC)
END: datetime = datetime(2026, 6, 1, tzinfo=UTC)

ASSETS: list[tuple[str, str]] = [
    ("BTC", "USDT"),
    ("ETH", "USDT"),
    ("SOL", "USDT"),
]
TIMEFRAMES: list[str] = ["1h", "5m"]

CHUNK_MS: int = 30 * 24 * 60 * 60 * 1000  # 30 days
RATE_LIMIT_SLEEP: float = 0.5  # seconds between API calls


def _ccxt_timeframe(tf: str) -> str:
    """Map our canonical timeframe to ccxt's string."""
    return tf  # same format for 1h, 5m


def _fetch_all_ohlcv(
    exchange: ccxt.binance,
    market: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> list[list[Any]]:
    """Fetch all OHLCV bars in 30-day chunks with rate limiting."""
    all_rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + CHUNK_MS, end_ms)
        try:
            rows = exchange.fetch_ohlcv(
                market, timeframe,
                since=cursor, limit=1000,
                params={"endTime": chunk_end},
            )
        except Exception as exc:
            logger.warning("Fetch failed for {} {} cursor={}: {}", market, timeframe, cursor, exc)
            time.sleep(5)
            try:
                rows = exchange.fetch_ohlcv(
                    market, timeframe,
                    since=cursor, limit=1000,
                    params={"endTime": chunk_end},
                )
            except Exception as exc2:
                logger.error("Retry failed: {}", exc2)
                rows = []

        if rows:
            all_rows.extend(rows)
            # Move cursor past the last bar timestamp
            last_ts = rows[-1][0]
            if last_ts >= chunk_end - 1:
                cursor = chunk_end
            else:
                cursor = last_ts + 1
        else:
            cursor = chunk_end

        time.sleep(RATE_LIMIT_SLEEP)

    return all_rows


def _rows_to_table(
    rows: list[list[Any]],
    start: datetime,
    end: datetime,
) -> pa.Table:
    """Convert ccxt OHLCV rows to pyarrow Table with OHLCV_SCHEMA."""
    if not rows:
        return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)

    # Parse and filter
    ts = []
    opens = []
    highs = []
    lows = []
    closes = []
    vols = []
    start_dt = start
    for r in rows:
        t = datetime.fromtimestamp(r[0] / 1000, tz=UTC)
        if t < start_dt:
            continue
        if t >= end:
            continue
        ts.append(t)
        opens.append(float(r[1]))
        highs.append(float(r[2]))
        lows.append(float(r[3]))
        closes.append(float(r[4]))
        vols.append(float(r[5]))

    return pa.table(
        {
            "ts": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        },
        schema=OHLCV_SCHEMA,
    )


def main() -> int:
    """Download all datasets using synchronous ccxt."""
    exchange = ccxt.binance({"enableRateLimit": True})
    results: list[dict[str, Any]] = []
    total_pairs = len(ASSETS) * len(TIMEFRAMES)
    pair_idx = 0

    start_ms = int(START.astimezone(UTC).timestamp() * 1000)
    end_ms = int(END.astimezone(UTC).timestamp() * 1000)

    for base, quote in ASSETS:
        symbol = crypto_spot(base, quote, CryptoVenue.BINANCE)
        market = f"{base}/{quote}"

        for tf in TIMEFRAMES:
            pair_idx += 1
            logger.info(
                "Downloading [{}/{}] {} {} from binance [{}, {})",
                pair_idx, total_pairs, symbol.canonical, tf,
                START.isoformat(), END.isoformat(),
            )
            try:
                t0 = time.monotonic()
                rows = _fetch_all_ohlcv(
                    exchange, market, _ccxt_timeframe(tf),
                    start_ms, end_ms,
                )
                table = _rows_to_table(rows, START, END)
                elapsed = time.monotonic() - t0
                n_rows = table.num_rows

                # Write parquet using the existing IO layer
                paths = DataPaths.default()
                written = write_ohlcv(
                    table, symbol=symbol, venue="binance",
                    timeframe=tf, paths=paths,
                )

                # Compute hash
                from kairon.data.io import table_hash
                h = table_hash(table)

                results.append({
                    "symbol": symbol.canonical,
                    "venue": "binance",
                    "timeframe": tf,
                    "start": START.isoformat(),
                    "end": END.isoformat(),
                    "n_rows": n_rows,
                    "table_hash": h,
                    "diagnostic_errors": [],
                    "written_paths": [str(p) for p in written],
                    "elapsed_seconds": round(elapsed, 1),
                    "status": "ok",
                })
                logger.info(
                    "  -> {} rows, hash={}, {} files in {:.1f}s",
                    n_rows, h[:12], len(written), elapsed,
                )
            except Exception as exc:
                logger.error("  -> FAILED: {}", exc)
                results.append({
                    "symbol": symbol.canonical,
                    "venue": "binance",
                    "timeframe": tf,
                    "start": START.isoformat(),
                    "end": END.isoformat(),
                    "n_rows": 0,
                    "table_hash": "",
                    "diagnostic_errors": [str(exc)],
                    "written_paths": [],
                    "elapsed_seconds": 0.0,
                    "status": f"error: {exc}",
                })

            # Small delay between pairs
            if pair_idx < total_pairs:
                time.sleep(2)

    # Write acquisition report
    report: dict[str, Any] = {
        "schema_version": "1",
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "Binance public REST API via ccxt (sync)",
        "date_range": f"{START.isoformat()} to {END.isoformat()}",
        "n_datasets": len(results),
        "datasets": results,
    }
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ACQUISITION SUMMARY")
    print("=" * 80)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = len(results) - ok_count
    total_rows = sum(r["n_rows"] for r in results)
    print(f"{'Symbol':<12} {'TF':<5} {'Rows':>10} {'Files':>6} {'Status':<10}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['symbol']:<12} {r['timeframe']:<5} {r['n_rows']:>10} "
            f"{len(r['written_paths']):>6} {r['status']:<10}"
        )
    print("-" * 50)
    print(
        f"Total: {ok_count} ok, {err_count} errors, "
        f"{total_rows:,} rows across {len(results)} datasets"
    )
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())