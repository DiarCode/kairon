"""Data loading, timeframe detection, and feature set selection for Kairon CLI."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.registry import ALL_FEATURES

# Volume-dependent features that require real volume data
VOLUME_FEATURES: frozenset[str] = frozenset({
    "volume.obv",
    "volume.vwap",
    "volume.cvd",
    "volume.vwap_deviation",
    "volume.volume_imbalance",
})

# Known timeframes mapped to their duration in seconds
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


@dataclass(frozen=True, slots=True)
class TimeframeInfo:
    """Auto-detected timeframe metadata."""

    name: str          # e.g. "1w", "1h", "1d"
    seconds: int       # duration in seconds
    bar_count: int     # number of bars in the dataset


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Result of loading and validating OHLCV data."""

    table: pa.Table           # Validated OHLCV with OHLCV_SCHEMA
    symbol: str
    timeframe: TimeframeInfo
    has_volume: bool          # True if volume column has real (non-zero) data


def detect_timeframe(ts_column: pa.ChunkedArray) -> TimeframeInfo:
    """Infer timeframe from timestamp spacing.

    Strategy:
    1. Compute deltas between consecutive timestamps
    2. Use the 25th percentile delta (robust to gaps and outliers)
    3. Map to nearest known timeframe
    4. Validate that >=50% of deltas fall within ±20% of the detected timeframe
    """
    ts_values = ts_column.to_pylist()
    if len(ts_values) < 3:
        raise ValueError(f"Need at least 3 bars to detect timeframe, got {len(ts_values)}")

    # Compute deltas in seconds
    deltas = []
    for i in range(1, len(ts_values)):
        if ts_values[i] is not None and ts_values[i - 1] is not None:
            delta = (ts_values[i] - ts_values[i - 1]).total_seconds()
            if delta > 0:
                deltas.append(delta)

    if not deltas:
        raise ValueError("Could not compute any valid timestamp deltas")

    deltas_arr = np.array(deltas)

    # Use 25th percentile (robust to weekends, holidays, gaps)
    p25 = np.percentile(deltas_arr, 25)
    median = np.median(deltas_arr)

    # Find nearest known timeframe
    best_name = "1d"  # default
    best_seconds = 86400
    best_diff = float("inf")

    for name, secs in TIMEFRAME_SECONDS.items():
        diff = abs(median - secs)
        if diff < best_diff:
            best_diff = diff
            best_name = name
            best_seconds = secs

    # Validate: >=50% of deltas should be within ±20% of detected timeframe
    lower = best_seconds * 0.8
    upper = best_seconds * 1.2
    within_range = np.sum((deltas_arr >= lower) & (deltas_arr <= upper))
    ratio = within_range / len(deltas_arr)

    if ratio < 0.3:
        # Low confidence — fall back to 25th percentile estimate
        for name, secs in TIMEFRAME_SECONDS.items():
            diff = abs(p25 - secs)
            if diff < best_diff:
                best_diff = diff
                best_name = name
                best_seconds = secs

    return TimeframeInfo(name=best_name, seconds=best_seconds, bar_count=len(ts_values))


def select_feature_set(has_volume: bool) -> tuple[str, ...]:
    """Select appropriate feature set based on data availability.

    Parameters
    ----------
    has_volume : bool
        True if the volume column has real (non-zero) data.

    Returns
    -------
    tuple[str, ...]
        Feature names to pass to FeaturePipeline.
    """
    if has_volume:
        return ALL_FEATURES
    return tuple(f for f in ALL_FEATURES if f not in VOLUME_FEATURES)


def _detect_delimiter(path: Path) -> str:
    """Auto-detect CSV delimiter (semicolon vs comma).

    Reads the first 2KB and uses csv.Sniffer.
    Falls back to comma if detection fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(2048)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=";,|\t")
        return dialect.delimiter
    except csv.Error:
        # Fallback: check if semicolons are more common than commas in data lines
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(2048)
        lines = sample.strip().split("\n")
        if len(lines) > 1:
            data_line = lines[1]
            if data_line.count(";") > data_line.count(","):
                return ";"
        return ","


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to match OHLCV_SCHEMA expectations."""
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("time", "timestamp", "date", "datetime"):
            col_map[col] = "ts"
        elif lower == "volume" or lower == "vol":
            col_map[col] = "volume"
        elif lower == "open" or lower == "o":
            col_map[col] = "open"
        elif lower == "high" or lower == "h":
            col_map[col] = "high"
        elif lower == "low" or lower == "l":
            col_map[col] = "low"
        elif lower == "close" or lower == "c":
            col_map[col] = "close"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def _check_volume(df: pd.DataFrame) -> bool:
    """Check if volume column has real data (not all zeros or missing)."""
    if "volume" not in df.columns:
        return False
    vol = df["volume"].values
    return bool(np.any(vol > 0))


def load_csv(
    path: Path,
    symbol: str = "UNKNOWN",
    *,
    sep: str = "auto",
    timeframe_override: str | None = None,
) -> LoadResult:
    """Load any CSV with OHLCV columns into a validated pyarrow Table.

    Handles:
    - Auto-detect delimiter (semicolon vs comma)
    - Normalize column names (Time→ts, Volume→volume, etc.)
    - Parse timestamps as UTC
    - Handle missing volume column (fill with 0.0)
    - Validate via OHLCV_SCHEMA
    - Auto-detect timeframe from timestamp spacing
    - Check for minimum data length

    Parameters
    ----------
    path : Path
        Path to CSV file.
    symbol : str
        Asset symbol (e.g. BTC, ETH).
    sep : str
        Delimiter. "auto" detects automatically.
    timeframe_override : str or None
        Override auto-detected timeframe (e.g. "1w", "1h").

    Returns
    -------
    LoadResult
        Validated table, symbol, timeframe info, and volume availability.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Detect delimiter
    delimiter = _detect_delimiter(path) if sep == "auto" else sep

    # Read CSV
    df = pd.read_csv(path, sep=delimiter)
    df = _normalize_columns(df)

    # Ensure required columns exist
    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # Add volume if missing
    has_volume = "volume" in df.columns and _check_volume(df)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    elif not has_volume:
        df["volume"] = 0.0

    # Parse timestamps as UTC
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Sort by timestamp
    df = df.sort_values("ts").reset_index(drop=True)

    # Cast OHLCV columns to float64
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(np.float64)

    # Select columns in OHLCV_SCHEMA order
    df = df[["ts", "open", "high", "low", "close", "volume"]]

    # Minimum data length check
    min_bars = 50
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} bars for analysis, got {len(df)}. "
            f"Provide more data or use a shorter timeframe."
        )

    # Convert to pyarrow Table
    table = pa.Table.from_pandas(df, schema=OHLCV_SCHEMA)

    # Detect timeframe
    if timeframe_override:
        if timeframe_override not in TIMEFRAME_SECONDS:
            raise ValueError(
                f"Unknown timeframe '{timeframe_override}'. "
                f"Valid: {list(TIMEFRAME_SECONDS.keys())}"
            )
        tf = TimeframeInfo(
            name=timeframe_override,
            seconds=TIMEFRAME_SECONDS[timeframe_override],
            bar_count=table.num_rows,
        )
    else:
        tf = detect_timeframe(table.column("ts"))

    return LoadResult(table=table, symbol=symbol, timeframe=tf, has_volume=has_volume)