"""Volatility indicators: Bollinger Bands, ATR."""

from __future__ import annotations

import math

import pyarrow as pa


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------
def bollinger(
    table: pa.Table,
    *,
    period: int = 20,
    std_dev: float = 2.0,
    source: str = "close",
) -> pa.Table:
    """Add ``bb_mid``, ``bb_upper``, ``bb_lower`` columns."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if std_dev <= 0:
        raise ValueError(f"std_dev must be > 0, got {std_dev}")
    values = [float(v) for v in table.column(source).to_pylist()]
    n = len(values)
    mid: list[float] = [math.nan] * n
    upper: list[float] = [math.nan] * n
    lower: list[float] = [math.nan] * n
    for i in range(n):
        if i + 1 >= period:
            window = values[i + 1 - period : i + 1]
            mean = sum(window) / period
            var = sum((v - mean) ** 2 for v in window) / period
            sd = math.sqrt(var)
            mid[i] = mean
            upper[i] = mean + std_dev * sd
            lower[i] = mean - std_dev * sd
    return (
        table.append_column("bb_mid", pa.array(mid, type=pa.float64()))
        .append_column("bb_upper", pa.array(upper, type=pa.float64()))
        .append_column("bb_lower", pa.array(lower, type=pa.float64()))
    )


# ---------------------------------------------------------------------------
# ATR (Average True Range) — Wilder's smoothing
# ---------------------------------------------------------------------------
def atr(table: pa.Table, *, period: int = 14) -> pa.Table:
    """Add an ``atr`` column."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    n = len(close)
    tr: list[float] = [math.nan] * n
    for i in range(1, n):
        h = high[i]
        l = low[i]
        pc = close[i - 1]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    out: list[float] = [math.nan] * n
    if n < period + 1:
        return table.append_column(f"atr_{period}", pa.array(out, type=pa.float64()))
    seed = sum(tr[1 : period + 1]) / period
    out[period] = seed
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return table.append_column(f"atr_{period}", pa.array(out, type=pa.float64()))
