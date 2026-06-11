"""Momentum indicators: RSI, Stochastic, Williams %R, CCI."""

from __future__ import annotations

import math

import pyarrow as pa


# ---------------------------------------------------------------------------
# RSI (Wilder's smoothing)
# ---------------------------------------------------------------------------
def rsi(table: pa.Table, *, period: int = 14, source: str = "close") -> pa.Table:
    """Add an RSI column in [0, 100]."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    values = [float(v) for v in table.column(source).to_pylist()]
    n = len(values)
    rsi_out: list[float] = [math.nan] * n
    if n < period + 1:
        return table.append_column(f"rsi_{period}", pa.array(rsi_out, type=pa.float64()))
    # Wilder's smoothing of gains and losses
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        diff = values[i] - values[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    rs0 = avg_gain / avg_loss if avg_loss != 0 else math.inf
    rsi_out[period] = 100.0 - 100.0 / (1.0 + rs0) if math.isfinite(rs0) else 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else math.inf
        rsi_out[i] = 100.0 - 100.0 / (1.0 + rs) if math.isfinite(rs) else 100.0
    return table.append_column(f"rsi_{period}", pa.array(rsi_out, type=pa.float64()))


# ---------------------------------------------------------------------------
# Stochastic Oscillator (%K and %D)
# ---------------------------------------------------------------------------
def stochastic(
    table: pa.Table,
    *,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pa.Table:
    """Add ``stoch_k`` and ``stoch_d`` columns in [0, 100]."""
    if k_period < 1 or d_period < 1 or smooth_k < 1:
        raise ValueError("periods must be >= 1")
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    n = len(close)
    k_raw: list[float] = [math.nan] * n
    for i in range(n):
        if i + 1 >= k_period:
            hh = max(high[i + 1 - k_period : i + 1])
            ll = min(low[i + 1 - k_period : i + 1])
            denom = hh - ll
            k_raw[i] = 100.0 * (close[i] - ll) / denom if denom != 0 else 50.0
    # smooth %K with simple moving average of width `smooth_k`
    k_smooth: list[float] = [math.nan] * n
    for i in range(n):
        if i + 1 >= smooth_k:
            window = [v for v in k_raw[i + 1 - smooth_k : i + 1] if not math.isnan(v)]
            if len(window) == smooth_k:
                k_smooth[i] = sum(window) / smooth_k
    # %D is SMA of %K over d_period
    d_out: list[float] = [math.nan] * n
    for i in range(n):
        if i + 1 >= d_period:
            window = [v for v in k_smooth[i + 1 - d_period : i + 1] if not math.isnan(v)]
            if len(window) == d_period:
                d_out[i] = sum(window) / d_period
    return table.append_column("stoch_k", pa.array(k_smooth, type=pa.float64())).append_column(
        "stoch_d", pa.array(d_out, type=pa.float64())
    )


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------
def williams_r(table: pa.Table, *, period: int = 14) -> pa.Table:
    """Add a ``williams_r`` column in [-100, 0]."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    n = len(close)
    out: list[float] = [math.nan] * n
    for i in range(n):
        if i + 1 >= period:
            hh = max(high[i + 1 - period : i + 1])
            ll = min(low[i + 1 - period : i + 1])
            denom = hh - ll
            if denom != 0:
                out[i] = -100.0 * (hh - close[i]) / denom
    return table.append_column("williams_r", pa.array(out, type=pa.float64()))


# ---------------------------------------------------------------------------
# CCI (Commodity Channel Index)
# ---------------------------------------------------------------------------
def cci(table: pa.Table, *, period: int = 20) -> pa.Table:
    """Add a ``cci`` column.

    CCI = (TP - SMA(TP)) / (0.015 * mean_deviation(TP))
    where TP = (high + low + close) / 3.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    n = len(close)
    tp = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close, strict=True)]
    out: list[float] = [math.nan] * n
    for i in range(n):
        if i + 1 >= period:
            window = tp[i + 1 - period : i + 1]
            mean = sum(window) / period
            md = sum(abs(v - mean) for v in window) / period
            if md != 0:
                out[i] = (tp[i] - mean) / (0.015 * md)
    return table.append_column("cci", pa.array(out, type=pa.float64()))
