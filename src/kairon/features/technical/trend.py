"""Trend indicators: EMA, SMA, MACD, ADX, Ichimoku Cloud.

All functions take a pyarrow Table with an OHLCV schema and a
``params`` dict; they return a single new column appended in place
via a copy. Implementations are pure (no IO, no randomness) and
typed with explicit dtypes.
"""

from __future__ import annotations

import math
from typing import Final

import pyarrow as pa


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def _ewma(values: list[float], period: int) -> list[float]:
    """Exponentially-weighted moving average.

    Uses the standard alpha = 2 / (period + 1) recurrence. The first
    ``period - 1`` values are NaN (incomplete window).
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    n = len(values)
    out: list[float] = [math.nan] * n
    if n < period:
        return out
    alpha = 2.0 / (period + 1)
    # seed with the SMA of the first `period` values
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    for i in range(period, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(values: list[float], period: int) -> list[float]:
    """Simple moving average; the first ``period - 1`` values are NaN."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    n = len(values)
    out: list[float] = [math.nan] * n
    if n < period:
        return out
    cumsum = 0.0
    for i in range(n):
        cumsum += values[i]
        if i >= period:
            cumsum -= values[i - period]
        if i >= period - 1:
            out[i] = cumsum / period
    return out


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def ema(table: pa.Table, *, period: int, source: str = "close", out: str | None = None) -> pa.Table:
    """Add an EMA column to ``table``."""
    out_name = out or f"ema_{period}_{source}"
    values = table.column(source).to_pylist()
    result = _ewma([float(v) for v in values], period)
    arr = pa.array(result, type=pa.float64())
    return table.append_column(out_name, arr)


def sma(table: pa.Table, *, period: int, source: str = "close", out: str | None = None) -> pa.Table:
    """Add an SMA column to ``table``."""
    out_name = out or f"sma_{period}_{source}"
    values = table.column(source).to_pylist()
    result = _sma([float(v) for v in values], period)
    arr = pa.array(result, type=pa.float64())
    return table.append_column(out_name, arr)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------
def macd(
    table: pa.Table,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    source: str = "close",
) -> pa.Table:
    """Add MACD, signal, and histogram columns.

    The standard 12/26/9 setup; both fast and slow must be positive
    integers with ``fast < slow``.
    """
    if not (fast > 0 and slow > 0 and signal > 0 and fast < slow):
        raise ValueError(f"invalid MACD params: fast={fast} slow={slow} signal={signal}")
    close = [float(v) for v in table.column(source).to_pylist()]
    fast_ema = _ewma(close, fast)
    slow_ema = _ewma(close, slow)
    macd_line = [a - b for a, b in zip(fast_ema, slow_ema, strict=True)]
    # signal is an EMA of the macd_line, but seeded only where macd_line is non-nan
    # Filter to the finite region, EMA, then put back.
    finite_pairs = [(i, v) for i, v in enumerate(macd_line) if not math.isnan(v)]
    if not finite_pairs:
        signal_line = [math.nan] * len(macd_line)
        hist = [math.nan] * len(macd_line)
    else:
        seed = [v for _, v in finite_pairs]
        signal_partial = _ewma(seed, signal)
        signal_line = [math.nan] * len(macd_line)
        offset = finite_pairs[0][0]
        for j, (_, _) in enumerate(finite_pairs):
            signal_line[offset + j] = signal_partial[j]
        hist = [a - b for a, b in zip(macd_line, signal_line, strict=True)]
    return table.append_column("macd_line", pa.array(macd_line, type=pa.float64())).append_column(
        "macd_signal", pa.array(signal_line, type=pa.float64())
    ).append_column("macd_hist", pa.array(hist, type=pa.float64()))


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------
def adx(table: pa.Table, *, period: int = 14) -> pa.Table:
    """Add ADX, +DI, and -DI columns.

    Standard Wilder smoothing. The first ``2 * period - 1`` values of
    ADX are NaN (insufficient history for the directional indicator
    smoothing).
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    n = len(close)
    if n < 2:
        nan_arr = pa.array([math.nan] * n, type=pa.float64())
        return table.append_column("adx", nan_arr).append_column("plus_di", nan_arr).append_column(
            "minus_di", nan_arr
        )
    # True range
    tr: list[float] = [math.nan] * n
    for i in range(1, n):
        h = high[i]
        l = low[i]
        pc_ = close[i - 1]
        tr[i] = max(h - l, abs(h - pc_), abs(l - pc_))
    # +DM, -DM
    plus_dm: list[float] = [math.nan] * n
    minus_dm: list[float] = [math.nan] * n
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    # Wilder smoothing
    def _wilder(values: list[float], period: int) -> list[float]:
        out: list[float] = [math.nan] * n
        # first valid smoothing starts at index `period` (since we have period valid + 1 bar of history)
        if n < period + 1:
            return out
        seed = sum(values[1 : period + 1])  # skip nan at 0
        out[period] = seed
        for i in range(period + 1, n):
            out[i] = out[i - 1] - out[i - 1] / period + values[i]
        return out

    tr_s = _wilder(tr, period)
    pdm_s = _wilder(plus_dm, period)
    mdm_s = _wilder(minus_dm, period)
    plus_di: list[float] = [math.nan] * n
    minus_di: list[float] = [math.nan] * n
    dx: list[float] = [math.nan] * n
    for i in range(n):
        if math.isnan(tr_s[i]) or tr_s[i] == 0:
            continue
        plus_di[i] = 100.0 * pdm_s[i] / tr_s[i]
        minus_di[i] = 100.0 * mdm_s[i] / tr_s[i]
        denom = plus_di[i] + minus_di[i]
        if denom != 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom
    # ADX is Wilder-smoothed DX
    adx_out: list[float] = [math.nan] * n
    # find first finite dx
    first_finite = next((i for i, v in enumerate(dx) if not math.isnan(v)), n)
    if first_finite + period <= n:
        seed = sum(dx[first_finite : first_finite + period]) / period
        adx_out[first_finite + period - 1] = seed
        for i in range(first_finite + period, n):
            adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
    return table.append_column("plus_di", pa.array(plus_di, type=pa.float64())).append_column(
        "minus_di", pa.array(minus_di, type=pa.float64())
    ).append_column("adx", pa.array(adx_out, type=pa.float64()))


# ---------------------------------------------------------------------------
# Ichimoku Cloud (lite)
# ---------------------------------------------------------------------------
def ichimoku(
    table: pa.Table,
    *,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pa.Table:
    """Add the five Ichimoku lines.

    Tenkan-sen and Kijun-sen are midpoints of the high/low over their
    windows. Senkou Span A is the midpoint of Tenkan and Kijun shifted
    forward by ``kijun`` bars; Senkou Span B is the midpoint of the
    Senkou B window shifted forward by ``kijun`` bars. Chikou Span is
    the close shifted back by ``kijun`` bars.
    """
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    n = len(close)
    nan = [math.nan] * n
    t: list[float] = list(nan)
    k: list[float] = list(nan)
    sb: list[float] = list(nan)
    for i in range(n):
        if i + 1 >= tenkan:
            t[i] = (max(high[i + 1 - tenkan : i + 1]) + min(low[i + 1 - tenkan : i + 1])) / 2
        if i + 1 >= kijun:
            k[i] = (max(high[i + 1 - kijun : i + 1]) + min(low[i + 1 - kijun : i + 1])) / 2
        if i + 1 >= senkou_b:
            sb[i] = (max(high[i + 1 - senkou_b : i + 1]) + min(low[i + 1 - senkou_b : i + 1])) / 2
    # Senkou Span A: midpoint of Tenkan and Kijun, shifted forward by kijun
    sa: list[float] = list(nan)
    for i in range(n):
        if not math.isnan(t[i]) and not math.isnan(k[i]):
            target = i + kijun
            if 0 <= target < n:
                sa[target] = (t[i] + k[i]) / 2
    # Senkou Span B: shift sb forward by kijun
    sb_shift: list[float] = list(nan)
    for i in range(n):
        if not math.isnan(sb[i]):
            target = i + kijun
            if 0 <= target < n:
                sb_shift[target] = sb[i]
    # Chikou Span: close shifted back by kijun
    chikou: list[float] = list(nan)
    for i in range(n):
        target = i - kijun
        if target >= 0:
            chikou[target] = close[i]
    out = (
        table.append_column("ichi_tenkan", pa.array(t, type=pa.float64()))
        .append_column("ichi_kijun", pa.array(k, type=pa.float64()))
        .append_column("ichi_senkou_a", pa.array(sa, type=pa.float64()))
        .append_column("ichi_senkou_b", pa.array(sb_shift, type=pa.float64()))
        .append_column("ichi_chikou", pa.array(chikou, type=pa.float64()))
    )
    _ = Final  # silence unused import for the IDE
    return out
