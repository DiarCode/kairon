"""Volume indicators: OBV, VWAP, CVD."""

from __future__ import annotations

import pyarrow as pa


def obv(table: pa.Table) -> pa.Table:
    """Add an ``obv`` column (cumulative volume, signed by bar direction)."""
    close = [float(v) for v in table.column("close").to_pylist()]
    vol = [float(v) for v in table.column("volume").to_pylist()]
    n = len(close)
    out: list[float] = [0.0] * n
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - vol[i]
        else:
            out[i] = out[i - 1]
    return table.append_column("obv", pa.array(out, type=pa.float64()))


def vwap(table: pa.Table) -> pa.Table:
    """Add a cumulative ``vwap`` column.

    Note: this is the **cumulative** VWAP. For session-anchored VWAP,
    the caller is expected to slice the frame by session before calling.
    """
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    vol = [float(v) for v in table.column("volume").to_pylist()]
    n = len(close)
    out = [0.0] * n
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        tp = (high[i] + low[i] + close[i]) / 3.0
        cum_pv += tp * vol[i]
        cum_v += vol[i]
        out[i] = cum_pv / cum_v if cum_v != 0 else close[i]
    return table.append_column("vwap", pa.array(out, type=pa.float64()))


def cvd(table: pa.Table) -> pa.Table:
    """Add a cumulative ``cvd`` (cumulative volume delta) column.

    The volume delta is approximated as ``sign(close - open) * volume``
    because we don't have a real tick-by-tick buy/sell classification
    in OHLCV data. For real CVD on tick data, use the Binance aggTrades
    feed.
    """
    opn = [float(v) for v in table.column("open").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    vol = [float(v) for v in table.column("volume").to_pylist()]
    n = len(close)
    delta = [0.0] * n
    for i in range(n):
        if close[i] > opn[i]:
            delta[i] = vol[i]
        elif close[i] < opn[i]:
            delta[i] = -vol[i]
    out = [0.0] * n
    s = 0.0
    for i in range(n):
        s += delta[i]
        out[i] = s
    return table.append_column("cvd", pa.array(out, type=pa.float64()))
