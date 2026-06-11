"""Autoregressive returns: lagged log-returns and rolling momentum.

These features capture the direct autoregressive structure of price
changes. Standard TA indicators (EMA, RSI, MACD) are *smoothed* versions
of these signals; the raw lagged returns provide the model with
unfiltered recent price dynamics.

- lagged_returns: log-returns at lags 1, 2, 3, 5, 10, 20 bars
- rolling_momentum: cumulative returns and z-scored momentum

The Kuznetsov et al. (2025) mutual-information feature selection found
short-term lagged returns among the top features for crypto direction
prediction.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def lagged_returns(table: pa.Table, *, lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)) -> pa.Table:
    """Add lagged log-return columns.

    For each lag ``L``, computes ``log(close[i] / close[i-L])``.
    First ``max(lags)`` bars are NaN; we fill with 0.0 for model compatibility.

    Output columns: ret_lag_1, ret_lag_2, ret_lag_3, ret_lag_5, ret_lag_10, ret_lag_20
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    n = len(close)

    out = table
    for lag in lags:
        col_name = f"ret_lag_{lag}"
        vals = np.zeros(n, dtype=np.float64)
        if n > lag:
            log_ret = np.zeros(n, dtype=np.float64)
            log_ret[lag:] = np.log(close[lag:] / close[:-lag])
            # Replace inf/nan with 0.0
            log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)
            vals = log_ret
        out = out.append_column(col_name, pa.array(vals, type=pa.float64()))
    return out


def rolling_momentum(table: pa.Table) -> pa.Table:
    """Add rolling momentum features.

    Output columns:
        ret_5d: 5-bar cumulative log-return
        ret_20d: 20-bar cumulative log-return
        ret_5d_z: z-score of 5-bar return vs rolling 60-bar window
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    n = len(close)

    # Log returns
    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)

    # 5-bar cumulative return
    ret_5d = np.zeros(n, dtype=np.float64)
    for i in range(5, n):
        ret_5d[i] = np.sum(log_ret[i - 4 : i + 1])
    ret_5d = np.where(np.isfinite(ret_5d), ret_5d, 0.0)

    # 20-bar cumulative return
    ret_20d = np.zeros(n, dtype=np.float64)
    for i in range(20, n):
        ret_20d[i] = np.sum(log_ret[i - 19 : i + 1])
    ret_20d = np.where(np.isfinite(ret_20d), ret_20d, 0.0)

    # Z-scored 5-bar return (rolling 60-bar window)
    ret_5d_z = np.zeros(n, dtype=np.float64)
    window = 60
    for i in range(window, n):
        w = ret_5d[i - window + 1 : i + 1]
        w_mean = np.nanmean(w)
        w_std = np.nanstd(w)
        if w_std > 1e-10:
            ret_5d_z[i] = (ret_5d[i] - w_mean) / w_std
    ret_5d_z = np.where(np.isfinite(ret_5d_z), ret_5d_z, 0.0)

    out = table
    out = out.append_column("ret_5d", pa.array(ret_5d, type=pa.float64()))
    out = out.append_column("ret_20d", pa.array(ret_20d, type=pa.float64()))
    out = out.append_column("ret_5d_z", pa.array(ret_5d_z, type=pa.float64()))
    return out


__all__ = ["lagged_returns", "rolling_momentum"]