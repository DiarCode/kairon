"""Hurst exponent estimation via R/S analysis.

The Hurst exponent characterizes the long-term memory of a time series:
- H < 0.5: mean-reverting (anti-persistent)
- H = 0.5: random walk (Brownian motion)
- H > 0.5: trending (persistent)

For crypto, H > 0.5 during trending regimes and H < 0.5 during ranging
regimes. This feature helps the model identify the current regime's
persistence characteristics.

Uses a rolling window R/S (rescaled range) analysis on log returns.

Output columns:
- hurst_exp: estimated Hurst exponent over a rolling window
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def _rs_analysis(returns: np.ndarray) -> float:
    """Compute R/S statistic for a single window of returns.

    R/S = max(cumsum) - min(cumsum) / std(returns)
    """
    if len(returns) < 10:
        return 0.5  # Default to random walk for short windows

    mean_ret = np.mean(returns)
    deviations = returns - mean_ret
    cumsum = np.cumsum(deviations)
    r = np.max(cumsum) - np.min(cumsum)
    s = np.std(returns, ddof=1)

    if s < 1e-12:
        return 0.5

    return r / s


def _estimate_hurst(returns: np.ndarray, window: int) -> float:
    """Estimate Hurst exponent using R/S analysis on sub-windows.

    Splits the returns into sub-windows of different sizes and fits
    log(R/S) vs log(n) to estimate H.
    """
    n = len(returns)
    if n < window:
        return 0.5

    # Use sub-window sizes: 10, 20, 40 (or smaller if window is small)
    sizes = []
    s = min(10, window // 4)
    while s <= window // 2 and s >= 4:
        sizes.append(s)
        s *= 2

    if len(sizes) < 2:
        return 0.5

    log_n = []
    log_rs = []

    for size in sizes:
        rs_values = []
        # Non-overlapping sub-windows
        num_windows = n // size
        for j in range(num_windows):
            chunk = returns[j * size : (j + 1) * size]
            rs = _rs_analysis(chunk)
            if np.isfinite(rs) and rs > 0:
                rs_values.append(rs)
        if rs_values:
            log_n.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_values)))

    if len(log_n) < 2:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    x = np.array(log_n)
    y = np.array(log_rs)
    slope = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x**2) - np.mean(x) ** 2)

    # Clamp to [0, 1]
    return float(np.clip(slope, 0.0, 1.0))


def hurst_exponent(table: pa.Table, *, window: int = 100) -> pa.Table:
    """Add Hurst exponent estimated over a rolling window.

    Walk-forward safe: each bar's Hurst estimate uses only the past
    ``window`` bars of returns.

    Parameters
    ----------
    window : int
        Rolling window size for Hurst estimation (default 100).
        Minimum recommended: 50. Smaller windows give noisier estimates.

    Output columns:
        hurst_exp: estimated Hurst exponent [0, 1]
            H < 0.5 = mean-reverting, H = 0.5 = random, H > 0.5 = trending
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    n = len(close)

    # Log returns
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = np.log(close[1:] / close[:-1])
    returns = np.where(np.isfinite(returns), returns, 0.0)

    hurst = np.full(n, 0.5, dtype=np.float64)  # Default: random walk

    for i in range(window, n):
        window_returns = returns[i - window + 1 : i + 1]
        hurst[i] = _estimate_hurst(window_returns, window)

    out = table
    out = out.append_column("hurst_exp", pa.array(hurst, type=pa.float64()))
    return out


__all__ = ["hurst_exponent"]