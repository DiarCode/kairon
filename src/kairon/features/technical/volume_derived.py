"""Volume-derived features: VWAP deviation and volume imbalance.

These features capture the relationship between price and volume-weighted
benchmarks, and the buy/sell pressure proxy from OHLCV data.

- vwap_deviation: distance from VWAP, normalized by VWAP
- vwap_z: z-score of VWAP deviation over rolling 60-bar window
- vol_imbalance: Bulk Volume Classification proxy for buy/sell ratio
- vol_ratio: current volume vs 20-bar average volume

Bieganowski & Slepaczuk (2026) found VWAP deviation to be the strongest
intraday signal with autocorrelation of 0.975 and half-life of 27.62 min.
Volume imbalance was the top SHAP feature across all crypto assets.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def vwap_deviation(table: pa.Table) -> pa.Table:
    """Add VWAP deviation and z-scored deviation columns.

    Requires that the ``vwap`` column already exists (produced by
    ``volume.vwap``). If not present, computes cumulative VWAP inline.

    Output columns:
        vwap_deviation: (close - vwap) / vwap — percent distance from VWAP
        vwap_z: z-score of vwap_deviation over rolling 60-bar window
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)

    # Compute VWAP if not present
    if "vwap" in table.column_names:
        vwap = np.array([float(v) for v in table.column("vwap").to_pylist()], dtype=np.float64)
    else:
        high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
        low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
        vol = np.array([float(v) for v in table.column("volume").to_pylist()], dtype=np.float64)
        n = len(close)
        vwap = np.zeros(n, dtype=np.float64)
        cum_pv = 0.0
        cum_v = 0.0
        for i in range(n):
            tp = (high[i] + low[i] + close[i]) / 3.0
            cum_pv += tp * vol[i]
            cum_v += vol[i]
            vwap[i] = cum_pv / cum_v if cum_v > 0 else close[i]

    # VWAP deviation (percent)
    vwap_dev = np.where(np.abs(vwap) > 1e-10, (close - vwap) / vwap, 0.0)
    vwap_dev = np.where(np.isfinite(vwap_dev), vwap_dev, 0.0)

    # Z-scored deviation over 60-bar window
    window = 60
    vwap_z = np.zeros_like(vwap_dev)
    for i in range(window, len(vwap_dev)):
        w = vwap_dev[i - window + 1 : i + 1]
        w_mean = np.nanmean(w)
        w_std = np.nanstd(w)
        if w_std > 1e-10:
            vwap_z[i] = (vwap_dev[i] - w_mean) / w_std
    vwap_z = np.where(np.isfinite(vwap_z), vwap_z, 0.0)

    out = table
    out = out.append_column("vwap_deviation", pa.array(vwap_dev, type=pa.float64()))
    out = out.append_column("vwap_z", pa.array(vwap_z, type=pa.float64()))
    return out


def volume_imbalance(table: pa.Table) -> pa.Table:
    """Add volume imbalance (BVC proxy) and relative volume columns.

    The Bulk Volume Classification (BVC) proxy estimates buy/sell pressure
    from OHLCV bars without tick data. It uses the close-open relationship
    relative to the bar's range.

    Output columns:
        vol_imbalance: (close - open) / (high - low + epsilon)
            Values near +1 = strong buying, near -1 = strong selling
        vol_ratio: volume[i] / SMA(volume, 20)
            Values > 1 = above-average volume, < 1 = below-average
    """
    open_ = np.array([float(v) for v in table.column("open").to_pylist()], dtype=np.float64)
    high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    vol = np.array([float(v) for v in table.column("volume").to_pylist()], dtype=np.float64)
    n = len(close)

    # Volume imbalance (BVC proxy)
    bar_range = high - low
    # Avoid division by zero for doji bars (range ≈ 0)
    bar_range = np.where(bar_range > 1e-10, bar_range, 1e-10)
    vol_imb = (close - open_) / bar_range
    vol_imb = np.clip(vol_imb, -1.0, 1.0)  # Clamp to [-1, 1]
    vol_imb = np.where(np.isfinite(vol_imb), vol_imb, 0.0)

    # Relative volume (current vs 20-bar SMA)
    sma_window = 20
    vol_sma = np.zeros(n, dtype=np.float64)
    for i in range(sma_window, n):
        vol_sma[i] = np.mean(vol[i - sma_window + 1 : i + 1])
    # Use np.divide with where= to avoid computing division where vol_sma ≈ 0
    vol_ratio = np.divide(vol, vol_sma, out=np.ones(n, dtype=np.float64), where=vol_sma > 1e-10)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 1.0)

    out = table
    out = out.append_column("vol_imbalance", pa.array(vol_imb, type=pa.float64()))
    out = out.append_column("vol_ratio", pa.array(vol_ratio, type=pa.float64()))
    return out


__all__ = ["vwap_deviation", "volume_imbalance"]