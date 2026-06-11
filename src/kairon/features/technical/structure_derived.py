"""Swing structure derived features.

Uses the swing high/low detection from ``structure.py`` and computes
distance-to-nearest and range features that capture where price sits
relative to recent market structure.

Output columns:
- dist_to_swing_high: ATR-normalized distance to nearest swing high
- dist_to_swing_low: ATR-normalized distance to nearest swing low
- swing_range_pct: percent of swing range from low (0=at low, 1=at high)
- structure_break_strength: strength of the most recent BOS/CHoCH
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from kairon.features.technical.structure import _rolling_extremum


def swing_structure_features(table: pa.Table, *, order: int = 5) -> pa.Table:
    """Add swing structure derived feature columns.

    Walk-forward safe: uses only past data for swing detection.

    Parameters
    ----------
    order : int
        Window half-width for swing high/low detection (default 5).

    Output columns:
        dist_to_swing_high: (close - last_swing_high) / ATR_14
            Negative = below swing high, positive = above
        dist_to_swing_low: (close - last_swing_low) / ATR_14
            Positive = above swing low (normal), negative = below
        swing_range_pct: (close - swing_low) / (swing_high - swing_low)
            0 = at swing low, 1 = at swing high, 0.5 = midpoint
        structure_break_strength: |close - last_swing_extreme| / ATR_14
            Large values indicate strong breaks of structure
    """
    high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    n = len(close)

    # Get ATR for normalization
    atr_available = "atr_14" in table.column_names
    if atr_available:
        atr_14 = np.array([float(v) for v in table.column("atr_14").to_pylist()], dtype=np.float64)
    else:
        # Compute simple ATR
        atr_14 = np.zeros(n, dtype=np.float64)
        atr_14[0] = high[0] - low[0]
        for i in range(1, n):
            atr_14[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Detect swing highs and lows
    high_list = [float(v) for v in table.column("high").to_pylist()]
    low_list = [float(v) for v in table.column("low").to_pylist()]
    sh_mask = _rolling_extremum(high_list, order=order, kind="high")
    sl_mask = _rolling_extremum(low_list, order=order, kind="low")

    # Track last swing high/low prices
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)

    for i in range(n):
        if i > 0:
            last_sh[i] = last_sh[i - 1]
            last_sl[i] = last_sl[i - 1]
        if sh_mask[i]:
            last_sh[i] = high[i]
        if sl_mask[i]:
            last_sl[i] = low[i]

    # Forward-fill initial NaN with first known swing
    first_sh = np.nan
    first_sl = np.nan
    for i in range(n):
        if not np.isnan(last_sh[i]) and np.isnan(first_sh):
            first_sh = last_sh[i]
        if not np.isnan(last_sl[i]) and np.isnan(first_sl):
            first_sl = last_sl[i]
        if np.isnan(last_sh[i]):
            last_sh[i] = first_sh if not np.isnan(first_sh) else close[i]
        if np.isnan(last_sl[i]):
            last_sl[i] = first_sl if not np.isnan(first_sl) else close[i]

    # Compute features
    safe_atr = np.where(atr_14 > 0, atr_14, 1.0)

    dist_to_sh = (close - last_sh) / safe_atr
    dist_to_sl = (close - last_sl) / safe_atr

    # Percent of swing range
    swing_range = last_sh - last_sl
    safe_range = np.where(np.abs(swing_range) > 1e-10, swing_range, 1.0)
    swing_pct = (close - last_sl) / safe_range
    swing_pct = np.clip(swing_pct, -2.0, 3.0)  # Allow some overshoot

    # Structure break strength: how far price has moved past the last extreme
    break_strength = np.maximum(np.abs(close - last_sh), np.abs(close - last_sl)) / safe_atr
    break_strength = np.where(np.isfinite(break_strength), break_strength, 0.0)

    # Replace NaN
    dist_to_sh = np.where(np.isfinite(dist_to_sh), dist_to_sh, 0.0)
    dist_to_sl = np.where(np.isfinite(dist_to_sl), dist_to_sl, 0.0)
    swing_pct = np.where(np.isfinite(swing_pct), swing_pct, 0.0)

    out = table
    out = out.append_column("dist_to_swing_high", pa.array(dist_to_sh, type=pa.float64()))
    out = out.append_column("dist_to_swing_low", pa.array(dist_to_sl, type=pa.float64()))
    out = out.append_column("swing_range_pct", pa.array(swing_pct, type=pa.float64()))
    out = out.append_column("structure_break_strength", pa.array(break_strength, type=pa.float64()))
    return out


__all__ = ["swing_structure_features"]