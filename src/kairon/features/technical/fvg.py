"""Fair Value Gap (FVG) detection.

FVGs are 3-bar patterns where institutional order flow leaves price gaps.
A bullish FVG occurs when bar[i-1].low > bar[i+1].high (gap between
bars i-1 and i+1). A bearish FVG occurs when bar[i-1].high < bar[i+1].low.

These gaps represent structural imbalances that often get filled (mean-reversion)
or act as support/resistance zones.

Output columns:
- fvg_bullish: 1 if a bullish FVG started at this bar, 0 otherwise
- fvg_bearish: 1 if a bearish FVG started at this bar, 0 otherwise
- fvg_fill_pct: percent fill of the nearest active FVG (0 = unfilled, 1 = filled)
- fvg_nearest_distance: ATR-normalized distance to the nearest unfilled FVG
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def fair_value_gap(table: pa.Table) -> pa.Table:
    """Add Fair Value Gap detection columns.

    Walk-forward safe: each bar's FVG detection uses only bars i-1, i, i+1
    (the i+1 bar is the current bar being processed, not future data).

    Output columns:
        fvg_bullish: 1 if bullish FVG at this bar, 0 otherwise
        fvg_bearish: 1 if bearish FVG at this bar, 0 otherwise
        fvg_fill_pct: percent fill of nearest active FVG [0, 1]
        fvg_nearest_distance: distance to nearest unfilled FVG / ATR_14
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
        # Compute simple ATR inline
        atr_14 = np.zeros(n, dtype=np.float64)
        atr_14[0] = high[0] - low[0]
        for i in range(1, n):
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            atr_14[i] = tr
        # 14-period SMA
        atr_sma = np.zeros(n, dtype=np.float64)
        for i in range(14, n):
            atr_sma[i] = np.mean(atr_14[i - 13 : i + 1])
        atr_14 = np.where(atr_sma > 0, atr_sma, atr_14)

    fvg_bullish = np.zeros(n, dtype=np.float64)
    fvg_bearish = np.zeros(n, dtype=np.float64)
    fvg_fill_pct = np.zeros(n, dtype=np.float64)
    fvg_nearest_dist = np.ones(n, dtype=np.float64)  # Default: 1 ATR away

    # Track active (unfilled) FVGs
    # Each FVG: (start_idx, gap_top, gap_bottom, is_bullish)
    active_fvgs: list[tuple[int, float, float, bool]] = []

    for i in range(2, n):
        # Detect new FVGs (3-bar pattern: bars i-2, i-1, i)
        # Bullish FVG: low[i-2] > high[i] (gap up)
        gap_top = low[i - 2]
        gap_bottom = high[i]
        if gap_top > gap_bottom:
            fvg_bullish[i] = 1.0
            active_fvgs.append((i, gap_top, gap_bottom, True))

        # Bearish FVG: high[i-2] < low[i] (gap down)
        gap_top_2 = low[i]
        gap_bottom_2 = high[i - 2]
        if high[i - 2] < low[i]:
            fvg_bearish[i] = 1.0
            active_fvgs.append((i, low[i], high[i - 2], False))

        # Check fill status of active FVGs
        remaining = []
        nearest_dist = float("inf")
        nearest_fill = 1.0  # Default: fully filled

        for start_idx, gap_top, gap_bottom, is_bullish in active_fvgs:
            gap_size = gap_top - gap_bottom
            if gap_size <= 0:
                continue

            if is_bullish:
                # Bullish FVG is filled when price trades below gap_bottom
                if close[i] <= gap_bottom:
                    continue  # Fully filled, remove
                # Partial fill: how much of the gap is filled from below
                fill_from_below = max(0, gap_top - close[i]) / gap_size
                fill_pct = 1.0 - fill_from_below
                dist = (close[i] - gap_bottom) / atr_14[i] if atr_14[i] > 0 else 1.0
            else:
                # Bearish FVG is filled when price trades above gap_top
                if close[i] >= gap_top:
                    continue  # Fully filled, remove
                # Partial fill: how much of the gap is filled from above
                fill_from_above = max(0, close[i] - gap_bottom) / gap_size
                fill_pct = fill_from_above
                dist = (gap_top - close[i]) / atr_14[i] if atr_14[i] > 0 else 1.0

            remaining.append((start_idx, gap_top, gap_bottom, is_bullish))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_fill = fill_pct

        active_fvgs = remaining

        if nearest_dist < float("inf"):
            fvg_nearest_dist[i] = nearest_dist
            fvg_fill_pct[i] = min(1.0, max(0.0, nearest_fill))

    # Clean up NaN/Inf
    fvg_nearest_dist = np.where(np.isfinite(fvg_nearest_dist), fvg_nearest_dist, 1.0)
    fvg_fill_pct = np.where(np.isfinite(fvg_fill_pct), fvg_fill_pct, 0.0)

    out = table
    out = out.append_column("fvg_bullish", pa.array(fvg_bullish, type=pa.float64()))
    out = out.append_column("fvg_bearish", pa.array(fvg_bearish, type=pa.float64()))
    out = out.append_column("fvg_fill_pct", pa.array(fvg_fill_pct, type=pa.float64()))
    out = out.append_column("fvg_nearest_distance", pa.array(fvg_nearest_dist, type=pa.float64()))
    return out


__all__ = ["fair_value_gap"]