"""Order Block detection.

Order blocks are the last bearish candle before a strong bullish move (bullish OB)
or the last bullish candle before a strong bearish move (bearish OB). They represent
institutional supply/demand zones where large orders were placed.

Output columns:
- ob_bullish_near: 1 if price is near a bullish order block (within 1 ATR)
- ob_bearish_near: 1 if price is near a bearish order block (within 1 ATR)
- ob_in_bullish_zone: 1 if price is inside a bullish OB zone
- ob_in_bearish_zone: 1 if price is inside a bearish OB zone
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def order_blocks(table: pa.Table, *, lookback: int = 20) -> pa.Table:
    """Add Order Block detection columns.

    An order block is identified when:
    - Bullish OB: Last bearish candle before a strong bullish move (engulfing)
    - Bearish OB: Last bullish candle before a strong bearish move (engulfing)

    Walk-forward safe: only uses past data for detection.

    Parameters
    ----------
    lookback : int
        How many bars back to track active order blocks (default 20).

    Output columns:
        ob_bullish_near: 1 if within 1 ATR of a bullish OB
        ob_bearish_near: 1 if within 1 ATR of a bearish OB
        ob_in_bullish_zone: 1 if price is inside a bullish OB zone
        ob_in_bearish_zone: 1 if price is inside a bearish OB zone
    """
    open_ = np.array([float(v) for v in table.column("open").to_pylist()], dtype=np.float64)
    high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    n = len(close)

    # Get ATR for normalization
    atr_available = "atr_14" in table.column_names
    if atr_available:
        atr_14 = np.array([float(v) for v in table.column("atr_14").to_pylist()], dtype=np.float64)
    else:
        atr_14 = np.ones(n, dtype=np.float64)
        atr_14[0] = high[0] - low[0]
        for i in range(1, n):
            atr_14[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    ob_bullish_near = np.zeros(n, dtype=np.float64)
    ob_bearish_near = np.zeros(n, dtype=np.float64)
    ob_in_bullish = np.zeros(n, dtype=np.float64)
    ob_in_bearish = np.zeros(n, dtype=np.float64)

    # Track active order blocks: (idx, ob_high, ob_low, is_bullish)
    # OB zone: for bullish OB, the zone is [low, high] of the bearish candle
    # For bearish OB, the zone is [low, high] of the bullish candle
    active_obs: list[tuple[int, float, float, bool]] = []

    for i in range(2, n):
        # Detect order blocks
        bar_body = close[i] - open_[i]
        prev_body = close[i - 1] - open_[i - 1]
        prev2_body = close[i - 2] - open_[i - 2]
        bar_range = high[i] - low[i]
        if bar_range > 0:
            # Bullish OB: bar[i-2] is bearish, bar[i-1] transitions, bar[i] is strong bullish
            if prev2_body < 0 and bar_body > 0:
                # The bearish candle at i-2 is a bullish OB
                ob_high_val = high[i - 2]
                ob_low_val = low[i - 2]
                active_obs.append((i - 2, ob_high_val, ob_low_val, True))

            # Bearish OB: bar[i-2] is bullish, bar[i-1] transitions, bar[i] is strong bearish
            if prev2_body > 0 and bar_body < 0:
                ob_high_val = high[i - 2]
                ob_low_val = low[i - 2]
                active_obs.append((i - 2, ob_high_val, ob_low_val, False))

        # Prune old order blocks beyond lookback
        active_obs = [(idx, oh, ol, ib) for idx, oh, ol, ib in active_obs if i - idx <= lookback]

        # Check if price is near/inside any order block
        near_bull = False
        near_bear = False
        in_bull = False
        in_bear = False

        for idx, ob_h, ob_l, is_bullish in active_obs:
            atr_val = atr_14[i] if atr_14[i] > 0 else 1.0

            if is_bullish:
                # Bullish OB: support zone
                if ob_l <= close[i] <= ob_h:
                    in_bull = True
                if abs(close[i] - ob_l) <= atr_val or abs(close[i] - ob_h) <= atr_val:
                    near_bull = True
            else:
                # Bearish OB: resistance zone
                if ob_l <= close[i] <= ob_h:
                    in_bear = True
                if abs(close[i] - ob_l) <= atr_val or abs(close[i] - ob_h) <= atr_val:
                    near_bear = True

        ob_bullish_near[i] = 1.0 if near_bull else 0.0
        ob_bearish_near[i] = 1.0 if near_bear else 0.0
        ob_in_bullish[i] = 1.0 if in_bull else 0.0
        ob_in_bearish[i] = 1.0 if in_bear else 0.0

    out = table
    out = out.append_column("ob_bullish_near", pa.array(ob_bullish_near, type=pa.float64()))
    out = out.append_column("ob_bearish_near", pa.array(ob_bearish_near, type=pa.float64()))
    out = out.append_column("ob_in_bullish_zone", pa.array(ob_in_bullish, type=pa.float64()))
    out = out.append_column("ob_in_bearish_zone", pa.array(ob_in_bearish, type=pa.float64()))
    return out


__all__ = ["order_blocks"]