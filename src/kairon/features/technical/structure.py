"""Market structure: BOS/CHoCH (Smart Money Concepts), Fibonacci levels,
and a small set of candlestick patterns.

These are *derived* features, not trading signals. They are meant to
feed the model as additional context, not to be acted on directly.
"""

from __future__ import annotations

import math

import pyarrow as pa

# ---------------------------------------------------------------------------
# Swing highs / lows
# ---------------------------------------------------------------------------


def _rolling_extremum(
    values: list[float],
    *,
    order: int,
    kind: str,  # "high" or "low"
) -> list[bool]:
    """Mark a True at index ``i`` if ``values[i]`` is the strict max/min
    of a window of size ``2 * order + 1`` centered on ``i``."""
    n = len(values)
    out = [False] * n
    for i in range(order, n - order):
        window = values[i - order : i + order + 1]
        if kind == "high" and values[i] == max(window) and values[i] not in (
            values[i - order],
            values[i + order],
        ):
            # must be strict
            if all(values[i] > v for v in window if v != values[i]):
                out[i] = True
        if kind == "low" and values[i] == min(window):
            if all(values[i] < v for v in window if v != values[i]):
                out[i] = True
    return out


def swing_highs(table: pa.Table, *, order: int = 5) -> pa.Table:
    """Mark swing-high bars (strict local maxima)."""
    high = [float(v) for v in table.column("high").to_pylist()]
    mask = _rolling_extremum(high, order=order, kind="high")
    return table.append_column("swing_high", pa.array(mask, type=pa.bool_()))


def swing_lows(table: pa.Table, *, order: int = 5) -> pa.Table:
    """Mark swing-low bars (strict local minima)."""
    low = [float(v) for v in table.column("low").to_pylist()]
    mask = _rolling_extremum(low, order=order, kind="low")
    return table.append_column("swing_low", pa.array(mask, type=pa.bool_()))


# ---------------------------------------------------------------------------
# BOS / CHoCH (Break of Structure / Change of Character)
# ---------------------------------------------------------------------------
def bos_choch(table: pa.Table, *, order: int = 5) -> pa.Table:
    """Add ``bos`` and ``choch`` columns.

    - ``bos`` is True when the close breaks above the most recent
      swing-high (bullish) or below the most recent swing-low (bearish).
    - ``choch`` is True when ``bos`` flips direction (e.g., bullish BOS
      after a bearish trend).
    """
    n = table.num_rows
    bos = [0] * n  # 1 = bullish, -1 = bearish, 0 = none
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    sh_mask = _rolling_extremum(high, order=order, kind="high")
    sl_mask = _rolling_extremum(low, order=order, kind="low")
    last_sh = math.nan
    last_sl = math.nan
    prev_bos = 0
    choch = [0] * n
    for i in range(n):
        if sh_mask[i]:
            last_sh = high[i]
        if sl_mask[i]:
            last_sl = low[i]
        if not math.isnan(last_sh) and close[i] > last_sh:
            bos[i] = 1
            if prev_bos == -1:
                choch[i] = 1
            last_sh = math.nan
            prev_bos = 1
        elif not math.isnan(last_sl) and close[i] < last_sl:
            bos[i] = -1
            if prev_bos == 1:
                choch[i] = -1
            last_sl = math.nan
            prev_bos = -1
    return table.append_column("bos", pa.array(bos, type=pa.int8())).append_column(
        "choch", pa.array(choch, type=pa.int8())
    )


# ---------------------------------------------------------------------------
# Fibonacci retracement levels (static, computed over the full frame)
# ---------------------------------------------------------------------------
def fibonacci_levels(
    table: pa.Table, *, lookback: int = 100
) -> pa.Table:
    """Add Fibonacci retracement levels computed from rolling window.

    Output columns: ``fib_236``, ``fib_382``, ``fib_500``, ``fib_618``,
    ``fib_786`` (each as a price level).
    """
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    n = len(high)
    ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    cols: dict[str, list[float]] = {f"fib_{int(r * 1000):03d}": [math.nan] * n for r in ratios}
    for i in range(n):
        if i + 1 < lookback:
            continue
        window_h = high[i + 1 - lookback : i + 1]
        window_l = low[i + 1 - lookback : i + 1]
        hh = max(window_h)
        ll = min(window_l)
        rng = hh - ll
        if rng <= 0:
            continue
        for r in ratios:
            cols[f"fib_{int(r * 1000):03d}"][i] = hh - r * rng
    out = table
    for name, vals in cols.items():
        out = out.append_column(name, pa.array(vals, type=pa.float64()))
    return out


# ---------------------------------------------------------------------------
# Fibonacci proximity (ATR-normalized distance to Fib levels)
# ---------------------------------------------------------------------------
def fibonacci_proximity(table: pa.Table, *, lookback: int = 100) -> pa.Table:
    """Add ATR-normalized distance from close to nearest Fibonacci levels.

    Unlike ``fibonacci_levels`` which outputs raw price columns, this
    function computes the *proximity* of the current price to each Fib
    level, normalized by ATR. Values near 0 mean "at the level", which
    is what the ML model needs.

    Output columns: ``fib_dist_236``, ``fib_dist_382``, ``fib_dist_500``,
    ``fib_dist_618``, ``fib_dist_786`` (each = (close - fib_level) / ATR_14)
    """
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close_list = [float(v) for v in table.column("close").to_pylist()]
    n = len(close_list)

    # Get ATR for normalization
    if "atr_14" in table.column_names:
        atr_vals = [float(v) for v in table.column("atr_14").to_pylist()]
    else:
        # Compute simple ATR inline
        atr_vals = [0.0] * n
        atr_vals[0] = high[0] - low[0]
        for i in range(1, n):
            atr_vals[i] = max(
                high[i] - low[i],
                abs(high[i] - close_list[i - 1]),
                abs(low[i] - close_list[i - 1]),
            )

    ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    col_names = [f"fib_dist_{int(r * 1000):03d}" for r in ratios]
    cols: dict[str, list[float]] = {name: [0.0] * n for name in col_names}

    for i in range(n):
        if i + 1 < lookback:
            # Not enough data for Fib calculation
            for name in col_names:
                cols[name][i] = 0.0
            continue

        window_h = high[i + 1 - lookback : i + 1]
        window_l = low[i + 1 - lookback : i + 1]
        hh = max(window_h)
        ll = min(window_l)
        rng = hh - ll
        atr_val = atr_vals[i] if atr_vals[i] > 0 else rng * 0.02  # 2% fallback

        if rng <= 0 or atr_val <= 0:
            for name in col_names:
                cols[name][i] = 0.0
            continue

        for r, name in zip(ratios, col_names):
            fib_level = hh - r * rng
            # Normalized distance: near 0 = at the Fib level
            dist = (close_list[i] - fib_level) / atr_val
            cols[name][i] = max(-10.0, min(10.0, dist))  # Clamp

    out = table
    for name in col_names:
        out = out.append_column(name, pa.array(cols[name], type=pa.float64()))
    return out


# ---------------------------------------------------------------------------
# Candlestick patterns (basic)
# ---------------------------------------------------------------------------
def candlestick_patterns(table: pa.Table) -> pa.Table:
    """Mark common candlestick patterns as boolean columns.

    Patterns detected (basic single-bar or 2-bar):
    - ``cdl_doji``: open ≈ close (within 0.1% of range)
    - ``cdl_hammer``: long lower wick, small body at top
    - ``cdl_shooting_star``: long upper wick, small body at bottom
    - ``cdl_engulfing_bull``: bullish bar engulfs previous bearish bar
    - ``cdl_engulfing_bear``: bearish bar engulfs previous bullish bar
    """
    opn = [float(v) for v in table.column("open").to_pylist()]
    high = [float(v) for v in table.column("high").to_pylist()]
    low = [float(v) for v in table.column("low").to_pylist()]
    close = [float(v) for v in table.column("close").to_pylist()]
    n = len(close)
    doji = [False] * n
    hammer = [False] * n
    star = [False] * n
    eng_bull = [False] * n
    eng_bear = [False] * n
    for i in range(n):
        rng = high[i] - low[i]
        if rng <= 0:
            continue
        body = abs(close[i] - opn[i])
        upper_wick = high[i] - max(opn[i], close[i])
        lower_wick = min(opn[i], close[i]) - low[i]
        if body / rng <= 0.001:
            doji[i] = True
        if lower_wick >= 2 * body and upper_wick <= body and close[i] > opn[i]:
            hammer[i] = True
        if upper_wick >= 2 * body and lower_wick <= body and close[i] < opn[i]:
            star[i] = True
        if i >= 1:
            prev_body = abs(close[i - 1] - opn[i - 1])
            prev_dir = 1 if close[i - 1] >= opn[i - 1] else -1
            this_dir = 1 if close[i] >= opn[i] else -1
            if prev_dir == -1 and this_dir == 1 and body > prev_body and close[i] > opn[i - 1] and opn[i] < close[i - 1]:
                eng_bull[i] = True
            if prev_dir == 1 and this_dir == -1 and body > prev_body and close[i] < opn[i - 1] and opn[i] > close[i - 1]:
                eng_bear[i] = True
    out = (
        table.append_column("cdl_doji", pa.array(doji, type=pa.bool_()))
        .append_column("cdl_hammer", pa.array(hammer, type=pa.bool_()))
        .append_column("cdl_shooting_star", pa.array(star, type=pa.bool_()))
    )
    out = out.append_column("cdl_engulfing_bull", pa.array(eng_bull, type=pa.bool_()))
    out = out.append_column("cdl_engulfing_bear", pa.array(eng_bear, type=pa.bool_()))
    return out
