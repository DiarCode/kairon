"""Elliott Wave Principle detection engine.

Produces structural position features that tell the ML model *where in the
wave cycle* the market currently sits. This is NOT a trading signal generator;
it is a feature extractor that encodes wave pattern recognition as continuous
features for the direction classifier.

Architecture
------------
1. **ATR-scaled zigzag pivot detection**: A pivot is confirmed when price
   reverses by ``pivot_scale * ATR(14)`` from the last confirmed pivot.
   This avoids noise-induced false pivots while adapting to volatility.

2. **Wave labeling**: Recursive pattern matching on the pivot sequence.
   Tries to match the most recent 5 pivots to an impulse pattern (1-2-3-4-5)
   or 3 pivots to a corrective pattern (A-B-C). Validates against the three
   immutable Elliott Wave rules:
     - Rule 1: Wave 2 retraces ≤ 100% of Wave 1
     - Rule 2: Wave 3 is never the shortest impulse wave
     - Rule 3: Wave 4 does not overlap Wave 1 territory

3. **Fibonacci confluence scoring**: For each labeled wave, compute
   retracement/extension ratios and compare to Fibonacci levels
   (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618).
   Score = 1 - min_distance_to_nearest_fib_level.

4. **Walk-forward safety**: Process bars sequentially. At each bar, update
   the pivot stack, attempt pattern labeling, and output the current wave
   state. No look-ahead.

Output columns
--------------
- ``ew_wave_position``: Current position in wave pattern (1-5 for impulse,
  1-3 for corrective, 0 = unknown)
- ``ew_wave_direction``: +1 = bullish impulse, -1 = bearish impulse,
  0 = corrective/unknown
- ``ew_fib_confluence``: [0, 1] alignment with Fibonacci levels
- ``ew_completion_prob``: [0, 1] estimated probability that current wave
  is ending (based on retracement ratios approaching Fib levels)
- ``ew_retracement_depth``: Current retracement ratio from last significant
  pivot (0 = at pivot, 1 = full retracement, >1 = extension)
- ``ew_is_impulse``: 1 if in impulse pattern, 0 if corrective or unknown

References
----------
- Elliott, R.N. (1946). "Nature's Law — The Secret of the Universe."
- Frost, A.J. & Prechter, R.R. (2005). "Elliott Wave Principle" (10th ed.).
- Volna, V. et al. (2013–2022). NN+EW achieving 77% direction accuracy.
- Vantuch, M. & Zelinka, I. (2018). SVM+RF with EW features >70%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

# Fibonacci ratios used for confluence scoring
_FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618)


@dataclass(frozen=True, slots=True)
class _Pivot:
    """A confirmed zigzag pivot point."""

    idx: int  # Bar index
    price: float  # Pivot price
    kind: str  # "high" or "low"


@dataclass(slots=True)
class _WaveState:
    """Current Elliott Wave labeling state (mutable, updated bar-by-bar)."""

    wave_position: int = 0  # 1-5 impulse, 1-3 corrective, 0 unknown
    wave_direction: int = 0  # +1 bullish, -1 bearish, 0 unknown
    fib_confluence: float = 0.0  # [0, 1]
    completion_prob: float = 0.0  # [0, 1]
    retracement_depth: float = 0.0  # [0, ∞)
    is_impulse: bool = False


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ATR using a simple rolling approach (walk-forward safe)."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan, dtype=np.float64)

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # EMA-based ATR
    atr = np.zeros(n, dtype=np.float64)
    atr[:period] = np.nan
    if n > period:
        atr[period - 1] = np.mean(tr[:period])
        multiplier = 2.0 / (period + 1)
        for i in range(period, n):
            atr[i] = (tr[i] - atr[i - 1]) * multiplier + atr[i - 1]
    return atr


def _zigzag_detect(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    pivot_scale: float = 1.5,
) -> list[list[_Pivot]]:
    """Walk-forward zigzag pivot detection.

    Returns a list (one per bar) of the current pivot stack.
    Each bar gets the pivots confirmed up to and including that bar.
    """
    n = len(close)
    pivot_stacks: list[list[_Pivot]] = [[] for _ in range(n)]
    confirmed: list[_Pivot] = []

    if n < 2:
        return pivot_stacks

    # Initialize with first bar
    direction = 0  # 0 = undecided, 1 = up, -1 = down
    last_pivot_idx = 0
    last_pivot_price = close[0]
    last_pivot_kind = "low"  # Assume start at low

    for i in range(1, n):
        if math.isnan(atr[i]):
            pivot_stacks[i] = list(confirmed)
            continue

        threshold = pivot_scale * atr[i]
        if threshold <= 0:
            threshold = close[i] * 0.01  # 1% fallback

        current_price = close[i]

        if direction == 0:
            # Still deciding initial direction
            if current_price > last_pivot_price + threshold:
                direction = 1
                confirmed.append(_Pivot(idx=last_pivot_idx, price=last_pivot_price, kind="low"))
                last_pivot_idx = i
                last_pivot_price = high[i]
                last_pivot_kind = "high"
            elif current_price < last_pivot_price - threshold:
                direction = -1
                confirmed.append(_Pivot(idx=last_pivot_idx, price=last_pivot_price, kind="high"))
                last_pivot_idx = i
                last_pivot_price = low[i]
                last_pivot_kind = "low"
        elif direction == 1:
            # In uptrend — check for continuation or reversal
            if high[i] > last_pivot_price:
                # New high — update last pivot
                last_pivot_idx = i
                last_pivot_price = high[i]
            elif last_pivot_price - low[i] >= threshold:
                # Reversal — confirm last high pivot, start downtrend
                confirmed.append(_Pivot(idx=last_pivot_idx, price=last_pivot_price, kind="high"))
                direction = -1
                last_pivot_idx = i
                last_pivot_price = low[i]
                last_pivot_kind = "low"
        elif direction == -1:
            # In downtrend — check for continuation or reversal
            if low[i] < last_pivot_price:
                # New low — update last pivot
                last_pivot_idx = i
                last_pivot_price = low[i]
            elif high[i] - last_pivot_price >= threshold:
                # Reversal — confirm last low pivot, start uptrend
                confirmed.append(_Pivot(idx=last_pivot_idx, price=last_pivot_price, kind="low"))
                direction = 1
                last_pivot_idx = i
                last_pivot_price = high[i]
                last_pivot_kind = "high"

        pivot_stacks[i] = list(confirmed)

    return pivot_stacks


def _validate_impulse_rules(pivots: list[_Pivot]) -> bool:
    """Validate the three immutable Elliott Wave rules on a 5-pivot sequence.

    Pivots alternate high/low. For a bullish impulse:
      P0(low) P1(high) P2(low) P3(high) P4(low)
      Wave 1 = P0→P1, Wave 2 = P1→P2, Wave 3 = P2→P3, Wave 4 = P3→P4
    """
    if len(pivots) < 5:
        return False

    p = pivots[-5:]

    # Determine if bullish or bearish impulse
    # Bullish: P0(low), P1(high), P2(low), P3(high), P4(low)
    # Bearish: P0(high), P1(low), P2(high), P3(low), P4(high)
    bullish = p[0].kind == "low"

    if bullish:
        w1_end = p[1].price
        w2_end = p[2].price
        w3_end = p[3].price
        w4_end = p[4].price
        w1_start = p[0].price
    else:
        w1_end = p[1].price
        w2_end = p[2].price
        w3_end = p[3].price
        w4_end = p[4].price
        w1_start = p[0].price

    wave1 = abs(w1_end - w1_start)
    wave2 = abs(w2_end - w1_end)
    wave3 = abs(w3_end - w2_end)

    # Rule 1: Wave 2 retraces ≤ 100% of Wave 1
    if wave1 > 0 and wave2 / wave1 > 1.0:
        return False

    # Rule 2: Wave 3 is never the shortest impulse wave
    if wave3 < wave1 and wave3 < wave2:
        return False

    # Rule 3: Wave 4 does not overlap Wave 1 territory
    if bullish:
        if w4_end <= w1_end:  # Wave 4 low overlaps Wave 1 high
            return False
    else:
        if w4_end >= w1_end:  # Wave 4 high overlaps Wave 1 low
            return False

    return True


def _identify_wave_position(
    pivots: list[_Pivot],
    current_price: float,
    current_idx: int,
) -> _WaveState:
    """Identify the current Elliott Wave position from the pivot stack.

    Examines the most recent pivots to determine if we're in an impulse
    pattern (5-wave) or corrective pattern (3-wave), and which wave
    we're currently in.
    """
    state = _WaveState()

    if len(pivots) < 3:
        return state

    # Try to match impulse pattern (need 5 pivots for full, 3+ for partial)
    # First try with the most recent pivots
    best_impulse = False
    best_corrective = False

    # Check for impulse pattern
    if len(pivots) >= 5:
        # Check last 5 pivots for impulse
        if _validate_impulse_rules(pivots):
            best_impulse = True
            p = pivots[-5:]
            bullish = p[0].kind == "low"

            state.is_impulse = True
            state.wave_direction = 1 if bullish else -1

            # Determine current wave position
            # After P4, we may be in wave 5 or starting a correction
            last_pivot_price = p[-1].price
            if bullish:
                # After wave 4 low, we're in wave 5 if price > P4 low
                state.wave_position = 5
                # Completion probability based on how far price has moved
                wave5_progress = (current_price - last_pivot_price) / abs(p[3].price - p[2].price) if abs(p[3].price - p[2].price) > 0 else 0
                state.completion_prob = min(1.0, wave5_progress / 1.618)  # Wave 5 often extends to 1.618 of wave 1
                state.retracement_depth = (current_price - p[3].price) / (p[1].price - p[3].price) if abs(p[1].price - p[3].price) > 0 else 0

                # Fibonacci confluence
                retracement = (p[3].price - p[4].price) / (p[3].price - p[2].price) if abs(p[3].price - p[2].price) > 0 else 0
            else:
                state.wave_position = 5
                wave5_progress = (last_pivot_price - current_price) / abs(p[2].price - p[3].price) if abs(p[2].price - p[3].price) > 0 else 0
                state.completion_prob = min(1.0, wave5_progress / 1.618)
                state.retracement_depth = (p[3].price - current_price) / (p[3].price - p[1].price) if abs(p[3].price - p[1].price) > 0 else 0
                retracement = (p[4].price - p[3].price) / (p[2].price - p[3].price) if abs(p[2].price - p[3].price) > 0 else 0

            # Fibonacci confluence: how close is the current retracement to a Fib level?
            state.fib_confluence = _fib_confluence_score(abs(retracement))

    # Try to identify position within a developing impulse (3-4 pivots)
    if not best_impulse and len(pivots) >= 3:
        # Use the most recent pivots to estimate position
        recent = pivots[-3:]
        bullish = recent[0].kind == "low"

        # Check if we can identify waves 1-3
        if len(pivots) >= 3:
            p3 = pivots[-3:]
            # Partial impulse: we have at least waves 1-3
            if bullish:
                # Waves 1(up) 2(down) 3(up) — we're in wave 3 or starting 4
                w1 = abs(p3[1].price - p3[0].price)
                w2 = abs(p3[1].price - p3[2].price) if len(p3) > 2 else 0

                # Check if current price is past wave 3 peak (starting wave 4)
                if len(pivots) >= 4:
                    state.wave_position = 4
                    state.wave_direction = 1
                    state.is_impulse = True
                    # Wave 4 retracement
                    p4 = pivots[-4:]
                    ret = (p4[-2].price - current_price) / (p4[-2].price - p4[-3].price) if abs(p4[-2].price - p4[-3].price) > 0 else 0
                    state.retracement_depth = abs(ret)
                    state.fib_confluence = _fib_confluence_score(abs(ret))
                    state.completion_prob = min(1.0, abs(ret) / 0.618)  # Wave 4 typically retraces to 0.382-0.618
                else:
                    state.wave_position = 3
                    state.wave_direction = 1
                    state.is_impulse = True
                    state.completion_prob = 0.5  # In the middle of wave 3
            else:
                # Bearish: waves 1(down) 2(up) 3(down)
                if len(pivots) >= 4:
                    state.wave_position = 4
                    state.wave_direction = -1
                    state.is_impulse = True
                    p4 = pivots[-4:]
                    ret = (current_price - p4[-2].price) / (p4[-3].price - p4[-2].price) if abs(p4[-3].price - p4[-2].price) > 0 else 0
                    state.retracement_depth = abs(ret)
                    state.fib_confluence = _fib_confluence_score(abs(ret))
                    state.completion_prob = min(1.0, abs(ret) / 0.618)
                else:
                    state.wave_position = 3
                    state.wave_direction = -1
                    state.is_impulse = True
                    state.completion_prob = 0.5

    # If no impulse found, try corrective pattern (A-B-C)
    if not state.is_impulse and len(pivots) >= 3:
        recent = pivots[-3:]
        # A-B-C corrective
        state.wave_position = min(3, len(pivots))
        state.wave_direction = 0  # Corrective
        state.is_impulse = False

        # Retracement in corrective
        if len(recent) >= 2:
            p0 = recent[-2].price
            p1 = recent[-1].price
            total_range = abs(p1 - p0) if abs(p1 - p0) > 0 else 1
            state.retracement_depth = abs(current_price - p1) / total_range
            state.fib_confluence = _fib_confluence_score(state.retracement_depth)
            state.completion_prob = min(1.0, state.retracement_depth / 0.618)

    return state


def _fib_confluence_score(retracement: float) -> float:
    """Score how close a retracement ratio is to the nearest Fibonacci level.

    Returns a value in [0, 1] where 1 = exactly at a Fib level,
    0 = far from any Fib level.
    """
    min_dist = min(abs(retracement - level) for level in _FIB_LEVELS)
    # Scale: within 5% of a Fib level → score ≥ 0.6
    return max(0.0, 1.0 - min_dist * 20.0)


def elliott_wave(table: pa.Table, *, atr_period: int = 14, pivot_scale: float = 1.5) -> pa.Table:
    """Add Elliott Wave structural position features.

    Uses ATR-scaled zigzag pivot detection with rule validation.
    Walk-forward safe: processes bars sequentially, no look-ahead.

    Parameters
    ----------
    atr_period : int
        Period for ATR calculation used in pivot detection threshold.
    pivot_scale : float
        Multiplier for ATR to set pivot detection threshold.
        Higher values = fewer pivots (smoother, less noise-sensitive).
        Typical range: 1.0 (sensitive) to 3.0 (smooth).

    Output columns:
        ew_wave_position: Current wave position (1-5 impulse, 1-3 corrective, 0 unknown)
        ew_wave_direction: +1 bullish, -1 bearish, 0 corrective/unknown
        ew_fib_confluence: [0, 1] alignment with Fibonacci levels
        ew_completion_prob: [0, 1] estimated probability current wave is ending
        ew_retracement_depth: Current retracement ratio from last pivot
        ew_is_impulse: 1 if in impulse pattern, 0 otherwise

    Requires: ``close``, ``high``, ``low`` columns on the input table.
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
    n = len(close)

    # Compute ATR for pivot threshold
    atr = _compute_atr(high, low, close, period=atr_period)

    # Run zigzag pivot detection (walk-forward)
    pivot_stacks = _zigzag_detect(high, low, close, atr, pivot_scale=pivot_scale)

    # Process each bar to determine wave state
    wave_position = np.zeros(n, dtype=np.float64)
    wave_direction = np.zeros(n, dtype=np.float64)
    fib_confluence = np.zeros(n, dtype=np.float64)
    completion_prob = np.zeros(n, dtype=np.float64)
    retracement_depth = np.zeros(n, dtype=np.float64)
    is_impulse = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if not pivot_stacks[i]:
            continue

        state = _identify_wave_position(pivot_stacks[i], close[i], i)

        wave_position[i] = float(state.wave_position)
        wave_direction[i] = float(state.wave_direction)
        fib_confluence[i] = state.fib_confluence
        completion_prob[i] = state.completion_prob
        retracement_depth[i] = state.retracement_depth
        is_impulse[i] = 1.0 if state.is_impulse else 0.0

    # Replace NaN with 0 (ATR warm-up period)
    wave_position = np.where(np.isfinite(wave_position), wave_position, 0.0)
    wave_direction = np.where(np.isfinite(wave_direction), wave_direction, 0.0)
    fib_confluence = np.where(np.isfinite(fib_confluence), fib_confluence, 0.0)
    completion_prob = np.where(np.isfinite(completion_prob), completion_prob, 0.0)
    retracement_depth = np.where(np.isfinite(retracement_depth), retracement_depth, 0.0)
    is_impulse = np.where(np.isfinite(is_impulse), is_impulse, 0.0)

    out = table
    out = out.append_column("ew_wave_position", pa.array(wave_position, type=pa.float64()))
    out = out.append_column("ew_wave_direction", pa.array(wave_direction, type=pa.float64()))
    out = out.append_column("ew_fib_confluence", pa.array(fib_confluence, type=pa.float64()))
    out = out.append_column("ew_completion_prob", pa.array(completion_prob, type=pa.float64()))
    out = out.append_column("ew_retracement_depth", pa.array(retracement_depth, type=pa.float64()))
    out = out.append_column("ew_is_impulse", pa.array(is_impulse, type=pa.float64()))
    return out


__all__ = ["elliott_wave"]