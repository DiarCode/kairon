"""Sweet spot detection — model-primary confluence with feature corroboration.

Design principle (from codebase docstrings): features feed models, not trading decisions.
The ML model is the primary signal generator. Feature conditions corroborate or
contradict the model's prediction within a bounded adjustment range (+/-0.25).

A SweetSpot fires when:
- model_confidence > threshold (default 0.45)
- combined_score >= model_confidence + 0.03 (at least one meaningful corroboration)
- at least one justification is produced
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SweetSpot:
    """A buy or sell sweet spot with timing, confidence, and justifications."""

    bar_index: int
    timestamp: datetime
    price: float
    direction: Literal["BUY", "SELL"]
    model_confidence: float            # proba[predicted_class] from multi-head model
    model_direction: int               # predicted class: 0=down, 1=flat, 2=up
    combined_score: float              # model_confidence + corroboration (bounded 0-1)
    timing_horizon: str                # "immediate", "near-term", "medium-term", "long-term"
    justifications: tuple[str, ...]   # Human-readable reasons
    corroboration: tuple[tuple[str, float], ...]  # (feature_name, adjustment_value)


# Timeframe to deduplication lookback mapping
_DEDUP_LOOKBACK: dict[str, int] = {
    "1m": 20, "5m": 20, "15m": 10, "30m": 10,
    "1h": 10, "4h": 5, "1d": 5, "1w": 3,
}

# Timeframe to timing horizon mapping
_TIMING_HORIZON: dict[str, str] = {
    "1m": "immediate", "5m": "immediate",
    "15m": "near-term", "30m": "near-term", "1h": "near-term",
    "4h": "medium-term", "1d": "medium-term",
    "1w": "long-term",
}

# Default threshold for model confidence
DEFAULT_THRESHOLD = 0.45


def _check_buy_conditions(
    row: dict,
    bos_window: dict[int, int],
) -> list[tuple[bool, float, str]]:
    """Evaluate BUY corroboration conditions for a single bar.

    Returns list of (condition_met, adjustment, justification) tuples.
    Only True conditions contribute to the score.
    """
    conditions = []

    # 1. EW impulse pullback (wave 2 or 4 in bullish impulse)
    ew_pos = row.get("ew_wave_position", 0.0)
    ew_dir = row.get("ew_wave_direction", 0.0)
    ew_imp = row.get("ew_is_impulse", 0.0)
    if ew_imp > 0.5 and ew_dir > 0.5 and ew_pos in (2.0, 4.0):
        conditions.append((
            True, 0.04,
            f"In bullish impulse wave W{int(ew_pos)} - pullback buying opportunity",
        ))
    else:
        conditions.append((False, 0.0, ""))

    # 2. EW correction ending (completion_prob > 0.6, not impulse, direction not bearish)
    ew_comp = row.get("ew_completion_prob", 0.0)
    if ew_comp > 0.6 and ew_imp < 0.5 and ew_dir > -0.5:
        conditions.append((
            True, 0.04,
            f"Corrective wave ending (prob={ew_comp:.0%}) - reversal expected",
        ))
    else:
        conditions.append((False, 0.0, ""))

    # 3. Fib support (near 61.8% or 78.6% retracement)
    fib_618 = abs(row.get("fib_dist_618", 99.0))
    fib_786 = abs(row.get("fib_dist_786", 99.0))
    if fib_618 < 0.5 or fib_786 < 0.5:
        conditions.append((
            True, 0.03,
            f"Near Fibonacci support (Fib618 dist={fib_618:.2f} ATR)",
        ))
    else:
        conditions.append((False, 0.0, ""))

    # 4. Bullish FVG (unfilled)
    fvg_bull = row.get("fvg_bullish", 0.0)
    fvg_fill = row.get("fvg_fill_pct", 1.0)
    if fvg_bull > 0.5 and fvg_fill < 0.3:
        conditions.append((
            True, 0.03,
            f"Unfilled bullish Fair Value Gap ({1 - fvg_fill:.0%} remaining)",
        ))
    else:
        conditions.append((False, 0.0, ""))

    # 5. Bullish order block
    ob_bull_zone = row.get("ob_in_bullish_zone", 0.0)
    ob_bull_near = row.get("ob_bullish_near", 0.0)
    if ob_bull_zone > 0.5 or ob_bull_near > 0.5:
        conditions.append((True, 0.03, "In/near bullish order block zone"))
    else:
        conditions.append((False, 0.0, ""))

    # 6. Bullish BOS (within last 5 bars)
    if bos_window.get(row.get("_bar_index", 0), 0) == 1:
        conditions.append((True, 0.03, "Bullish Break of Structure confirmed"))
    else:
        conditions.append((False, 0.0, ""))

    # 7. RSI oversold
    rsi = row.get("rsi_14", 50.0)
    if rsi < 35:
        conditions.append((True, 0.03, f"RSI oversold ({rsi:.1f}) - potential reversal"))
    else:
        conditions.append((False, 0.0, ""))

    # 8. Trending regime
    regime_trend = row.get("regime_prob_trending", 0.0)
    if regime_trend > 0.5:
        conditions.append((
            True, 0.02,
            f"Trending regime (prob={regime_trend:.0%}) - momentum supported",
        ))
    else:
        conditions.append((False, 0.0, ""))

    return conditions


def _check_sell_conditions(
    row: dict,
    bos_window: dict[int, int],
) -> list[tuple[bool, float, str]]:
    """Evaluate SELL corroboration conditions for a single bar.

    Mirror of buy conditions for bearish direction.
    """
    conditions = []

    # 1. EW impulse exhaustion (wave 3 or 5 in bearish impulse)
    ew_pos = row.get("ew_wave_position", 0.0)
    ew_dir = row.get("ew_wave_direction", 0.0)
    ew_imp = row.get("ew_is_impulse", 0.0)
    if ew_imp > 0.5 and ew_dir < -0.5 and ew_pos in (3.0, 5.0):
        conditions.append((
            True, 0.04,
            f"In bearish impulse wave W{int(ew_pos)} - selling pressure",
        ))
    else:
        conditions.append((False, 0.0, ""))

    # 2. EW impulse ending
    ew_comp = row.get("ew_completion_prob", 0.0)
    if ew_comp > 0.6 and ew_imp > 0.5 and ew_dir < 0.5:
        conditions.append((
            True, 0.04,
            f"Impulse wave ending (prob={ew_comp:.0%}) - reversal risk",
        ))
    else:
        conditions.append((False, 0.0, ""))

    # 3. Fib resistance (near 23.6% or 38.2% retracement)
    fib_236 = abs(row.get("fib_dist_236", 99.0))
    fib_382 = abs(row.get("fib_dist_382", 99.0))
    if fib_382 < 0.5 or fib_236 < 0.5:
        conditions.append((True, 0.03, "Near Fibonacci resistance"))
    else:
        conditions.append((False, 0.0, ""))

    # 4. Bearish FVG (unfilled)
    fvg_bear = row.get("fvg_bearish", 0.0)
    fvg_fill = row.get("fvg_fill_pct", 1.0)
    if fvg_bear > 0.5 and fvg_fill < 0.3:
        conditions.append((
            True, 0.03,
            f"Unfilled bearish Fair Value Gap ({1 - fvg_fill:.0%} remaining)",
        ))
    else:
        conditions.append((False, 0.0, ""))

    # 5. Bearish order block
    ob_bear_zone = row.get("ob_in_bearish_zone", 0.0)
    ob_bear_near = row.get("ob_bearish_near", 0.0)
    if ob_bear_zone > 0.5 or ob_bear_near > 0.5:
        conditions.append((True, 0.03, "In/near bearish order block zone"))
    else:
        conditions.append((False, 0.0, ""))

    # 6. Bearish BOS
    if bos_window.get(row.get("_bar_index", 0), 0) == -1:
        conditions.append((True, 0.03, "Bearish Break of Structure confirmed"))
    else:
        conditions.append((False, 0.0, ""))

    # 7. RSI overbought
    rsi = row.get("rsi_14", 50.0)
    if rsi > 65:
        conditions.append((True, 0.03, f"RSI overbought ({rsi:.1f}) - potential reversal"))
    else:
        conditions.append((False, 0.0, ""))

    # 8. Risk regime
    regime_stressed = row.get("regime_prob_stressed", 0.0)
    regime_volatile = row.get("regime_prob_volatile", 0.0)
    if regime_stressed > 0.3 or regime_volatile > 0.4:
        conditions.append((
            True, 0.02,
            f"Stressed/volatile regime - elevated risk",
        ))
    else:
        conditions.append((False, 0.0, ""))

    return conditions


def detect_sweet_spots(
    df: pd.DataFrame,
    *,
    model_predictions: np.ndarray | None = None,
    model_probas: np.ndarray | None = None,
    model_name: str = "lr",
    timeframe: str = "1w",
    threshold: float = DEFAULT_THRESHOLD,
    heuristic_mode: bool = False,
) -> list[SweetSpot]:
    """Detect buy/sell sweet spots from feature-augmented data.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-augmented OHLCV data (output of FeaturePipeline).
    model_predictions : np.ndarray or None
        Model direction predictions per bar (0=down, 1=flat, 2=up).
        Required unless heuristic_mode=True.
    model_probas : np.ndarray or None
        Model direction probabilities per bar, shape (N, 3).
        Required unless heuristic_mode=True.
    model_name : str
        Name of the model for provenance ("lr" or "tree").
    timeframe : str
        Timeframe name for timing horizon and deduplication.
    threshold : float
        Minimum model_confidence to fire a sweet spot.
    heuristic_mode : bool
        If True, use base confidence of 0.50 instead of model predictions.

    Returns
    -------
    list[SweetSpot]
        Detected sweet spots, sorted by bar_index, deduplicated.
    """
    n = len(df)
    if n == 0:
        return []

    # Pre-compute BOS window (direction within last 5 bars)
    bos_direction = df.get("bos", pd.Series([0] * n, dtype=int))
    bos_window: dict[int, int] = {}
    for i in range(n):
        recent = bos_direction.iloc[max(0, i - 4): i + 1]
        if (recent == 1).any():
            bos_window[i] = 1
        elif (recent == -1).any():
            bos_window[i] = -1
        else:
            bos_window[i] = 0

    lookback = _DEDUP_LOOKBACK.get(timeframe, 5)
    timing = _TIMING_HORIZON.get(timeframe, "medium-term")

    raw_spots: list[SweetSpot] = []

    for i in range(n):
        row_dict = df.iloc[i].to_dict()
        row_dict["_bar_index"] = i

        # Determine base confidence and direction
        if heuristic_mode:
            # Heuristic mode: use structural signals only
            ew_dir = row_dict.get("ew_wave_direction", 0.0)
            ew_imp = row_dict.get("ew_is_impulse", 0.0)
            if ew_imp > 0.5 and ew_dir > 0.5:
                model_dir_class = 2  # UP
                base_confidence = 0.50
            elif ew_imp > 0.5 and ew_dir < -0.5:
                model_dir_class = 0  # DOWN
                base_confidence = 0.50
            else:
                continue  # Skip flat/unclear in heuristic mode
        else:
            if model_predictions is None or model_probas is None:
                continue
            model_dir_class = int(model_predictions[i])
            proba = model_probas[i]
            # Handle variable-length probability arrays robustly
            if hasattr(proba, "__len__") and len(proba) >= 3:
                base_confidence = float(proba[model_dir_class])
            elif hasattr(proba, "__len__") and len(proba) > 0:
                # Fewer than 3 classes — map: 0=down, last=up, rest=flat
                if model_dir_class >= len(proba):
                    base_confidence = float(proba[-1]) if model_dir_class >= 2 else float(proba[0])
                else:
                    base_confidence = float(proba[model_dir_class])
            else:
                base_confidence = float(proba) if np.isscalar(proba) else 0.33

        # Skip flat predictions
        if model_dir_class == 1:
            continue

        direction: Literal["BUY", "SELL"] = "BUY" if model_dir_class == 2 else "SELL"

        # Evaluate corroboration conditions
        if direction == "BUY":
            conditions = _check_buy_conditions(row_dict, bos_window)
        else:
            conditions = _check_sell_conditions(row_dict, bos_window)

        # Calculate combined score
        adjustments = []
        justifications = []
        corroboration = []
        total_adj = 0.0

        for met, adj, just in conditions:
            if met and adj > 0:
                total_adj += adj
                justifications.append(just)
                corroboration.append((just.split(" - ")[0] if " - " in just else just, adj))

        # Bound total adjustment to +/-0.25
        total_adj = min(total_adj, 0.25)
        combined_score = min(1.0, base_confidence + total_adj)

        # Firing criteria
        if base_confidence < threshold:
            continue
        if combined_score < base_confidence + 0.03:
            continue  # Need at least one meaningful corroboration
        if len(justifications) < 1:
            continue

        ts_val = df.iloc[i].get("ts", df.index[i])
        if isinstance(ts_val, pd.Timestamp):
            ts_val = ts_val.to_pydatetime()

        raw_spots.append(SweetSpot(
            bar_index=i,
            timestamp=ts_val,
            price=float(row_dict.get("close", 0)),
            direction=direction,
            model_confidence=base_confidence,
            model_direction=model_dir_class,
            combined_score=combined_score,
            timing_horizon=timing,
            justifications=tuple(justifications),
            corroboration=tuple(corroboration),
        ))

    # Deduplicate: keep highest combined_score in each lookback window per direction
    deduped: list[SweetSpot] = []
    last_buy_idx = -lookback - 1
    last_sell_idx = -lookback - 1

    for spot in sorted(raw_spots, key=lambda s: -s.combined_score):
        if spot.direction == "BUY":
            if spot.bar_index - last_buy_idx >= lookback:
                deduped.append(spot)
                last_buy_idx = spot.bar_index
        else:
            if spot.bar_index - last_sell_idx >= lookback:
                deduped.append(spot)
                last_sell_idx = spot.bar_index

    deduped.sort(key=lambda s: s.bar_index)
    return deduped