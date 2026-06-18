"""Signal strategies for live trading: produce LivePrediction from bar data.

These strategies compute technical indicators (EMA, RSI, ATR) from a rolling
window of OHLCV bars and emit :class:`LivePrediction` objects that the
:class:`TradingLoop` can consume. They do not require a pre-trained model —
they use pure-numpy indicator math on the incoming bar stream.

The :class:`MACrossoverStrategy` (EMA fast/slow crossover + RSI filter) is
the default for 1-hour sessions: it produces enough signals on 1m bars to
validate the full pipeline, and its warmup period is only 22 bars.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.live.predictor import LivePrediction

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SignalStrategy(Protocol):
    """Protocol for strategies that generate predictions from bar data."""

    def predict(self, bars: pa.Table, symbol: str) -> LivePrediction:
        """Produce a prediction from a table of OHLCV bars."""
        ...

    @property
    def warmup_bars(self) -> int:
        """Number of bars needed before the strategy can produce predictions."""
        ...


# ---------------------------------------------------------------------------
# Indicator helpers (pure numpy, no external TA lib)
# ---------------------------------------------------------------------------


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    multiplier = 2.0 / (period + 1)
    result[:period - 1] = np.nan
    # Seed with SMA
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = values[i] * multiplier + result[i - 1] * (1.0 - multiplier)
    return result


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute relative strength index."""
    if len(closes) < period + 1:
        return np.full_like(closes, np.nan, dtype=float)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Use Wilder's smoothing (same as TA-Lib)
    avg_gain = np.empty(len(closes), dtype=float)
    avg_loss = np.empty(len(closes), dtype=float)
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    # Avoid divide by zero: when avg_loss is 0, RSI is 100 (all gains, no losses)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute average true range."""
    if len(closes) < period + 1:
        return np.full_like(closes, np.nan, dtype=float)
    tr = np.empty(len(closes), dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    # Wilder's smoothing for ATR
    result = np.empty_like(closes, dtype=float)
    result[:period] = np.nan
    result[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(closes)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


# ---------------------------------------------------------------------------
# MACrossoverStrategy
# ---------------------------------------------------------------------------


@dataclass
class MACrossoverStrategy:
    """EMA crossover strategy with RSI filter.

    Generates BUY signals when the fast EMA crosses above the slow EMA
    and RSI is below 70 (not overbought). Generates SELL signals when
    the fast EMA crosses below the slow EMA and RSI is above 30 (not
    oversold). When there is no crossover, the signal is flat (0.0).

    The strategy uses a rolling buffer of bars. Before ``warmup_bars``
    bars have been accumulated, it returns a neutral prediction with
    confidence 0.0.
    """

    fast_period: int = 9
    slow_period: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    confidence_floor: float = 0.3
    confidence_ceiling: float = 0.8

    _buffer: deque = field(default_factory=lambda: deque(maxlen=100), repr=False, hash=False)
    _prev_fast: float | None = field(default=None, repr=False, hash=False)
    _prev_slow: float | None = field(default=None, repr=False, hash=False)

    @property
    def warmup_bars(self) -> int:
        return self.slow_period + 1

    def predict(self, bars: pa.Table, symbol: str) -> LivePrediction:
        """Compute a LivePrediction from OHLCV bar data.

        Parameters
        ----------
        bars:
            A pyarrow Table with OHLCV_SCHEMA columns.
            Must have at least ``warmup_bars`` rows.
        symbol:
            The trading symbol (e.g. "BTC-USDT-PERP").
        """
        from datetime import UTC, datetime

        n = bars.num_rows
        if n < self.warmup_bars:
            return LivePrediction(
                symbol=symbol,
                direction=0.0,
                magnitude=0.0,
                volatility=0.01,
                confidence=self.confidence_floor,
                horizon="day",
                ts=datetime.now(UTC).isoformat(),
            )

        closes = np.array(bars.column("close").to_pylist(), dtype=float)
        highs = np.array(bars.column("high").to_pylist(), dtype=float)
        lows = np.array(bars.column("low").to_pylist(), dtype=float)

        # Compute indicators
        fast_ema = _ema(closes, self.fast_period)
        slow_ema = _ema(closes, self.slow_period)
        rsi_values = _rsi(closes, self.rsi_period)
        atr_values = _atr(highs, lows, closes, self.atr_period)

        # Current values (last bar)
        current_fast = fast_ema[-1]
        current_slow = slow_ema[-1]
        current_rsi = rsi_values[-1]
        current_atr = atr_values[-1]
        current_close = closes[-1]

        # Check for NaN
        if math.isnan(current_fast) or math.isnan(current_slow):
            return LivePrediction(
                symbol=symbol,
                direction=0.0,
                magnitude=0.0,
                volatility=0.01,
                confidence=self.confidence_floor,
                horizon="day",
                ts=datetime.now(UTC).isoformat(),
            )

        # Detect crossover
        direction = 0.0
        confidence = self.confidence_floor

        if self._prev_fast is not None and self._prev_slow is not None:
            # Bullish crossover: fast crosses above slow, RSI < 70
            if self._prev_fast <= self._prev_slow and current_fast > current_slow:
                if math.isnan(current_rsi) or current_rsi < 70:
                    direction = 1.0
                    # Confidence based on crossover strength and RSI
                    crossover_strength = abs(current_fast - current_slow) / current_close
                    rsi_bonus = (70 - (current_rsi if not math.isnan(current_rsi) else 50)) / 70
                    confidence = min(
                        self.confidence_ceiling,
                        self.confidence_floor + crossover_strength * 10 + rsi_bonus * 0.2,
                    )

            # Bearish crossover: fast crosses below slow, RSI > 30
            elif self._prev_fast >= self._prev_slow and current_fast < current_slow:
                if math.isnan(current_rsi) or current_rsi > 30:
                    direction = -1.0
                    crossover_strength = abs(current_fast - current_slow) / current_close
                    rsi_bonus = ((current_rsi if not math.isnan(current_rsi) else 50) - 30) / 70
                    confidence = min(
                        self.confidence_ceiling,
                        self.confidence_floor + crossover_strength * 10 + rsi_bonus * 0.2,
                    )

        # Update previous values for next call
        self._prev_fast = current_fast
        self._prev_slow = current_slow

        # Magnitude: normalized EMA distance
        magnitude = abs(current_fast - current_slow) / max(current_close, 1e-9)

        # Volatility: ATR as fraction of price (annualized estimate)
        volatility = (current_atr / current_close) if (not math.isnan(current_atr) and current_close > 0) else 0.01

        return LivePrediction(
            symbol=symbol,
            direction=direction,
            magnitude=magnitude,
            volatility=max(volatility, 0.001),  # Floor to avoid division by zero
            confidence=confidence,
            horizon="day",
            ts=datetime.now(UTC).isoformat(),
        )


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------


@dataclass
class MomentumStrategy:
    """Momentum strategy using RSI + MACD confirmation.

    BUY when RSI crosses above 30 (oversold bounce) AND MACD histogram > 0.
    SELL when RSI crosses below 70 (overbought reversal) AND MACD histogram < 0.
    """

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    confidence_floor: float = 0.3
    confidence_ceiling: float = 0.8

    _prev_rsi: float | None = field(default=None, repr=False, hash=False)

    @property
    def warmup_bars(self) -> int:
        return self.macd_slow + self.macd_signal + 1

    def predict(self, bars: pa.Table, symbol: str) -> LivePrediction:
        from datetime import UTC, datetime

        n = bars.num_rows
        if n < self.warmup_bars:
            return LivePrediction(
                symbol=symbol, direction=0.0, magnitude=0.0,
                volatility=0.01, confidence=self.confidence_floor,
                horizon="day", ts=datetime.now(UTC).isoformat(),
            )

        closes = np.array(bars.column("close").to_pylist(), dtype=float)
        highs = np.array(bars.column("high").to_pylist(), dtype=float)
        lows = np.array(bars.column("low").to_pylist(), dtype=float)

        ema_fast = _ema(closes, self.macd_fast)
        ema_slow = _ema(closes, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line[~np.isnan(macd_line)], self.macd_signal)
        rsi_values = _rsi(closes, self.rsi_period)
        atr_values = _atr(highs, lows, closes, self.atr_period)

        current_macd = macd_line[-1] if not np.isnan(macd_line[-1]) else 0.0
        current_signal = signal_line[-1] if len(signal_line) > 0 and not np.isnan(signal_line[-1]) else 0.0
        histogram = current_macd - current_signal
        current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0
        current_atr = atr_values[-1] if not np.isnan(atr_values[-1]) else closes[-1] * 0.01
        current_close = closes[-1]

        direction = 0.0
        confidence = self.confidence_floor

        if self._prev_rsi is not None:
            # Oversold bounce: RSI crosses above 30 with MACD confirmation
            if self._prev_rsi <= 30 and current_rsi > 30 and histogram > 0:
                direction = 1.0
                confidence = min(self.confidence_ceiling, self.confidence_floor + abs(histogram) / current_close * 10)
            # Overbought reversal: RSI crosses below 70 with MACD confirmation
            elif self._prev_rsi >= 70 and current_rsi < 70 and histogram < 0:
                direction = -1.0
                confidence = min(self.confidence_ceiling, self.confidence_floor + abs(histogram) / current_close * 10)

        self._prev_rsi = current_rsi

        magnitude = abs(histogram) / max(current_close, 1e-9)
        volatility = max(current_atr / current_close, 0.001)

        return LivePrediction(
            symbol=symbol, direction=direction, magnitude=magnitude,
            volatility=volatility, confidence=confidence,
            horizon="day", ts=datetime.now(UTC).isoformat(),
        )


__all__ = ["ComprehensiveStrategy", "MACrossoverStrategy", "MomentumStrategy", "SignalStrategy"]


# ---------------------------------------------------------------------------
# Additional indicator helpers for ComprehensiveStrategy
# ---------------------------------------------------------------------------


def _macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    valid = macd_line[~np.isnan(macd_line)]
    if len(valid) < signal:
        empty = np.full_like(closes, np.nan, dtype=float)
        return empty, empty.copy(), empty.copy()
    signal_line = _ema(valid, signal)
    # Pad signal_line to match macd_line length
    pad_len = len(macd_line) - len(signal_line)
    signal_padded = np.concatenate([np.full(pad_len, np.nan), signal_line])
    histogram = macd_line - signal_padded
    return macd_line, signal_padded, histogram


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Average Directional Index."""
    n = len(closes)
    if n < period * 2:
        return np.full(n, np.nan, dtype=float)
    # True range
    tr = np.empty(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    # Plus/minus directional movement
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0
    # Wilder's smoothing
    atr_vals = _atr(highs, lows, closes, period)
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    dx = np.full(n, np.nan, dtype=float)
    for i in range(period, n):
        if not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
            s_plus = np.mean(plus_dm[max(1, i - period + 1) : i + 1])
            s_minus = np.mean(minus_dm[max(1, i - period + 1) : i + 1])
            plus_di[i] = 100.0 * s_plus / atr_vals[i]
            minus_di[i] = 100.0 * s_minus / atr_vals[i]
            total = plus_di[i] + minus_di[i]
            if total > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / total
    # ADX = Wilder-smoothed DX
    adx = np.full(n, np.nan, dtype=float)
    first_valid = 0
    for i in range(n):
        if not np.isnan(dx[i]):
            first_valid = i
            break
    if first_valid + period >= n:
        return adx
    adx[first_valid + period - 1] = np.mean(dx[first_valid : first_valid + period])
    for i in range(first_valid + period, n):
        if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def _bollinger(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    n = len(closes)
    if n < period:
        empty = np.full(n, np.nan, dtype=float)
        return empty, empty.copy(), empty.copy()
    mid = np.full(n, np.nan, dtype=float)
    upper = np.full(n, np.nan, dtype=float)
    lower = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mid[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = mid[i] + num_std * std
        lower[i] = mid[i] - num_std * std
    return upper, mid, lower


def _stochastic(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Stochastic Oscillator (%K and %D)."""
    n = len(closes)
    k_line = np.full(n, np.nan, dtype=float)
    if n < k_period:
        d_line = np.full(n, np.nan, dtype=float)
        return k_line, d_line
    for i in range(k_period - 1, n):
        window_high = np.max(highs[i - k_period + 1 : i + 1])
        window_low = np.min(lows[i - k_period + 1 : i + 1])
        diff = window_high - window_low
        k_line[i] = ((closes[i] - window_low) / diff * 100.0) if diff > 0 else 50.0
    d_line = _ema(k_line[~np.isnan(k_line)], d_period)
    pad = n - len(d_line) - np.sum(np.isnan(k_line))
    d_padded = np.concatenate([np.full(n - len(d_line), np.nan), d_line])
    return k_line, d_padded[:n] if len(d_padded) >= n else d_padded


def _cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Compute Commodity Channel Index."""
    n = len(closes)
    if n < period:
        return np.full(n, np.nan, dtype=float)
    tp = (highs + lows + closes) / 3.0
    cci = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mean = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean))
        cci[i] = (tp[i] - mean) / (0.015 * mean_dev) if mean_dev > 0 else 0.0
    return cci


def _williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Williams %R."""
    n = len(closes)
    wr = np.full(n, np.nan, dtype=float)
    if n < period:
        return wr
    for i in range(period - 1, n):
        hh = np.max(highs[i - period + 1 : i + 1])
        ll = np.min(lows[i - period + 1 : i + 1])
        diff = hh - ll
        wr[i] = ((hh - closes[i]) / diff * -100.0) if diff > 0 else -50.0
    return wr


def _obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Compute On-Balance Volume."""
    n = len(closes)
    obv = np.zeros(n, dtype=float)
    if n < 2:
        return obv
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def _vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Compute Volume-Weighted Average Price (cumulative session VWAP)."""
    tp = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(tp * volumes)
    cum_vol = np.cumsum(volumes)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, closes)
    return vwap


def _swing_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    left: int = 5,
    right: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect swing high and swing low pivot points.

    Returns two arrays of the same length as inputs:
    - swing_highs: high values at pivot highs, NaN elsewhere
    - swing_lows: low values at pivot lows, NaN elsewhere
    """
    n = len(highs)
    swing_highs = np.full(n, np.nan, dtype=float)
    swing_lows = np.full(n, np.nan, dtype=float)
    if n < left + right + 1:
        return swing_highs, swing_lows
    for i in range(left, n - right):
        window_high = highs[i - left : i + right + 1]
        window_low = lows[i - left : i + right + 1]
        if highs[i] == np.max(window_high):
            swing_highs[i] = highs[i]
        if lows[i] == np.min(window_low):
            swing_lows[i] = lows[i]
    return swing_highs, swing_lows


def _fibonacci_levels(pivot_high: float, pivot_low: float) -> dict[str, float]:
    """Compute Fibonacci retracement levels from a pivot range.

    Returns a dict with keys: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
    mapped to price levels.
    """
    diff = pivot_high - pivot_low
    levels = {
        "0.0": pivot_high,
        "0.236": pivot_high - 0.236 * diff,
        "0.382": pivot_high - 0.382 * diff,
        "0.5": pivot_high - 0.5 * diff,
        "0.618": pivot_high - 0.618 * diff,
        "0.786": pivot_high - 0.786 * diff,
        "1.0": pivot_low,
    }
    return levels


def _regime_probabilities(
    closes: np.ndarray,
    atr: np.ndarray,
    adx: np.ndarray,
) -> tuple[float, float, float, float]:
    """Estimate regime probabilities from ATR and ADX.

    Returns (trending, ranging, volatile, stressed) probabilities.
    - Trending: ADX > 25, moderate ATR
    - Ranging: ADX < 20, low ATR
    - Volatile: ATR > 2 * median ATR
    - Stressed: ATR > 3 * median ATR AND negative returns
    """
    n = len(closes)
    if n < 2 or np.isnan(adx[-1]) or np.isnan(atr[-1]):
        return 0.25, 0.25, 0.25, 0.25

    current_adx = float(adx[-1])
    current_atr = float(atr[-1])
    valid_atr = atr[~np.isnan(atr)]
    if len(valid_atr) < 5:
        return 0.25, 0.25, 0.25, 0.25
    median_atr = float(np.median(valid_atr))
    recent_return = float((closes[-1] - closes[-2]) / closes[-2]) if closes[-2] > 0 else 0.0

    # Compute raw scores
    trending_score = min(1.0, max(0.0, (current_adx - 15) / 25))  # 0 at ADX=15, 1 at ADX=40
    ranging_score = min(1.0, max(0.0, (25 - current_adx) / 20))  # 1 at ADX=5, 0 at ADX=25

    volatility_ratio = current_atr / median_atr if median_atr > 0 else 1.0
    volatile_score = min(1.0, max(0.0, (volatility_ratio - 1.0) / 2.0))  # 0 at 1x, 1 at 3x
    stressed_score = volatile_score * (1.0 if recent_return < -0.01 else 0.3)

    # Normalize to probabilities
    total = trending_score + ranging_score + volatile_score + stressed_score
    if total < 1e-9:
        return 0.25, 0.25, 0.25, 0.25
    return (
        trending_score / total,
        ranging_score / total,
        volatile_score / total,
        stressed_score / total,
    )


# ---------------------------------------------------------------------------
# ComprehensiveStrategy
# ---------------------------------------------------------------------------


@dataclass
class ComprehensiveStrategy:
    """Multi-indicator confluence strategy for live trading.

    Computes a rich set of technical indicators and uses a weighted confluence
    scoring system to generate BUY/SELL signals. Inspired by the offline
    sweet-spot detection but adapted for real-time bar-by-bar scoring.

    Scoring breakdown:
    - Trend score (max 0.30): EMA crossover direction + ADX trend strength
    - Momentum score (max 0.25): RSI + Stochastic + MACD histogram
    - Structure score (max 0.25): Fibonacci proximity + swing pivots + BOS
    - Volume score (max 0.20): OBV direction + VWAP position + volume surge

    Direction = sign of weighted sum.
    Confidence = total confluence score, bounded [floor, ceiling].
    Justifications list every sub-condition that fired.
    """

    # EMA periods
    fast_period: int = 9
    slow_period: int = 21
    # Indicator periods
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    cci_period: int = 20
    williams_r_period: int = 14
    pivot_left: int = 5
    pivot_right: int = 5
    # Confidence bounds
    confidence_floor: float = 0.3
    confidence_ceiling: float = 0.85
    # Minimum confluence to fire a signal
    min_confluence: float = 0.15

    # Internal state
    _buffer: deque = field(default_factory=lambda: deque(maxlen=200), repr=False, hash=False)
    _prev_fast: float | None = field(default=None, repr=False, hash=False)
    _prev_slow: float | None = field(default=None, repr=False, hash=False)
    _prev_rsi: float | None = field(default=None, repr=False, hash=False)
    _prev_stoch_k: float | None = field(default=None, repr=False, hash=False)
    _last_snapshot: dict[str, float | None] | None = field(default=None, repr=False, hash=False)
    _last_justifications: tuple[str, ...] = field(default_factory=tuple, repr=False, hash=False)
    _last_confluence: dict[str, float] | None = field(default=None, repr=False, hash=False)

    @property
    def warmup_bars(self) -> int:
        return max(self.slow_period + 1, self.macd_slow + self.macd_signal + 1, self.adx_period * 2)

    @property
    def last_indicator_snapshot(self) -> dict[str, float | None]:
        """Return the most recent indicator values for journal persistence."""
        return self._last_snapshot or {}

    @property
    def last_justifications(self) -> tuple[str, ...]:
        """Return the most recent justifications for journal persistence."""
        return self._last_justifications

    @property
    def last_confluence_scores(self) -> dict[str, float]:
        """Return the most recent confluence score breakdown."""
        return self._last_confluence or {}

    def predict(self, bars: pa.Table, symbol: str) -> LivePrediction:
        """Compute a LivePrediction using multi-indicator confluence."""
        from datetime import UTC, datetime

        n = bars.num_rows
        if n < self.warmup_bars:
            return LivePrediction(
                symbol=symbol,
                direction=0.0,
                magnitude=0.0,
                volatility=0.01,
                confidence=self.confidence_floor,
                horizon="day",
                ts=datetime.now(UTC).isoformat(),
                justifications=(),
            )

        closes = np.array(bars.column("close").to_pylist(), dtype=float)
        highs = np.array(bars.column("high").to_pylist(), dtype=float)
        lows = np.array(bars.column("low").to_pylist(), dtype=float)
        volumes = np.array(bars.column("volume").to_pylist(), dtype=float)

        # ---- Compute all indicators ----
        fast_ema = _ema(closes, self.fast_period)
        slow_ema = _ema(closes, self.slow_period)
        rsi_values = _rsi(closes, self.rsi_period)
        atr_values = _atr(highs, lows, closes, self.atr_period)
        macd_line, macd_signal, macd_hist = _macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        adx_values = _adx(highs, lows, closes, self.adx_period)
        bb_upper, bb_mid, bb_lower = _bollinger(closes, self.bollinger_period, self.bollinger_std)
        stoch_k, stoch_d = _stochastic(highs, lows, closes, self.stoch_k_period, self.stoch_d_period)
        cci_values = _cci(highs, lows, closes, self.cci_period)
        wr_values = _williams_r(highs, lows, closes, self.williams_r_period)
        obv_values = _obv(closes, volumes)
        vwap_values = _vwap(highs, lows, closes, volumes)
        swing_highs, swing_lows = _swing_pivots(highs, lows, self.pivot_left, self.pivot_right)

        # Current values (last bar)
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        current_volume = volumes[-1]
        current_fast = fast_ema[-1]
        current_slow = slow_ema[-1]
        current_rsi = rsi_values[-1]
        current_atr = atr_values[-1]
        current_macd = macd_line[-1] if not np.isnan(macd_line[-1]) else 0.0
        current_signal = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0.0
        current_hist = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
        current_adx = adx_values[-1]
        current_bb_upper = bb_upper[-1] if not np.isnan(bb_upper[-1]) else None
        current_bb_mid = bb_mid[-1] if not np.isnan(bb_mid[-1]) else None
        current_bb_lower = bb_lower[-1] if not np.isnan(bb_lower[-1]) else None
        current_stoch_k = stoch_k[-1] if not np.isnan(stoch_k[-1]) else None
        current_stoch_d = stoch_d[-1] if len(stoch_d) > 0 and not np.isnan(stoch_d[-1]) else None
        current_cci = cci_values[-1] if not np.isnan(cci_values[-1]) else None
        current_wr = wr_values[-1] if not np.isnan(wr_values[-1]) else None
        current_obv = obv_values[-1]
        current_vwap = vwap_values[-1]

        # Volume vs 20-bar average
        vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_vs_avg = current_volume / vol_avg if vol_avg > 0 else 1.0

        # Regime probabilities
        regime_trending, regime_ranging, regime_volatile, regime_stressed = _regime_probabilities(
            closes, atr_values, adx_values
        )

        # Fibonacci levels from recent swing range
        valid_swings_h = swing_highs[~np.isnan(swing_highs)]
        valid_swings_l = swing_lows[~np.isnan(swing_lows)]
        pivot_high = float(valid_swings_h[-1]) if len(valid_swings_h) > 0 else current_high
        pivot_low = float(valid_swings_l[-1]) if len(valid_swings_l) > 0 else current_low
        fib_levels = _fibonacci_levels(pivot_high, pivot_low)

        # BOS: detect from recent swing pivots
        bos_direction = 0  # neutral
        if len(valid_swings_h) >= 2 and len(valid_swings_l) >= 2:
            if valid_swings_h[-1] > valid_swings_h[-2]:
                bos_direction = 1  # bullish BOS
            elif valid_swings_l[-1] < valid_swings_l[-2]:
                bos_direction = -1  # bearish BOS

        # ---- Confluence scoring ----
        justifications: list[str] = []

        def _apply_score(score: float, delta: float, label: str) -> float:
            """Apply a score delta only if it is material, recording justification."""
            if abs(delta) > 1e-6:
                score += delta
                justifications.append(label)
            return score

        # -- Trend score (max 0.30) --
        trend_score = 0.0
        ema_bullish = False
        ema_bearish = False

        # Classify the EMA trend direction from the current fast/slow relationship.
        # This must NOT depend on _prev_*: on the first prediction after warmup the
        # previous EMAs are None, but the trend direction is still knowable and must
        # contribute its vote to direction_raw. Without this, the very first
        # decision of every session loses its primary trend-following cue (the EMA
        # term in direction_raw vanishes) and mean-reversion noise can flip the sign.
        if not math.isnan(current_fast) and not math.isnan(current_slow):
            if current_fast > current_slow:
                ema_bullish = True
            elif current_fast < current_slow:
                ema_bearish = True

        if self._prev_fast is not None and self._prev_slow is not None:
            if self._prev_fast <= self._prev_slow and current_fast > current_slow:
                trend_score = _apply_score(trend_score, 0.15, "Bullish EMA crossover")
            elif self._prev_fast >= self._prev_slow and current_fast < current_slow:
                trend_score = _apply_score(trend_score, 0.15, "Bearish EMA crossover")
            elif ema_bullish:
                trend_score = _apply_score(trend_score, 0.08, "EMA trend continuation (bullish)")
            elif ema_bearish:
                trend_score = _apply_score(trend_score, 0.08, "EMA trend continuation (bearish)")
        elif ema_bullish:
            trend_score = _apply_score(trend_score, 0.08, "EMA trend continuation (bullish)")
        elif ema_bearish:
            trend_score = _apply_score(trend_score, 0.08, "EMA trend continuation (bearish)")

        if not math.isnan(current_adx):
            if current_adx > 25:
                trend_score = _apply_score(trend_score, 0.10, f"Strong trend (ADX={current_adx:.1f})")
            elif current_adx > 20:
                trend_score = _apply_score(trend_score, 0.05, f"Moderate trend (ADX={current_adx:.1f})")

        if current_bb_upper is not None and current_bb_lower is not None:
            bb_range = current_bb_upper - current_bb_lower
            if bb_range > 0:
                bb_position = (current_close - current_bb_lower) / bb_range
                if bb_position > 0.8:
                    trend_score = _apply_score(
                        trend_score, -0.05, f"Near upper Bollinger band ({bb_position:.0%})"
                    )
                elif bb_position < 0.2:
                    trend_score = _apply_score(
                        trend_score, -0.05, f"Near lower Bollinger band ({bb_position:.0%})"
                    )

        # -- Momentum score (max 0.25) --
        momentum_score = 0.0
        momentum_direction = 0.0

        if not math.isnan(current_rsi):
            if current_rsi < 30:
                momentum_score = _apply_score(momentum_score, 0.10, f"RSI oversold ({current_rsi:.1f})")
                momentum_direction += 1.0
            elif current_rsi > 70:
                momentum_score = _apply_score(momentum_score, 0.10, f"RSI overbought ({current_rsi:.1f})")
                momentum_direction -= 1.0
            elif 40 <= current_rsi <= 60:
                momentum_score = _apply_score(momentum_score, 0.02, "Neutral RSI")
            elif current_rsi < 45:
                momentum_direction += 0.3
            elif current_rsi > 55:
                momentum_direction -= 0.3

        if current_stoch_k is not None and current_stoch_d is not None:
            if not math.isnan(current_stoch_k) and not math.isnan(current_stoch_d):
                if self._prev_stoch_k is not None:
                    if (
                        self._prev_stoch_k <= current_stoch_d
                        and current_stoch_k > current_stoch_d
                        and current_stoch_k < 20
                    ):
                        momentum_score = _apply_score(
                            momentum_score, 0.08, "Stochastic bullish crossover (oversold)"
                        )
                        momentum_direction += 1.0
                    elif (
                        self._prev_stoch_k >= current_stoch_d
                        and current_stoch_k < current_stoch_d
                        and current_stoch_k > 80
                    ):
                        momentum_score = _apply_score(
                            momentum_score, 0.08, "Stochastic bearish crossover (overbought)"
                        )
                        momentum_direction -= 1.0

        if not math.isnan(current_hist):
            if current_hist > 0:
                momentum_score = _apply_score(momentum_score, 0.07, "MACD histogram positive")
                momentum_direction += 0.5
            elif current_hist < 0:
                momentum_score = _apply_score(momentum_score, 0.07, "MACD histogram negative")
                momentum_direction -= 0.5

        # -- Structure score (max 0.25) --
        structure_score = 0.0
        structure_direction = 0.0

        if current_close > 0 and pivot_high != pivot_low:
            for fib_name, fib_price in fib_levels.items():
                dist_pct = abs(current_close - fib_price) / current_close
                if dist_pct < 0.01:
                    if fib_name in ("0.618", "0.786"):
                        structure_score = _apply_score(
                            structure_score, 0.08, f"Near Fibonacci {fib_name} support"
                        )
                        structure_direction += 1.0
                    elif fib_name in ("0.236", "0.382"):
                        structure_score = _apply_score(
                            structure_score, 0.08, f"Near Fibonacci {fib_name} resistance"
                        )
                        structure_direction -= 1.0
                    elif fib_name == "0.5":
                        structure_score = _apply_score(
                            structure_score, 0.04, "Near Fibonacci 50% level"
                        )
                    break

        if len(valid_swings_l) > 0:
            nearest_support_dist = abs(current_close - float(valid_swings_l[-1])) / current_close
            if nearest_support_dist < 0.02:
                structure_score = _apply_score(
                    structure_score, 0.05, "Near swing low support"
                )
                structure_direction += 0.5

        if len(valid_swings_h) > 0:
            nearest_resist_dist = abs(current_close - float(valid_swings_h[-1])) / current_close
            if nearest_resist_dist < 0.02:
                structure_score = _apply_score(
                    structure_score, 0.05, "Near swing high resistance"
                )
                structure_direction -= 0.5

        if bos_direction == 1:
            structure_score = _apply_score(structure_score, 0.06, "Bullish Break of Structure")
            structure_direction += 1.0
        elif bos_direction == -1:
            structure_score = _apply_score(structure_score, 0.06, "Bearish Break of Structure")
            structure_direction -= 1.0

        if current_cci is not None and not math.isnan(current_cci):
            if current_cci > 100:
                structure_score = _apply_score(structure_score, 0.03, f"CCI bullish ({current_cci:.0f})")
                structure_direction += 0.3
            elif current_cci < -100:
                structure_score = _apply_score(structure_score, 0.03, f"CCI bearish ({current_cci:.0f})")
                structure_direction -= 0.3

        if current_wr is not None and not math.isnan(current_wr):
            if current_wr < -80:
                structure_score = _apply_score(
                    structure_score, 0.03, f"Williams %R oversold ({current_wr:.1f})"
                )
                structure_direction += 0.3
            elif current_wr > -20:
                structure_score = _apply_score(
                    structure_score, 0.03, f"Williams %R overbought ({current_wr:.1f})"
                )
                structure_direction -= 0.3

        # -- Volume score (max 0.20) --
        volume_score = 0.0
        volume_direction = 0.0

        if len(obv_values) >= 10:
            obv_recent = obv_values[-5:]
            obv_older = obv_values[-10:-5]
            if np.mean(obv_recent) > np.mean(obv_older):
                volume_score = _apply_score(volume_score, 0.07, "OBV rising (buying pressure)")
                volume_direction += 1.0
            elif np.mean(obv_recent) < np.mean(obv_older):
                volume_score = _apply_score(volume_score, 0.07, "OBV falling (selling pressure)")
                volume_direction -= 1.0

        if not math.isnan(current_vwap):
            if current_close > current_vwap:
                volume_score = _apply_score(volume_score, 0.06, "Price above VWAP (bullish)")
                volume_direction += 0.5
            else:
                volume_score = _apply_score(volume_score, 0.06, "Price below VWAP (bearish)")
                volume_direction -= 0.5

        if volume_vs_avg > 1.5:
            volume_score = _apply_score(
                volume_score, 0.07, f"Volume surge ({volume_vs_avg:.1f}x average)"
            )
        elif volume_vs_avg > 1.2:
            volume_score = _apply_score(
                volume_score, 0.03, f"Volume above average ({volume_vs_avg:.1f}x)"
            )

        # ---- Compute direction and confidence ----
        direction_raw = (
            (0.30 if ema_bullish else -0.30 if ema_bearish else 0.0)
            * (1.0 if trend_score > 0.05 else 0.5)
            + momentum_direction * 0.25
            + structure_direction * 0.25
            + volume_direction * 0.20
        )

        if direction_raw > 0.05:
            direction = 1.0
        elif direction_raw < -0.05:
            direction = -1.0
        else:
            direction = 0.0

        total_confluence = trend_score + momentum_score + structure_score + volume_score
        confidence = min(
            self.confidence_ceiling,
            max(self.confidence_floor, self.confidence_floor + total_confluence * 0.6),
        )

        # Quality gates
        if direction != 0.0 and (math.isnan(current_adx) or current_adx <= 22):
            justifications.append(f"ADX too weak ({current_adx if not math.isnan(current_adx) else 'nan'}); signal flattened")
            direction = 0.0
            confidence = self.confidence_floor

        if direction != 0.0 and volume_vs_avg < 0.8:
            justifications.append(f"Volume below threshold ({volume_vs_avg:.2f}x); signal flattened")
            direction = 0.0
            confidence = self.confidence_floor

        # Regime-aware confidence adjustment
        if regime_trending < 0.4 and direction != 0.0:
            confidence *= 0.7
            justifications.append("Low trending regime; confidence reduced")
        if regime_stressed > 0.3:
            direction = 0.0
            confidence = self.confidence_floor
            justifications.append("Stressed regime; signal flattened")

        # Trend alignment: refuse to trade against a strong trend.
        # Mean-reversion cues (upper Bollinger band, swing-high resistance,
        # Williams %R overbought, OBV divergence) can push direction_raw the
        # wrong way even when the trend is clearly against them — that is how
        # the strategy would short into a strong uptrend. The EMA trend direction
        # is the authoritative trend filter: a signal that disagrees with it in a
        # strong trend (ADX > 25) is flattened, and in a moderate trend
        # (ADX > 22) its confidence is capped. MACD disagreement alone (e.g. a
        # steady-slope trend where the MACD histogram ~ 0) is NOT a counter-trend
        # condition, so it only caps confidence rather than flattening.
        if direction != 0.0 and not math.isnan(current_adx):
            ema_aligned = (direction > 0 and ema_bullish) or (direction < 0 and ema_bearish)
            macd_aligned = (direction > 0 and current_hist > 0) or (direction < 0 and current_hist < 0)
            if not ema_aligned:
                if current_adx > 25:
                    justifications.append(
                        f"Counter-trend signal in strong trend (ADX={current_adx:.1f}); flattened"
                    )
                    direction = 0.0
                    confidence = self.confidence_floor
                elif current_adx > 22 and confidence > 0.6:
                    confidence = min(confidence, 0.6)
                    justifications.append("Trend alignment check capped confidence")
            elif not macd_aligned and confidence > 0.6:
                confidence = min(confidence, 0.6)
                justifications.append("Trend alignment check capped confidence")

        if total_confluence < self.min_confluence:
            direction = 0.0
            confidence = self.confidence_floor

        # Update previous values for next call
        self._prev_fast = float(current_fast) if not math.isnan(current_fast) else None
        self._prev_slow = float(current_slow) if not math.isnan(current_slow) else None
        self._prev_rsi = float(current_rsi) if not math.isnan(current_rsi) else None
        self._prev_stoch_k = (
            float(current_stoch_k)
            if current_stoch_k is not None and not math.isnan(current_stoch_k)
            else None
        )

        # Magnitude and volatility
        magnitude = (
            abs(current_fast - current_slow) / max(current_close, 1e-9)
            if not math.isnan(current_fast) and not math.isnan(current_slow)
            else 0.01
        )
        volatility = (
            max(current_atr / current_close, 0.001)
            if not math.isnan(current_atr) and current_close > 0
            else 0.01
        )

        # Swing-based SL/TP
        atr_for_sl = current_atr if not math.isnan(current_atr) else current_close * 0.01
        if direction > 0:
            sl_price = (float(valid_swings_l[-1]) if len(valid_swings_l) > 0 else current_low) - atr_for_sl
            tp_price = float(valid_swings_h[-1]) if len(valid_swings_h) > 0 else current_high
        elif direction < 0:
            sl_price = (float(valid_swings_h[-1]) if len(valid_swings_h) > 0 else current_high) + atr_for_sl
            tp_price = float(valid_swings_l[-1]) if len(valid_swings_l) > 0 else current_low
        else:
            sl_price = current_close - 2 * atr_for_sl
            tp_price = current_close + 3 * atr_for_sl

        # Store the indicator snapshot for journal persistence
        self._last_snapshot = {
            "ema_fast": float(current_fast) if not math.isnan(current_fast) else None,
            "ema_slow": float(current_slow) if not math.isnan(current_slow) else None,
            "rsi_14": float(current_rsi) if not math.isnan(current_rsi) else None,
            "atr_14": float(current_atr) if not math.isnan(current_atr) else None,
            "macd_line": float(current_macd) if not math.isnan(current_macd) else None,
            "macd_signal": float(current_signal) if not math.isnan(current_signal) else None,
            "macd_histogram": float(current_hist) if not math.isnan(current_hist) else None,
            "adx": float(current_adx) if not math.isnan(current_adx) else None,
            "bollinger_upper": float(current_bb_upper) if current_bb_upper is not None else None,
            "bollinger_mid": float(current_bb_mid) if current_bb_mid is not None else None,
            "bollinger_lower": float(current_bb_lower) if current_bb_lower is not None else None,
            "stochastic_k": float(current_stoch_k) if current_stoch_k is not None and not math.isnan(current_stoch_k) else None,
            "stochastic_d": float(current_stoch_d) if current_stoch_d is not None and not math.isnan(current_stoch_d) else None,
            "cci": float(current_cci) if current_cci is not None and not math.isnan(current_cci) else None,
            "williams_r": float(current_wr) if current_wr is not None and not math.isnan(current_wr) else None,
            "obv": float(current_obv),
            "vwap": float(current_vwap) if not math.isnan(current_vwap) else None,
            "volume_vs_avg": float(volume_vs_avg),
            "regime_prob_trending": float(regime_trending),
            "regime_prob_ranging": float(regime_ranging),
            "regime_prob_volatile": float(regime_volatile),
            "regime_prob_stressed": float(regime_stressed),
            "bos_direction": bos_direction,
            "swing_high": float(pivot_high),
            "swing_low": float(pivot_low),
            "close": float(current_close),
            "high": float(current_high),
            "low": float(current_low),
            "volume": float(current_volume),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
        }
        self._last_justifications = tuple(justifications)
        self._last_confluence = {
            "trend": round(trend_score, 4),
            "momentum": round(momentum_score, 4),
            "structure": round(structure_score, 4),
            "volume": round(volume_score, 4),
        }

        return LivePrediction(
            symbol=symbol,
            direction=direction,
            magnitude=magnitude,
            volatility=volatility,
            confidence=confidence,
            horizon="day",
            ts=datetime.now(UTC).isoformat(),
            justifications=tuple(justifications),
        )
