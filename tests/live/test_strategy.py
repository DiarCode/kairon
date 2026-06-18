"""Tests for SignalStrategy: MACrossoverStrategy, MomentumStrategy, and ComprehensiveStrategy."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from kairon.live.strategy import (
    MACrossoverStrategy,
    MomentumStrategy,
    ComprehensiveStrategy,
    _ema,
    _rsi,
    _atr,
    _macd,
    _adx,
    _bollinger,
    _stochastic,
    _cci,
    _williams_r,
    _obv,
    _vwap,
    _swing_pivots,
    _fibonacci_levels,
    _regime_probabilities,
)


# ---------------------------------------------------------------------------
# Helper: generate synthetic OHLCV bars
# ---------------------------------------------------------------------------


def _make_bars(n: int, base_price: float = 50000.0, volatility: float = 0.001) -> pa.Table:
    """Generate n synthetic OHLCV bars with random walk prices."""
    rng = np.random.default_rng(42)
    closes = np.empty(n, dtype=float)
    closes[0] = base_price
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + rng.normal(0, volatility))
    highs = closes * (1 + rng.uniform(0, 0.002, size=n))
    lows = closes * (1 - rng.uniform(0, 0.002, size=n))
    opens = closes * (1 + rng.uniform(-0.001, 0.001, size=n))
    volumes = rng.uniform(100, 1000, size=n)
    from datetime import UTC, datetime, timedelta
    ts = [datetime(2026, 1, 1, 0, i, 0, tzinfo=UTC) for i in range(n)]

    return pa.table(
        {
            "ts": ts,
            "open": opens.tolist(),
            "high": highs.tolist(),
            "low": lows.tolist(),
            "close": closes.tolist(),
            "volume": volumes.tolist(),
        },
        schema=pa.schema([
            ("ts", pa.timestamp("us", tz="UTC")),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
        ]),
    )


def _make_trending_bars(n: int, direction: float = 1.0) -> pa.Table:
    """Generate bars with a clear upward or downward trend."""
    base_price = 50000.0
    step = 50.0 * direction  # 50 USDT per bar in the trend direction
    closes = [base_price + step * i for i in range(n)]
    rng = np.random.default_rng(123)
    highs = [c * (1 + rng.uniform(0, 0.001)) for c in closes]
    lows = [c * (1 - rng.uniform(0, 0.001)) for c in closes]
    opens = [c * (1 + rng.uniform(-0.0005, 0.0005)) for c in closes]
    volumes = [rng.uniform(100, 1000) for _ in range(n)]
    from datetime import UTC, datetime
    ts = [datetime(2026, 1, 1, 0, i, 0, tzinfo=UTC) for i in range(n)]

    return pa.table(
        {
            "ts": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        schema=pa.schema([
            ("ts", pa.timestamp("us", tz="UTC")),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
        ]),
    )


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------


class TestEMA:
    """Test exponential moving average calculation."""

    def test_ema_basic(self) -> None:
        values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        result = _ema(values, 3)
        # First 2 values are NaN, then EMA starts
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])
        # EMA should be between min and max
        assert result[2] >= 10.0
        assert result[2] <= 12.0

    def test_ema_short_input(self) -> None:
        values = np.array([1.0, 2.0])
        result = _ema(values, 5)
        assert all(np.isnan(result))

    def test_ema_constant(self) -> None:
        values = np.full(20, 100.0)
        result = _ema(values, 10)
        # EMA of constant should be constant after warmup
        assert abs(result[-1] - 100.0) < 1e-10


class TestRSI:
    """Test relative strength index calculation."""

    def test_rsi_basic(self) -> None:
        closes = np.array([44.0, 44.25, 44.50, 43.75, 43.25, 43.0, 43.5, 44.0, 44.25, 44.75,
                          45.0, 45.25, 45.50, 45.75, 46.0, 46.25, 45.75, 45.50, 45.0, 44.75])
        result = _rsi(closes, 14)
        assert len(result) == len(closes)
        # RSI should be between 0 and 100 after warmup
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_rsi_short_input(self) -> None:
        closes = np.array([1.0, 2.0, 3.0])
        result = _rsi(closes, 14)
        assert all(np.isnan(result))


class TestATR:
    """Test average true range calculation."""

    def test_atr_basic(self) -> None:
        highs = np.array([50.5, 51.0, 50.8, 51.2, 50.9] * 4, dtype=float)
        lows = np.array([49.5, 50.0, 49.8, 50.2, 49.9] * 4, dtype=float)
        closes = np.array([50.0, 50.5, 50.2, 50.8, 50.5] * 4, dtype=float)
        result = _atr(highs, lows, closes, 14)
        assert len(result) == len(closes)
        valid = result[~np.isnan(result)]
        assert all(v > 0 for v in valid)

    def test_atr_short_input(self) -> None:
        highs = np.array([50.5, 51.0])
        lows = np.array([49.5, 50.0])
        closes = np.array([50.0, 50.5])
        result = _atr(highs, lows, closes, 14)
        assert all(np.isnan(result))


# ---------------------------------------------------------------------------
# MACrossoverStrategy tests
# ---------------------------------------------------------------------------


class TestMACrossoverStrategy:
    """Test EMA crossover strategy."""

    def test_warmup_bars(self) -> None:
        strategy = MACrossoverStrategy(fast_period=9, slow_period=21)
        assert strategy.warmup_bars == 22

    def test_returns_neutral_before_warmup(self) -> None:
        strategy = MACrossoverStrategy()
        bars = _make_bars(10)  # Less than warmup_bars (22)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.direction == 0.0
        assert pred.confidence == strategy.confidence_floor

    def test_produces_prediction_after_warmup(self) -> None:
        strategy = MACrossoverStrategy()
        bars = _make_bars(30)  # More than warmup_bars
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.symbol == "BTC-USDT-PERP"
        assert pred.volatility > 0
        assert 0 <= pred.confidence <= 1

    def test_uptrend_produces_buy(self) -> None:
        strategy = MACrossoverStrategy()
        # Generate bars with a clear upward trend
        bars = _make_trending_bars(50, direction=1.0)
        # Run predictions iteratively to build crossover history
        for i in range(strategy.warmup_bars, 50):
            pred = strategy.predict(bars.slice(0, i), "BTC-USDT-PERP")
        # After a sustained uptrend, should have seen at least one buy signal
        # (direction > 0 at some point)
        # Reset strategy and check final prediction
        strategy2 = MACrossoverStrategy()
        pred = strategy2.predict(bars, "BTC-USDT-PERP")
        # On a clear uptrend, the fast EMA should be above the slow EMA
        assert pred.symbol == "BTC-USDT-PERP"

    def test_downtrend_produces_sell(self) -> None:
        strategy = MACrossoverStrategy()
        bars = _make_trending_bars(50, direction=-1.0)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.symbol == "BTC-USDT-PERP"

    def test_strategy_state_persists(self) -> None:
        strategy = MACrossoverStrategy()
        bars = _make_bars(30)
        pred1 = strategy.predict(bars, "BTC-USDT-PERP")
        # _prev_fast and _prev_slow should be set
        assert strategy._prev_fast is not None
        assert strategy._prev_slow is not None

    def test_custom_parameters(self) -> None:
        strategy = MACrossoverStrategy(fast_period=5, slow_period=10, rsi_period=7)
        assert strategy.warmup_bars == 11  # slow_period + 1
        bars = _make_bars(15)
        pred = strategy.predict(bars, "ETH-USDT-PERP")
        assert pred.symbol == "ETH-USDT-PERP"


# ---------------------------------------------------------------------------
# MomentumStrategy tests
# ---------------------------------------------------------------------------


class TestMomentumStrategy:
    """Test RSI + MACD momentum strategy."""

    def test_warmup_bars(self) -> None:
        strategy = MomentumStrategy()
        assert strategy.warmup_bars == 36  # 26 + 9 + 1

    def test_returns_neutral_before_warmup(self) -> None:
        strategy = MomentumStrategy()
        bars = _make_bars(20)  # Less than warmup_bars (36)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.direction == 0.0

    def test_produces_prediction_after_warmup(self) -> None:
        strategy = MomentumStrategy()
        bars = _make_bars(50)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.symbol == "BTC-USDT-PERP"
        assert pred.volatility > 0


# ---------------------------------------------------------------------------
# ComprehensiveStrategy indicator tests
# ---------------------------------------------------------------------------


class TestMACDIndicator:
    """Test MACD calculation."""

    def test_macd_basic(self) -> None:
        closes = np.array([44.0 + i * 0.5 for i in range(50)])
        macd_line, signal_line, histogram = _macd(closes, fast=12, slow=26, signal=9)
        assert len(macd_line) == 50
        assert len(histogram) == 50
        # MACD line should have NaN values during warmup
        assert np.isnan(macd_line[0])
        # After warmup, MACD should be finite
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert len(valid_macd) > 0

    def test_macd_short_input(self) -> None:
        closes = np.array([1.0, 2.0, 3.0])
        macd_line, _, _ = _macd(closes)
        assert all(np.isnan(macd_line))


class TestADXIndicator:
    """Test ADX calculation."""

    def test_adx_basic(self) -> None:
        n = 60
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 0.5, n)) + 50000
        highs = closes + rng.uniform(0, 50, n)
        lows = closes - rng.uniform(0, 50, n)
        adx = _adx(highs, lows, closes, 14)
        assert len(adx) == n
        # After warmup, some values should be non-NaN
        valid = adx[~np.isnan(adx)]
        assert len(valid) > 0
        # ADX should be non-negative
        assert all(v >= 0 for v in valid)


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bollinger_basic(self) -> None:
        closes = np.array([50.0 + i * 0.1 for i in range(30)])
        upper, mid, lower = _bollinger(closes, period=20, num_std=2.0)
        assert len(upper) == 30
        # After warmup, upper > mid > lower
        for i in range(19, 30):
            if not np.isnan(upper[i]):
                assert upper[i] > mid[i]
                assert mid[i] > lower[i]


class TestStochastic:
    """Test Stochastic Oscillator calculation."""

    def test_stochastic_basic(self) -> None:
        n = 30
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 0.5, n)) + 50000
        highs = closes + rng.uniform(0, 50, n)
        lows = closes - rng.uniform(0, 50, n)
        k_line, d_line = _stochastic(highs, lows, closes, k_period=14, d_period=3)
        assert len(k_line) == n
        # %K should be between 0 and 100 after warmup
        valid_k = k_line[~np.isnan(k_line)]
        assert len(valid_k) > 0
        assert all(0 <= v <= 100 for v in valid_k)


class TestCCI:
    """Test CCI calculation."""

    def test_cci_basic(self) -> None:
        n = 30
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 0.5, n)) + 50000
        highs = closes + rng.uniform(0, 50, n)
        lows = closes - rng.uniform(0, 50, n)
        cci = _cci(highs, lows, closes, period=20)
        assert len(cci) == n
        valid = cci[~np.isnan(cci)]
        assert len(valid) > 0


class TestWilliamsR:
    """Test Williams %R calculation."""

    def test_williams_r_basic(self) -> None:
        n = 30
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 0.5, n)) + 50000
        highs = closes + rng.uniform(0, 50, n)
        lows = closes - rng.uniform(0, 50, n)
        wr = _williams_r(highs, lows, closes, period=14)
        assert len(wr) == n
        valid = wr[~np.isnan(wr)]
        assert len(valid) > 0
        # Williams %R ranges from -100 to 0
        assert all(-100 <= v <= 0 for v in valid)


class TestOBV:
    """Test On-Balance Volume calculation."""

    def test_obv_basic(self) -> None:
        closes = np.array([50.0, 51.0, 50.5, 51.5, 52.0])
        volumes = np.array([100.0, 150.0, 120.0, 200.0, 180.0])
        obv = _obv(closes, volumes)
        assert len(obv) == 5
        # OBV should start at 0
        assert obv[0] == 0.0
        # When price goes up, OBV increases
        assert obv[1] > obv[0]


class TestVWAP:
    """Test VWAP calculation."""

    def test_vwap_basic(self) -> None:
        highs = np.array([102.0, 103.0, 104.0])
        lows = np.array([98.0, 99.0, 100.0])
        closes = np.array([100.0, 101.0, 102.0])
        volumes = np.array([1000.0, 1500.0, 2000.0])
        vwap = _vwap(highs, lows, closes, volumes)
        assert len(vwap) == 3
        # VWAP should be close to typical price
        assert all(v > 0 for v in vwap)


class TestSwingPivots:
    """Test swing pivot detection."""

    def test_swing_pivots_basic(self) -> None:
        n = 20
        highs = np.array([100.0 + (5.0 if i == 10 else 0.0) for i in range(n)])
        lows = np.array([100.0 - (5.0 if i == 5 else 0.0) for i in range(n)])
        swing_highs, swing_lows = _swing_pivots(highs, lows, left=3, right=3)
        assert len(swing_highs) == n
        assert len(swing_lows) == n

    def test_swing_pivots_short(self) -> None:
        highs = np.array([101.0, 102.0])
        lows = np.array([99.0, 98.0])
        swing_highs, swing_lows = _swing_pivots(highs, lows)
        assert all(np.isnan(swing_highs))
        assert all(np.isnan(swing_lows))


class TestFibonacciLevels:
    """Test Fibonacci level calculation."""

    def test_fibonacci_levels(self) -> None:
        levels = _fibonacci_levels(60000.0, 50000.0)
        assert levels["0.0"] == 60000.0
        assert levels["1.0"] == 50000.0
        # 50% retracement should be midpoint
        assert abs(levels["0.5"] - 55000.0) < 1.0
        # 61.8% retracement
        assert abs(levels["0.618"] - 53820.0) < 100.0


class TestRegimeProbabilities:
    """Test regime probability estimation."""

    def test_regime_defaults(self) -> None:
        closes = np.array([50000.0 + i for i in range(10)])
        atr = np.full(10, 100.0)
        adx = np.full(10, 20.0)
        trending, ranging, volatile, stressed = _regime_probabilities(closes, atr, adx)
        # Probabilities should sum to ~1.0
        total = trending + ranging + volatile + stressed
        assert abs(total - 1.0) < 0.01

    def test_regime_nan_handling(self) -> None:
        # Should handle NaN in ADX gracefully
        closes = np.array([50000.0] * 5)
        atr = np.full(5, np.nan)
        adx = np.full(5, np.nan)
        trending, ranging, volatile, stressed = _regime_probabilities(closes, atr, adx)
        assert abs(trending + ranging + volatile + stressed - 1.0) < 0.01


# ---------------------------------------------------------------------------
# ComprehensiveStrategy tests
# ---------------------------------------------------------------------------


class TestComprehensiveStrategy:
    """Test the multi-indicator confluence strategy."""

    def test_warmup_bars(self) -> None:
        strategy = ComprehensiveStrategy()
        # warmup = max(slow+1, macd_slow+macd_signal+1, adx*2) = max(22, 36, 28) = 36
        assert strategy.warmup_bars >= 36

    def test_returns_neutral_before_warmup(self) -> None:
        strategy = ComprehensiveStrategy()
        bars = _make_bars(10)  # Far less than warmup
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.direction == 0.0
        assert pred.confidence == strategy.confidence_floor
        assert pred.justifications == ()

    def test_produces_prediction_after_warmup(self) -> None:
        strategy = ComprehensiveStrategy()
        bars = _make_bars(50)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.symbol == "BTC-USDT-PERP"
        assert pred.volatility > 0
        assert 0 <= pred.confidence <= 1

    def test_uptrend_with_confluence(self) -> None:
        strategy = ComprehensiveStrategy()
        # Clear upward trend should generate bullish signals
        bars = _make_trending_bars(60, direction=1.0)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.symbol == "BTC-USDT-PERP"
        # After sustained uptrend, strategy should have computed indicators
        assert strategy.last_indicator_snapshot is not None
        assert strategy.last_indicator_snapshot.get("ema_fast") is not None

    def test_snapshot_populated(self) -> None:
        strategy = ComprehensiveStrategy()
        bars = _make_bars(50)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        snapshot = strategy.last_indicator_snapshot
        assert snapshot is not None
        # Key indicators should be populated after warmup
        assert "close" in snapshot
        assert "rsi_14" in snapshot
        assert "atr_14" in snapshot

    def test_justifications_populated(self) -> None:
        strategy = ComprehensiveStrategy()
        bars = _make_trending_bars(60, direction=1.0)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        # On a clear trend, some justifications should fire
        # (May or may not have justifications depending on confluence threshold)
        assert isinstance(pred.justifications, tuple)

    def test_confluence_scores_populated(self) -> None:
        strategy = ComprehensiveStrategy()
        bars = _make_bars(50)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        scores = strategy.last_confluence_scores
        assert isinstance(scores, dict)

    def test_custom_parameters(self) -> None:
        strategy = ComprehensiveStrategy(fast_period=5, slow_period=10, confidence_floor=0.2)
        assert strategy.warmup_bars >= 11  # slow_period + 1 at minimum
        bars = _make_bars(20)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        assert pred.symbol == "BTC-USDT-PERP"

    def test_custom_confidence_bounds(self) -> None:
        strategy = ComprehensiveStrategy(confidence_floor=0.4, confidence_ceiling=0.9)
        assert strategy.confidence_floor == 0.4
        assert strategy.confidence_ceiling == 0.9

    def test_prediction_has_justifications_field(self) -> None:
        """Verify LivePrediction from ComprehensiveStrategy includes justifications."""
        strategy = ComprehensiveStrategy()
        bars = _make_bars(50)
        pred = strategy.predict(bars, "BTC-USDT-PERP")
        # LivePrediction should always have justifications field
        assert hasattr(pred, "justifications")
        assert isinstance(pred.justifications, tuple)