# ETH Weekly Technical Analysis Report

**Date:** 2026-06-09 07:13 UTC
**Current Price:** $1,686.62
**Weekly Change:** +0.01%
**Data Range:** 2018-05-28 to 2026-06-08 (420 bars)

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Price** | $1,686.62 |
| **Directional Bias** | **BEARISH** |
| **Confidence** | 0% Fib confluence |
| **Regime** | ranging (trending prob: 20%) |
| **Hurst Exponent** | 0.765 (trending) |
| **EW Position** | W5 (Impulse) |
| **EW Direction** | Bearish ↓ |
| **GARCH Volatility** | 0.0705 |
| **ATR (14)** | $304.54 |

**Justification:** In impulse wave W5 with bearish direction. Hurst exponent (0.76) suggests the market is currently in a trending regime, which partially contradicts the directional bias.

---

## 2. Elliott Wave Analysis

| Wave Feature | Value |
|-------------|-------|
| **Current Wave Position** | W5 |
| **Wave Direction** | Bearish (-1) |
| **Is Impulse?** | Yes |
| **Fibonacci Confluence** | 0.00% |
| **Wave Completion Prob** | 29.16% |
| **Retracement Depth** | -0.07 |

### Key Pivot Points (Last 8 Zigzag Pivots)

- **LOW** at $2,116.68 (2025-06-16)
- **HIGH** at $4,953.73 (2025-08-18)
- **LOW** at $3,829.01 (2025-09-22)
- **HIGH** at $4,755.22 (2025-10-06)
- **LOW** at $2,626.87 (2025-11-17)
- **HIGH** at $3,397.90 (2026-01-12)
- **LOW** at $1,748.63 (2026-02-02)
- **HIGH** at $2,464.78 (2026-04-13)

### Wave Interpretation

- Currently in **Wave 5** of an impulse pattern — this is typically the final wave
- Wave 5 may be weakening; watch for completion signals and reversal patterns

---

## 3. Regime Analysis

| Regime Feature | Value |
|---------------|-------|
| **Dominant Regime** | ranging |
| **Trending Probability** | 20.0% |
| **Ranging Probability** | 80.0% |
| **Volatile Probability** | 0.0% |
| **Stressed Probability** | 0.0% |
| **Run Length (mean)** | 108.2 bars |
| **Run Length (MAP)** | 108 bars |

- **Ranging regime**: Mean-reversion strategies (buy support, sell resistance) are preferred.

---

## 4. Volatility Assessment

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **GARCH Vol** | 0.0705 | High volatility |
| **ATR (14)** | $304.54 | Wide ranges |
| **Hurst Exponent** | 0.765 | Trending (H > 0.5) |
| **Bollinger Width** | Available | Within bands |

---

## 5. Smart Money Concepts

| SMC Feature | Value |
|-------------|-------|
| **Bullish FVG Active** | Yes |
| **Bearish FVG Active** | No |
| **FVG Fill %** | 0% |
| **FVG Nearest Distance** | 1.00 ATR |
| **Near Bullish OB** | Yes |
| **Near Bearish OB** | Yes |
| **In Bullish OB Zone** | No |
| **In Bearish OB Zone** | No |

- ⚡ Fair Value Gap is largely **unfilled** (100% remaining) — price may be attracted to fill it.

---

## 6. Model Predictions

| Model | Direction | Confidence | Notes |
|-------|-----------|------------|-------|
| **LR** | DOWN ↓ | 96.4% | Logistic Regression baseline |
| **Tree** | DOWN ↓ | 54.7% | RandomForest/gradient boosted |

**Consensus:** **BEARISH** — both models agree

---

## 7. Trading Levels

### Stop Loss Levels

| Level | Price | Basis |
|-------|-------|-------|
| **SL Long** | $1,077.55 | 2× ATR below current |
| **SL Long (tight)** | $1,229.82 | 1.5× ATR below current |
| **SL Short** | $2,295.69 | 2× ATR above current |
| **SL Short (tight)** | $2,143.42 | 1.5× ATR above current |

### Take Profit Levels

| Level | Price | Basis |
|-------|-------|-------|
| **TP Long 1** | $2,295.69 | 2x ATR above (1:1 R:R for longs) |
| **TP Long 2** | $2,600.23 | 3x ATR above (1:1.5 R:R for longs) |
| **TP Long Fib** | $3,623.52 | Fib 1.618 extension above |
| **TP Short 1** | $1,077.55 | 2x ATR below (1:1 R:R for shorts) |
| **TP Short 2** | $773.01 | 3x ATR below (1:1.5 R:R for shorts) |
| **TP Short Fib** | $1,306.04 | Fib 1.618 extension below |

### Key Fibonacci Levels (from recent swing)


---

## 8. Risk Assessment

### Data Quality
- **Dataset size:** 420 weekly bars
- **Coverage:** 2018-05-28 to 2026-06-08
- **Volume data:** Available
- **Missing values:** Any NaN/inf in features are replaced with 0

### Model Reliability Caveats
- ⚠️ **Weekly data has limited sample size** (420 bars). Model accuracy is approximately 50-60% on this timeframe.
- ⚠️ **Walk-forward CV shows significant variance** across folds (std ~3-7%).
- ⚠️ **Tree-based models may overfit** on weekly data due to high feature-to-sample ratio (71 features / 420 bars).
- ⚠️ **Elliott Wave detection is probabilistic** — the wave count may change as new data arrives.

### Position Sizing Guidance
- **Conservative:** Risk 0.5% of portfolio per trade (recommended for weekly timeframe)
- **Moderate:** Risk 1% of portfolio per trade
- **Aggressive:** Risk 2% of portfolio per trade (only for high-conviction setups)

### Key Risks
1. **Regime shift risk:** BOCPD changepoint detection may lag; sudden regime changes can invalidate current bias
2. **Wave count ambiguity:** Different pivot scales can produce different wave counts
3. **Model drift:** The LR model is calibrated on historical weekly data; structural breaks reduce accuracy
4. **Liquidity risk:** Weekly bars smooth intraday volatility; actual slippage may exceed ATR-based estimates

---

## 9. Key Justifications

### Why This Directional Bias?

1. **Elliott Wave:** Currently in W5 (impulse), direction is bearish.
   - Fibonacci confluence: 0% — weak alignment with key Fib levels.

2. **Hurst Exponent (0.76):** Above 0.5 — trending behavior expected to continue.

3. **BOCPD Regime:** Currently in ranging regime with 20% trending probability.
   - Mean-reversion strategies may work better.

4. **Model consensus:** **BEARISH**.

5. **Smart Money:** No active order block proximity.

---

*Report generated by Kairon Technical Analysis Engine*
*Model: LR MultiHead + TreeMultiHead (RandomForest fallback)*
*Features: 77 engineered features from ALL_FEATURES pipeline*
