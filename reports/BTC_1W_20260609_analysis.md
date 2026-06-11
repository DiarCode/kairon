# BTC Weekly Technical Analysis Report

**Date:** 2026-06-09 07:13 UTC
**Current Price:** $63,284.70
**Weekly Change:** +0.07%
**Data Range:** 2020-03-30 to 2026-06-08 (324 bars)

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Price** | $63,284.70 |
| **Directional Bias** | **BEARISH** |
| **Confidence** | 0% Fib confluence |
| **Regime** | trending (trending prob: 70%) |
| **Hurst Exponent** | 0.667 (trending) |
| **EW Position** | W4 (Impulse) |
| **EW Direction** | Bearish ↓ |
| **GARCH Volatility** | 0.0593 |
| **ATR (14)** | $7,136.77 |

**Justification:** In impulse wave W4 with bearish direction. Hurst exponent (0.67) suggests the market is currently in a trending regime, which partially contradicts the directional bias.

---

## 2. Elliott Wave Analysis

| Wave Feature | Value |
|-------------|-------|
| **Current Wave Position** | W4 |
| **Wave Direction** | Bearish (-1) |
| **Is Impulse?** | Yes |
| **Fibonacci Confluence** | 0.00% |
| **Wave Completion Prob** | 13.75% |
| **Retracement Depth** | 0.08 |

### Key Pivot Points (Last 8 Zigzag Pivots)

- **LOW** at $98,286.21 (2025-06-16)
- **HIGH** at $124,457.12 (2025-08-11)
- **LOW** at $107,271.18 (2025-09-01)
- **HIGH** at $126,198.07 (2025-10-06)
- **LOW** at $80,659.81 (2025-11-17)
- **HIGH** at $97,860.60 (2026-01-12)
- **LOW** at $60,074.20 (2026-02-02)
- **HIGH** at $82,792.21 (2026-05-04)

### Wave Interpretation

- Currently in **Wave 4** (corrective wave within impulse) — expect relief selling after this correction completes

---

## 3. Regime Analysis

| Regime Feature | Value |
|---------------|-------|
| **Dominant Regime** | trending |
| **Trending Probability** | 70.0% |
| **Ranging Probability** | 30.0% |
| **Volatile Probability** | 0.0% |
| **Stressed Probability** | 0.0% |
| **Run Length (mean)** | 125.9 bars |
| **Run Length (MAP)** | 2 bars |

- **Trending regime**: Momentum strategies (trend following) are likely to perform well.

---

## 4. Volatility Assessment

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **GARCH Vol** | 0.0593 | High volatility |
| **ATR (14)** | $7,136.77 | Wide ranges |
| **Hurst Exponent** | 0.667 | Trending (H > 0.5) |
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
| **In Bullish OB Zone** | Yes |
| **In Bearish OB Zone** | No |

- 🟢 Price is **inside a bullish order block zone** — potential support area.
- ⚡ Fair Value Gap is largely **unfilled** (100% remaining) — price may be attracted to fill it.

---

## 6. Model Predictions

| Model | Direction | Confidence | Notes |
|-------|-----------|------------|-------|
| **LR** | DOWN ↓ | 75.3% | Logistic Regression baseline |
| **Tree** | UP ↑ | 54.1% | RandomForest/gradient boosted |

**Consensus:** **DISAGREEMENT** — models diverge, reduce confidence

---

## 7. Trading Levels

### Stop Loss Levels

| Level | Price | Basis |
|-------|-------|-------|
| **SL Long** | $49,011.15 | 2× ATR below current |
| **SL Long (tight)** | $52,579.54 | 1.5× ATR below current |
| **SL Short** | $77,558.24 | 2× ATR above current |
| **SL Short (tight)** | $73,989.85 | 1.5× ATR above current |

### Take Profit Levels

| Level | Price | Basis |
|-------|-------|-------|
| **TP Long 1** | $77,558.24 | 2x ATR above (1:1 R:R for longs) |
| **TP Long 2** | $84,695.01 | 3x ATR above (1:1.5 R:R for longs) |
| **TP Long Fib** | $119,549.95 | Fib 1.618 extension above |
| **TP Short 1** | $49,011.15 | 2x ATR below (1:1 R:R for shorts) |
| **TP Short 2** | $41,874.38 | 3x ATR below (1:1.5 R:R for shorts) |
| **TP Short Fib** | $46,034.47 | Fib 1.618 extension below |

### Key Fibonacci Levels (from recent swing)


---

## 8. Risk Assessment

### Data Quality
- **Dataset size:** 324 weekly bars
- **Coverage:** 2020-03-30 to 2026-06-08
- **Volume data:** NOT available (filled with 0)
- **Missing values:** Any NaN/inf in features are replaced with 0

### Model Reliability Caveats
- ⚠️ **Weekly data has limited sample size** (324 bars). Model accuracy is approximately 50-60% on this timeframe.
- ⚠️ **Walk-forward CV shows significant variance** across folds (std ~3-7%).
- ⚠️ **Tree-based models may overfit** on weekly data due to high feature-to-sample ratio (71 features / 324 bars).
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

1. **Elliott Wave:** Currently in W4 (impulse), direction is bearish.
   - Fibonacci confluence: 0% — weak alignment with key Fib levels.

2. **Hurst Exponent (0.67):** Above 0.5 — trending behavior expected to continue.

3. **BOCPD Regime:** Currently in trending regime with 70% trending probability.
   - Trend-following strategies are appropriate.

4. **Model consensus:** **DISAGREEMENT**.

5. **Smart Money:** Price near bullish order block — support zone.

---

*Report generated by Kairon Technical Analysis Engine*
*Model: LR MultiHead + TreeMultiHead (RandomForest fallback)*
*Features: 83 engineered features from ALL_FEATURES pipeline*
