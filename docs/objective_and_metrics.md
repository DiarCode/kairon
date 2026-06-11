# Objective & Metrics — Kairon

**Date:** 2026-06-05
**Purpose:** Replace the aspirational "90% prediction accuracy" with a defensible objective hierarchy.

## 1. Why raw accuracy is misleading for trading

1. **Class imbalance**: small caps, exotic FX, and short-horizon crypto are near 50/50, so 51% is meaningful. After costs, 51% is rarely tradable.
2. **Coverage vs accuracy trade-off**: you can hit 95% accuracy by predicting on 2% of bars (R-002 T=1.9). It is not a useful signal at that coverage.
3. **Asymmetric costs**: false positives and false negatives have different PnL impact — accuracy hides this.
4. **No calibration**: a 60% confident model with Brier 0.30 is more useful than a 60% accurate model with Brier 0.49.
5. **Cost-blind**: accuracy ignores spread, commission, market impact, funding, borrow.
6. **Regime blindness**: the same accuracy on a 2017 bull and a 2022 bear hides vastly different live tradability.
7. **Selection bias**: backtest overfitting, multiple testing, and survivorship inflate apparent accuracy (Bailey & López de Prado 2014; Bhand & Joshi 2026).

## 2. Kairon's objective hierarchy

### Primary economic objective
**Maximize risk-adjusted, cost-aware, walk-forward PnL on out-of-sample data, with calibrated confidence and explainable drivers.**

Specifically:
- Sharpe ratio (after costs, after slippage, after funding) ≥ 0.7 sustained across 3 consecutive walk-forward folds in 2+ regimes, with **Deflated Sharpe Ratio (DSR)** ≥ 0.95.
- Maximum drawdown ≤ 15% of equity.
- Win rate and accuracy reported only **conditional on coverage**, alongside cost-aware P&L.

### Primary metric
**Cost-aware Sharpe (CAS)** — `mean(per-trade-pnl-after-cost) / std(...)` annualized, multiplied by `sqrt(N_trades / year)`. DSR-corrected. Walk-forward validated.

### Secondary metrics
- **Brier score** for calibration of `p(up)`.
- **Expected Calibration Error (ECE)** ≤ 0.05.
- **Log-loss** ≤ baseline (LR / persistence).
- **Coverage at threshold T** (user-tunable; see R-002).
- **Direction accuracy at threshold T** (always reported with coverage).
- **Information ratio** vs a passive benchmark.
- **Tail-loss** (CVaR at 5%) after costs.
- **Calm/Sharpe-by-regime** breakdown.
- **Hit rate × avg-win / (avg-loss × commission-multiplier)** — must exceed 1.0.

### Tertiary / diagnostic metrics
- **Probability of Backtest Overfitting (PBO)** via CPCV (Bailey et al. 2017) ≤ 0.10.
- **Leakage audit score** from "Illusion of Alpha" diagnostic (Bhand & Joshi 2026) — must match the clean baseline.
- **Staleness** — how many bars since the last training refresh.
- **Drift** — PSI / KS between train and live feature distributions.

## 3. Reformulated milestone ladder

| Level | Target |
|-------|--------|
| **Baseline** | Cost-aware Sharpe 0.3 on out-of-sample, accuracy at full coverage not statistically different from LR, calibration ECE ≤ 0.10. |
| **Competitive** | Cost-aware Sharpe 0.7, accuracy 53-56% (1d equity) / 56-62% (5-min crypto with conf T) with non-zero coverage, PBO ≤ 0.20. |
| **Strong** | Cost-aware Sharpe 1.0, PBO ≤ 0.10, ECE ≤ 0.05, regime breakdown non-degrading, DSR ≥ 0.95. |
| **Breakthrough** | Cost-aware Sharpe ≥ 1.5 sustained 12 months, drawdown < 12%, paper trade before live. |

## 4. Separation of concerns

| Component | What it measures | Kairon reporting |
|-----------|------------------|------------------|
| **Label quality** | Construct validity of the target | Backtest on (a) persistence baseline, (b) close-to-close direction, (c) close-to-close magnitude, (d) close-to-high/low extremes — verify model is not just learning artifact |
| **Predictive quality** | Out-of-sample ML skill | Direction accuracy, Brier, ECE, log-loss, hit rate, all conditional on coverage |
| **Tradability** | Cost-aware P&L | Cost-aware Sharpe, net PnL, max drawdown, profit factor |
| **Execution quality** | Implementation shortfall | Slippage model vs realized, missed fills, latency |
| **Risk-adjusted** | Combined | CAS, DSR, Sortino, Calmar, Information ratio |

## 5. Multi-horizon targets

Different horizons have different achievable ceilings (per the local research):

| Horizon | Asset | Achievable direction accuracy (evidence) |
|---------|-------|------------------------------------------|
| 5-min (60 bars ahead) | Crypto | 60-75% with conf threshold (R-002) |
| 1-hour | Crypto / equity | 55-62% (industry consensus) |
| 1-day | Equity index | 55-60% (R-001, Fischer-Krauss) |
| 1-week | Equity / FX | 53-58% |

We **publish these as the realistic ceiling table** in the product and in CI dashboards; any model exceeding them is auto-flagged for leakage audit.

## 6. The honest verdict on "90%"

> **90% raw directional accuracy is not a defensible target** for any horizon, asset, or method under scientifically valid evaluation. The 90% number is achievable only on a tiny minority of bars after confidence filtering (R-002 T=1.9: 93% on 1.8% of bars) — and that is **not the same thing** as a 90%-accurate model.

Kairon replaces it with the multi-metric objective above. The product will tell users the realistic ceiling for each (asset, horizon) combination and refuse to display non-defensible accuracy numbers.

## 7. What we *will* show in the UI

For every signal:
- **Coverage** (% of bars where we have a signal)
- **Accuracy at this coverage**
- **Cost-aware Sharpe** (paper-trade)
- **Calibration** (Brier / ECE)
- **Regime context** (vol bucket, trend strength)
- **Top 3 drivers** (SHAP or attention)
- **Confidence band** (e.g., 5th-95th percentile of p(up) across ensemble)
- **"Why not higher accuracy"** honest explanation
