# Evaluation Framework — Kairon

**Date:** 2026-06-05
**Stack:** walk-forward, purged, embargo, CPCV, DSR, PBO, calibration, regime breakdown, ablations.

## 1. Splitting strategy

### 1.1 Walk-forward (the default)
- **Cadence:** train on `[t-W_train, t]`, validate on `[t+1, t+W_val]`, test on `[t+W_val+1, t+W_val+W_test]`.
- **Default windows** (configurable, but these are sane defaults):
  - 5-min crypto: train 7d, val 1d, test 1d (rolling daily).
  - 1-hour: train 90d, val 14d, test 14d.
  - 1-day: train 4y, val 6m, test 6m.
- **Anchored vs rolling:** both supported. Anchored preserves the longest history.
- **Monotonicity guarantee:** `test_fold[k].start > test_fold[k-1].end` (tested in `tests/splits/`).

### 1.2 Purging
- Remove from the train set any sample whose label overlaps the test set.
- Example: label is `close[t+H] - close[t]`; if the train sample is at `t_train` and the test starts at `t_test`, remove train samples where `t_train + label_overlap_window > t_test`.
- The `label_overlap_window` is set per-label-spec (default = horizon).

### 1.3 Embargo
- After the test set, embargo an additional `embargo_window` of train samples.
- `embargo_window` defaults to `1%` of the train window length, or the longest serial-correlation horizon in features (whichever is greater).
- Detected by autocorrelation function on the most autocorrelated feature.

### 1.4 Combinatorial Purged Cross-Validation (CPCV)
- Used **only** for the **Probability of Backtest Overfitting (PBO)** diagnostic, not for model selection.
- N folds, ~N paths; report the distribution of OOS Sharpe.

## 2. Cost model

| Cost | Source | Default |
|------|--------|---------|
| Commission | per-exchange, per-asset-class | spot 0.05%, perp 0.02% / 0.05% funding |
| Spread | snapshot at decision time, or modeled | 1 tick |
| Market impact | `impact = sigma * sqrt(order_size / ADV)` with conservative ADV | `impact_coef=0.1` |
| Slippage | 1 tick | configurable |
| Funding | 8h on perps | modeled |
| Borrow | on shorts | modeled |

All costs are in pydantic `CostModel`; the same model is used in backtest and in live inference.

## 3. Statistical metrics

| Metric | When | Threshold |
|--------|------|-----------|
| Direction accuracy | all | reported with coverage |
| Brier score | all | ≤ baseline (LR / persistence) |
| Expected Calibration Error (ECE) | all | ≤ 0.05 |
| Log-loss | all | ≤ baseline |
| Hit rate (at threshold T) | all | reported with T |
| McNemar p-value | for ensemble vs base | <0.05 desired |
| Bootstrap CI on Sharpe | all | reported |
| Permutation test (no-skill null) | for headline claims | p<0.05 |

## 4. Economic metrics

| Metric | Why |
|--------|-----|
| Cost-aware Sharpe (CAS) | primary economic metric |
| Deflated Sharpe Ratio (DSR) | corrects for multiple testing (Bailey & López de Prado 2014) |
| Sortino | downside-only Sharpe |
| Calmar | return / max DD |
| Information Ratio | vs benchmark |
| Profit factor | gross profit / gross loss |
| Expectancy per trade | mean / std |
| Hit rate × avg win / (avg loss × commission mult) | >1 required |
| Max drawdown | always reported |
| CVaR (5%) | tail loss |
| Net PnL (post-cost) | honest $ figure |

## 5. Risk metrics

| Metric | Use |
|--------|-----|
| Annualized vol | regime context |
| Beta to benchmark | if applicable |
| Correlation to benchmark | if applicable |
| Drawdown duration | pain |
| Tail ratio | 95th / 5th percentile ratio |

## 6. Operational metrics

| Metric | Use |
|--------|-----|
| Coverage at threshold T | % of bars with a signal |
| Inference latency p50/p99 | SLA |
| Data staleness | freshness |
| Calibration drift | ECE drift between folds |
| Feature drift | PSI / KS on top features |
| Retrain success rate | mlflow |
| Alert volume per day | UX |

## 7. Calibration

- **Method:** isotonic regression on a held-out calibration fold.
- **Reporting:** calibration curve (binned predicted vs observed), Brier, ECE.
- **Drift:** if `|ECE_now - ECE_baseline| > 0.03`, banner.

## 8. Robustness tests

1. **Regime segmentation:** metric broken down by {Trending, Ranging, Volatile, Stressed} — required.
2. **Asset segmentation:** metric broken down by {BTC, ETH, top 10 alts, top 10 US stocks}.
3. **Window sensitivity:** if Sharpe varies > 30% across starting points, flag.
4. **Cost sensitivity:** metric computed at 0.5×, 1×, 2×, 5× the default cost.
5. **Slippage shock:** metric with slippage × 3.
6. **Latency shock:** metric with simulated 1s / 5s / 30s execution delay.
7. **Random-k-fold leakage audit:** random k-fold should *not* beat walk-forward by a wide margin; if it does, we have leakage (Bhand & Joshi 2026 diagnostic).

## 9. Ablation framework

Every shipped model ships with a typed ablation JSON:

```json
{
  "model": "ensemble_v3",
  "asset": "BTC-USDT",
  "horizon": "1h",
  "folds": 12,
  "ablations": {
    "full":                                 {"dsr": 0.97, "pbo": 0.08, "cas": 1.1},
    "no_architecture_diversity":            {"dsr": 0.93, "pbo": 0.13, "cas": 0.9},
    "no_confidence_threshold":              {"dsr": 0.96, "pbo": 0.10, "cas": 0.8},
    "no_sentiment":                         {"dsr": 0.96, "pbo": 0.09, "cas": 1.05},
    "no_onchain":                           {"dsr": 0.97, "pbo": 0.08, "cas": 1.1},
    "no_regime_filter":                     {"dsr": 0.92, "pbo": 0.14, "cas": 0.95},
    "no_calibration":                       {"dsr": 0.96, "pbo": 0.10, "cas": 1.0},
    "no_embargo":                           {"dsr": 0.94, "pbo": 0.12, "cas": 1.05},
    "no_cost_model":                        {"dsr": 0.99, "pbo": 0.05, "cas": 2.4},
    "random_kfold_sanity":                  {"dsr": 0.99, "pbo": 0.02, "cas": 3.0, "flag": "expected_to_overfit"}
  }
}
```

CI gates: any ablation that *hurts* is documented; any that *helps* is verified for robustness.

## 10. Live / paper trading evaluation

- Paper trading for at least 30 days before any live deployment.
- Daily PnL reconciliation vs backtest.
- Drift detectors: PSI on top features, ECE on calibration, regime classifier agreement.
- A "live PnL within 1.5 std of expected" gate before graduating from paper to live.

## 11. Code-level enforcement

- `kairon.evaluation` is the only allowed evaluator.
- A custom ruff rule (or pre-commit) forbids:
  - `sklearn.model_selection.KFold` (or `ShuffleSplit`) on financial time series
  - `train_test_split` without `shuffle=False`
  - `sklearn.preprocessing.StandardScaler` fit on a frame that includes the test set
  - `numpy.random.seed` outside a seeded `Seed` context
- Leakage tests in `tests/splits/` are mandatory in CI.

## 12. Documentation

- Every backtest produces a `report.html` (quantstats) + a `report.json` (Kairon-native) with full provenance.
- The `report.json` is what the UI consumes.
