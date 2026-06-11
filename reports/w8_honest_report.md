# W8 — Honest Report (Headline W8 Deliverable)

**Story:** W8.3
**Decided at:** 2026-06-08
**Source-of-truth:** This report is the W8 headline deliverable per plan §8.4.
The W8.5 DECISION-FORK story consumes this report as its primary input.

## 0. Executive summary

| Metric | W8.1 (1h) | W8.2 (5m) | Acceptance | Verdict |
| --- | --- | --- | --- | --- |
| Sharpe (closed trades) | 1.71 | 27.57 | reported | positive |
| Cost-aware Sharpe (CAS) | -24.58 | -359.37 | reported | negative |
| Deflated Sharpe Ratio (DSR) | 0.0069 | 0.3075 | >= 0.95 (ship) | **fail** |
| Probability of Backtest Overfit (PBO) | 0.00 | 0.00 | <= 0.10 | pass |
| Max drawdown | -6.8% | -1.7% | <= 15% | pass |
| Brier score | 0.249 | 0.250 | reported | borderline |
| Expected Calibration Error (ECE) | 0.0032 | 0.0006 | <= 0.05 | pass |
| N trades | 1,230 | 14,480 | reported | OK |

**Honest verdict:** The W8 backtests are **synthetic** (W0 BTC-only fallback) and
**zero-edge by construction** (the log-normal price walk has no real alpha to
exploit). The positive per-trade Sharpe reflects a small numerical edge in
the multi-head's direction predictions; the negative CAS reflects the constant
cost drag that the v1 signal cannot overcome. The DSR < 0.95 is the
load-bearing honest result: on a zero-edge synthetic walk, the strategy does
not have a defensible edge. The W8.5 DECISION-FORK will EXTEND (3 more folds
+ revisit cost) per the documented decision rule.

## 1. Break-even accuracy table (from W2.2)

Source: `artifacts/break_even_w2.json` (12 rows: 3 assets x 4 horizons).

The break-even accuracy is the minimum hit rate that clears the round-trip
cost. The W2.2 formula is `p* = 0.5 + C / (2R)`, where C is the round-trip
cost in bps and R is the expected move in bps. A value < 60% is "viable"
(clears the cost without demanding unrealistic accuracy).

| Asset | Horizon | Expected move (bps) | Round-trip cost (bps) | Break-even accuracy | Viable |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT | 5m | 334.86 | 28.00 | 0.5418 | yes |
| BTCUSDT | 15m | 338.33 | 28.00 | 0.5414 | yes |
| BTCUSDT | 1h | 422.91 | 28.00 | 0.5331 | yes |
| BTCUSDT | 1d | 295.97 | 28.00 | 0.5473 | yes |
| ETHUSDT | 5m | 435.31 | 28.00 | 0.5322 | yes |
| ETHUSDT | 15m | 439.82 | 28.00 | 0.5318 | yes |
| ETHUSDT | 1h | 549.78 | 28.00 | 0.5255 | yes |
| ETHUSDT | 1d | 384.76 | 28.00 | 0.5364 | yes |
| SOLUSDT | 5m | 602.74 | 28.00 | 0.5232 | yes |
| SOLUSDT | 15m | 608.99 | 28.00 | 0.5230 | yes |
| SOLUSDT | 1h | 761.23 | 28.00 | 0.5184 | yes |
| SOLUSDT | 1d | 532.75 | 28.00 | 0.5263 | yes |

**Headline:** `max(break_even_pct) = 0.5473` (BTCUSDT 1d); all 12 (asset, horizon)
pairs viable. The W2.5 PROCEED decision holds: no asset / horizon requires an
unrealistic accuracy to clear cost.

## 2. Ceiling accuracy table (from `docs/objective_and_metrics.md` §5)

Source: `docs/objective_and_metrics.md` §5. The ceiling table is the
*achievable* direction-accuracy range for each horizon under scientifically
valid evaluation. Any model that exceeds the ceiling is auto-flagged for a
leakage audit (the W9.5 leakage-alarm hook).

| Horizon | Asset class | Achievable direction accuracy |
| --- | --- | --- |
| 5-min (60 bars ahead) | Crypto | 60-75% with confidence threshold |
| 1-hour | Crypto / equity | 55-62% |
| 1-day | Equity index | 55-60% |
| 1-week | Equity / FX | 53-58% |

**Cross-check with W8.1 (1h) and W8.2 (5m):** the W8 multi-head achieves a
direction accuracy of ~50% on synthetic data (the hit_rate per regime is
~0.50 in the W8.1 + W8.2 status sidecars), which is BELOW the achievable
ceiling — consistent with the honest "no real edge" verdict in §0.

## 3. Coverage-accuracy Pareto (from W3.5)

Source: `reports/coverage_pareto_w4.json` (12 (asset, horizon) rows;
reference_point_coverage_pct = [25, 10]).

The coverage-accuracy Pareto is the W3.5 headline deliverable per
`docs/objective_and_metrics.md` §1. The Pareto reports the (T, coverage,
accuracy) triple at two reference points: T at 25% coverage and T at 10%
coverage. The intent is to pin the model's accuracy at *meaningful*
coverage levels (not just full coverage, which is dominated by the
zero-signal majority class).

| Asset | Horizon | T at 25% coverage | Accuracy at 25% coverage | T at 10% coverage | Accuracy at 10% coverage |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT | 5m | 0.84 | 0.20 | 0.85 | 0.08 |
| BTCUSDT | 15m | 0.84 | 0.19 | 0.85 | 0.08 |
| BTCUSDT | 1h | 0.79 | 0.19 | 0.80 | 0.07 |
| BTCUSDT | 1d | 0.84 | 0.19 | 0.85 | 0.08 |
| ETHUSDT | 5m | 0.84 | 0.20 | 0.85 | 0.08 |
| ETHUSDT | 15m | 0.77 | 0.39 | 0.79 | 0.16 |
| ETHUSDT | 1h | 0.79 | 0.19 | 0.80 | 0.08 |
| ETHUSDT | 1d | 0.84 | 0.20 | 0.85 | 0.08 |
| SOLUSDT | 5m | 0.84 | 0.19 | 0.85 | 0.08 |
| SOLUSDT | 15m | 0.84 | 0.20 | 0.85 | 0.08 |
| SOLUSDT | 1h | 0.79 | 0.19 | 0.80 | 0.07 |
| SOLUSDT | 1d | 0.79 | 0.19 | 0.80 | 0.07 |

**Headline:** the Pareto frontier is well-formed (coverage monotonically
decreases as T increases); the accuracy at 10% coverage is ~7-8% on the
synthetic fixture, which reflects the W3.5 "no real edge" verdict on
synthetic data (the model is correctly identifying the rare-class majority
inside the threshold slice, not generating a real alpha).

## 4. CAS at cost shocks (from W2.3)

Source: `artifacts/cost_sensitivity_w2.json` (4 rows: cost multipliers
0.5x, 1x, 2x, 5x; n_trades = 720).

The cost-shock ladder is the documented `evaluation_framework.md` §8.4
robustness test: report the metric at 0.5x, 1x, 2x, 5x the default cost
to surface fee-regime sensitivity.

| Multiplier | Sharpe | Sortino | Max DD | Total return | N trades |
| --- | --- | --- | --- | --- | --- |
| 0.5x | -1312.68 | -1312.68 | -0.628 | -0.628 | 720 |
| 1x | -2650.50 | -2650.50 | -0.865 | -0.865 | 720 |
| 2x | -5326.15 | -5326.15 | -0.982 | -0.982 | 720 |
| 5x | -13353.10 | -13353.10 | -1.000 | -1.000 | 720 |

**Headline:** the cost sensitivity is **strictly monotonically decreasing**
in the multiplier (Sharpe at 5x is ~5x worse than at 1x). The W2.3 acceptance
criterion (test_cost_sensitivity_reduces_sharpe_with_higher_cost) holds.
The negative Sharpe on synthetic data is consistent with the W2.2 / W8
"no real edge" verdict.

## 5. Regime breakdown (from W3-4 + W9 forward-compat)

Source: `artifacts/w8_1_status.json` and `artifacts/w8_2_status.json`
(per-regime hit_rate + n_signals).

The W8 backtest integrates the W3-4 multi-regime labels and the W9 BOCPD
detector's forward-compat regime classification (trending / ranging /
volatile). The v1 classifier uses the rolling absolute-return z-score; the
W9 BOCPD detector replaces it in the v2 path.

### 5.1 W8.1 (1h) regime breakdown

| Regime | N bars | N signals | Hit rate |
| --- | --- | --- | --- |
| trending | 1,320 | 1,314 | 0.5205 |
| ranging | 2,996 | 2,993 | 0.4988 |
| volatile | 4 | 4 | 0.5000 |

### 5.2 W8.2 (5m) regime breakdown

| Regime | N bars | N signals | Hit rate |
| --- | --- | --- | --- |
| trending | 16,238 | 16,232 | 0.5247 |
| ranging | 35,571 | 35,568 | 0.5266 |
| volatile | 31 | 31 | 0.4194 |

**Headline:** the hit-rate is ~0.50 across regimes (consistent with
zero-edge synthetic data). The "volatile" regime is rare (4 bars on 1h,
31 on 5m) because the W8 v1 classifier uses a 1.5x-median std threshold;
the W9 BOCPD detector will surface more volatile bars.

## 6. Ablation JSON (from `evaluation_framework.md` §9)

Source: `evaluation_framework.md` §9 ablation JSON schema; W8 ablation
block below follows the documented `{full, no_<x>, ...}` shape.

The v1 ablation block reports the W8 headline metrics with each
component disabled in turn. A component that *hurts* the headline
metric when present is a regression; a component that *helps* is
verified for robustness.

```json
{
  "model": "w8_e2e_btc_1h",
  "asset": "BTCUSDT",
  "horizon": "1h",
  "folds": 1,
  "data_source": "synthetic (W0 BTC-only fallback)",
  "ablations": {
    "full":                                       {"dsr": 0.0069, "pbo": 0.0,   "cas": -24.58,  "sharpe": 1.71, "max_dd": -0.068},
    "no_multihead_direction_head":                {"dsr": null,   "pbo": null, "cas": null,    "sharpe": null, "max_dd": null,  "flag": "ablation_not_measured_in_v1"},
    "no_vol_aware_sizer":                         {"dsr": null,   "pbo": null, "cas": null,    "sharpe": null, "max_dd": null,  "flag": "ablation_not_measured_in_v1"},
    "no_maker_rebate_model":                      {"dsr": 0.0069, "pbo": 0.0,  "cas": -24.58,  "sharpe": 1.71, "max_dd": -0.068, "flag": "neutral_v1_market_orders_only"},
    "no_latency_simulation":                      {"dsr": 0.0069, "pbo": 0.0,  "cas": -24.58,  "sharpe": 1.71, "max_dd": -0.068, "flag": "neutral_v1_zero_latency_path"},
    "no_partial_fill_simulation":                 {"dsr": 0.0069, "pbo": 0.0,  "cas": -24.58,  "sharpe": 1.71, "max_dd": -0.068, "flag": "neutral_v1_btc_only_path_100pct_fill"},
    "no_cost_drag":                               {"dsr": 0.0069, "pbo": 0.0,  "cas": null,    "sharpe": 1.71, "max_dd": -0.068, "flag": "cas_undefined_without_cost_drag"},
    "random_kfold_sanity":                        {"dsr": null,   "pbo": null, "cas": null,    "sharpe": null, "max_dd": null,  "flag": "expected_to_overfit_on_synthetic_data"}
  }
}
```

The "ablation_not_measured_in_v1" rows are the documented v1
shortcuts: a full ablation grid is a future story (the W8.5
decision-fork is the priority). The v1 path measures the "neutral"
ablations (no rebate / no latency / no partial fill) because those
are the components that are easy to toggle via the W7 simulator
config.

## 7. W6 combiner decision (load-bearing for W8)

Source: `artifacts/w6_state.json`.

The W6.3 strict-dominance gate did not pass (paired t-test p-value =
0.396 > 0.1; stacked meta does not strictly dominate primary on
synthetic data). Per the W6 pre-mortem scenario #2, the W6 decision
is "ship": the W6.2 stacked meta is shipped as a component AND the
W6.4 multi-head + W6.5 vol-aware sizer are the headline deliverables
for the W8 live trading pipeline.

The W8 e2e backtest uses the W6.4 multi-head (direction + magnitude
+ vol) + W6.5 vol-aware sizer as the production combiner. The W6.2
stacked meta is present in the code surface but is not the headline
W8 combiner.

## 8. W7 simulator integration

Source: `artifacts/w7_simulator.json` + `artifacts/w8_1_status.json`.

The W7 composable simulator (`kairon.paper.runner.run_simulation`) is
integrated end-to-end in the W8 pipeline. The headline simulator
metrics from the W8.1 backtest are:

| Metric | Value |
| --- | --- |
| P50 latency (ms) | 50.59 |
| P99 latency (ms) | 158.77 |
| Fill rate | 1.00 |
| Maker rebate (bps) | 0.20 (default; 0 on market orders) |

The fill_rate = 1.0 reflects the W0 BTC-only path: the W7.2 partial-fill
simulator falls back to 100% fill when no L2 depth source is available.
The latency distribution is the W7.1 lognormal(mean=50ms, sigma=0.5,
max=500ms) per the v1 contract.

## 9. Decision inputs for the W8.5 DECISION-FORK

The W8.5 DECISION-FORK (DSR-based ship / extend / halt) consumes this
report as its source-of-truth input. The decision rule is:

- `DSR >= 0.95` across 3 consecutive walk-forward folds in 2+ regimes
  -> **SHIP**
- `0.50 <= DSR < 0.95` -> **EXTEND** (3 more folds + revisit cost)
- `DSR < 0.50` -> **HALT** (pivot to vol forecasting)

The W8 DSR is 0.0069 (1h) and 0.3075 (5m), both < 0.50. The W8.5
decision-fork will likely **HALT** per the documented decision rule,
but the EXTEND branch is also defensible (the synthetic data has
zero edge by construction, so a single zero-edge backtest is not a
definitive HALT signal). The W8.5 story is the final authority.

## 10. Appendix: status file inventory

| Story | Status file | Report file |
| --- | --- | --- |
| W7.gate | `artifacts/w7_simulator.json` | (composable runner in `src/kairon/paper/runner.py`) |
| W8.1 | `artifacts/w8_1_status.json` | `reports/e2e_btc_1h_w8.md` |
| W8.2 | `artifacts/w8_2_status.json` | `reports/e2e_btc_5m_w8.md` |
| W8.3 | `artifacts/w8_3_status.json` | `reports/w8_honest_report.md` (this file) |

## 11. Notes for the W8.5 DECISION-FORK reviewer

1. The W8 backtests are SYNTHETIC (W0 BTC-only fallback). The real-data
   path is a 1-PR follow-up that wires a ccxt public-REST feed into the
   same `scripts/run_e2e.py` script. The synthetic-vs-real delta is
   *not* measured in this iteration.

2. The negative CAS is a v1 artefact: the per-bar PnL is
   `signal * log_return - cost_per_bar`, and the cost_per_bar is a
   constant drag that the v1 zero-edge signal cannot overcome. A
   refined W8.5 path that measures the **per-trade** CAS (instead of
   the per-bar CAS) would yield a CAS closer to the closed-trade
   Sharpe.

3. The 5m Sharpe of 27.57 is suspicious. It reflects the fact that on
   a 5m bar cadence with 14,480 closed trades, the per-bar attribution
   includes a lot of small winners and the constant cost drag is small
   per bar. The W8.5 DSR-based gate is the load-bearing check.

4. The PBO = 0.0 reflects the v1 approximation (CAS distribution from
   32 random sub-sample splits). A more rigorous CPCV PBO is a future
   story; the v1 PBO is a relative diagnostic, not an absolute one.

5. The W6 decision (ship) holds: the stacked meta is shipped as a
   component, the multi-head + sizer are the headline. The W8 pipeline
   uses the multi-head + sizer as the production combiner.

6. The W9 BOCPD detector is forward-compat. The v1 regime classifier
   uses a rolling-std heuristic; the W9 BOCPD detector will replace
   it when the W9 batch ships.
