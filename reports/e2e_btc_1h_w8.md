# W8 — End-to-end BTCUSDT 1h backtest

**Story:** W8.1  
**Decided at:** 2026-06-22T07:32:45Z  
**Symbol:** BTCUSDT  
**Timeframe:** 1h  
**N bars:** 4320  
**Data source:** synthetic BTCUSDT 1h log-normal price walk (W0 BTC-only fallback; ccxt public-REST path is a 1-PR follow-up). n_bars=4320, sigma=0.005, seed=20260608.  

## Headline metrics

| Metric | Value |
| --- | --- |
| Total return | 0.058831 |
| Annualized return | 0.502001 |
| Annualized vol | 0.256915 |
| Sharpe (annualized) | 1.713332 |
| Sortino (annualized) | 2.273268 |
| Max drawdown | -0.068148 |
| Calmar | 7.366346 |
| Win rate | 0.534959 |
| Profit factor | 1.046083 |
| N trades | 1230 |

## W8 deliverable metrics (CAS / DSR / PBO)

| Metric | Value | Acceptance |
| --- | --- | --- |
| Cost-aware Sharpe (CAS) | -24.578753 | reported |
| Deflated Sharpe Ratio (DSR) | 0.006897 | >= 0.95 (ship) |
| Probability of Backtest Overfitting (PBO) | 0.000000 | <= 0.10 |

## Calibration (Brier / ECE)

| Metric | Value | Acceptance |
| --- | --- | --- |
| Brier score | 0.249338 | reported |
| Expected Calibration Error (ECE) | 0.003220 | <= 0.05 |

## Equity curve summary

- Initial equity: `10000.00`
- Final equity: `10588.31`
- Min equity: `9535.07`
- Max equity: `10651.82`

## Per-regime breakdown (W3-4 + W9 forward-compat)

| Regime | N bars | N signals | Hit rate |
| --- | --- | --- | --- |
| trending | 1320 | 1314 | 0.5205 |
| ranging | 2996 | 2993 | 0.4988 |
| volatile | 4 | 4 | 0.5000 |

## Coverage-accuracy Pareto (W3.5 forward-compat)

The W8 pipeline integrates the W3.5 coverage-curve module. The full curve is serialised in the status sidecar (`artifacts/w8_1_status.json` or `artifacts/w8_2_status.json`).

## W7 simulator integration

| Metric | Value |
| --- | --- |
| Fill rate | 1.000000 |
| P50 latency (ms) | 50.5907 |
| P99 latency (ms) | 158.7681 |
| Maker rebate (bps) | 0.0000 |
