# W8 — End-to-end BTCUSDT 5m backtest

**Story:** W8.2  
**Decided at:** 2026-06-22T07:32:49Z  
**Symbol:** BTCUSDT  
**Timeframe:** 5m  
**N bars:** 51840  
**Data source:** synthetic BTCUSDT 5m log-normal price walk (W0 BTC-only fallback; ccxt public-REST path is a 1-PR follow-up). n_bars=51840, sigma=0.0012, seed=20260608.  

## Headline metrics

| Metric | Value |
| --- | --- |
| Total return | 1.243578 |
| Annualized return | 351.809985 |
| Annualized vol | 0.213598 |
| Sharpe (annualized) | 27.571950 |
| Sortino (annualized) | 36.732507 |
| Max drawdown | -0.016812 |
| Calmar | 20926.368106 |
| Win rate | 0.579627 |
| Profit factor | 1.249858 |
| N trades | 14480 |

## W8 deliverable metrics (CAS / DSR / PBO)

| Metric | Value | Acceptance |
| --- | --- | --- |
| Cost-aware Sharpe (CAS) | -359.365204 | reported |
| Deflated Sharpe Ratio (DSR) | 0.307468 | >= 0.95 (ship) |
| Probability of Backtest Overfitting (PBO) | 0.000000 | <= 0.10 |

## Calibration (Brier / ECE)

| Metric | Value | Acceptance |
| --- | --- | --- |
| Brier score | 0.249968 | reported |
| Expected Calibration Error (ECE) | 0.000589 | <= 0.05 |

## Equity curve summary

- Initial equity: `10000.00`
- Final equity: `22435.78`
- Min equity: `9993.85`
- Max equity: `22457.74`

## Per-regime breakdown (W3-4 + W9 forward-compat)

| Regime | N bars | N signals | Hit rate |
| --- | --- | --- | --- |
| trending | 16238 | 16232 | 0.5247 |
| ranging | 35571 | 35568 | 0.5266 |
| volatile | 31 | 31 | 0.4194 |

## Coverage-accuracy Pareto (W3.5 forward-compat)

The W8 pipeline integrates the W3.5 coverage-curve module. The full curve is serialised in the status sidecar (`artifacts/w8_1_status.json` or `artifacts/w8_2_status.json`).

## W7 simulator integration

| Metric | Value |
| --- | --- |
| Fill rate | 1.000000 |
| P50 latency (ms) | 50.0827 |
| P99 latency (ms) | 161.7068 |
| Maker rebate (bps) | 0.0000 |
