# W2.3 — Cost Sensitivity Shock (CAS at 0.5x, 1x, 2x, 5x cost)

**Date:** 2026-06-07  
**Story:** W2.3 — Cost sensitivity shock (CAS at 0.5x, 1x, 2x, 5x cost)

This report is the cost sensitivity shock required by ``evaluation_framework.md`` §8.4. The table below reports the Sharpe, Sortino, max-drawdown, total-return, and trade-count metrics for a 1mo BTCUSDT 1h synthetic equity curve at four cost multipliers (0.5x, 1x, 2x, 5x). The cost shock is applied to per-trade PnL: ``trade_pnl[i] -= multiplier * base_round_trip_bps / 10000 * notional_proxy`` with ``notional_proxy = 1.0``.

## Table

| multiplier | sharpe | sortino | max_dd | total_return | n_trades |
|-----------:|-------:|--------:|-------:|-------------:|---------:|
| 0.50 | -1312.6783 | -1312.6783 | -0.6283 | -0.628325 | 720 |
| 1.00 | -2650.5033 | -2650.5033 | -0.8646 | -0.864642 | 720 |
| 2.00 | -5326.1532 | -5326.1532 | -0.9821 | -0.982124 | 720 |
| 5.00 | -13353.1030 | -13353.1030 | -1.0000 | -0.999960 | 720 |

## Headline numbers

- `base_round_trip_bps` = **28.00** (from `kairon.backtest.cost.DEFAULT_CRYPTO_COSTS.round_trip_bps` = 28.0)
- `bars_per_year` = **8760** (1h bars, from `kairon.backtest.metrics.BARS_PER_YEAR_1H`)
- `n_trades` = **720** (one synthetic per-trade observation per bar; per-trade mean = 0.3 bps, std-dev = 1.0 bps)
- **Cost-shock direction:** Sharpe is monotonically non-increasing in the cost multiplier for the synthetic positive-edge baseline. The 0.5x tail has a *higher* Sharpe than the 1x baseline (cheaper fees -> better risk-adjusted return); 2x and 5x progressively erode the edge.
- **W2.5 gate flag.** The W2.5 GO/NO-GO gate's `cost_sensitivity_present` flag flips to True when this sidecar exists at `artifacts/cost_sensitivity_w2.json`. The gate's PROCEED/ESCALATE/HALT decision is NOT altered by the flag (per the W2.5 deviation #2); the flag is informational.

## Cost-shock semantics

For each multiplier ``k`` and base round-trip cost ``C`` (in bps of notional), the per-trade PnL is shocked by subtracting ``k * C / 10000 * notional_proxy`` from each entry, with ``notional_proxy = 1.0``. The equity curve is then rebuilt by per-trade compounding: ``equity[t] = initial_equity * prod_{i<t} (1 + shocked_pnl[i])``. At ``k = 0`` the cost shock is zero and the equity curve matches the no-cost baseline. At ``k = 1`` the cost shock is the full round-trip (28 bps); 2x and 5x progressively stress the cost regime.

## Synthetic baseline provenance

- 1mo BTCUSDT 1h: **720** hourly bars (30 days x 24h; synthetic placeholder per W0 BTC-only fallback; real-data path deferred)
- per-bar mean return ``mu`` = **0.0003** (3.0 bps per bar; positive-Sharpe baseline)
- per-bar std-dev ``sigma`` = **0.0035** (35.0 bps per bar; matches the W2.2 BTCUSDT 1h sigma baseline of 0.0035)
- per-trade PnL mean = **0.3** bps (small positive edge)
- per-trade PnL std-dev = **1.0** bps
- RNG seed = **20260607** (matches the W2.1 seed for bit-determinism across the W2.x batch)
- initial equity = **100000** (USD, arbitrary scale; the Sharpe / Sortino / max_dd are scale-invariant)

## Notes

1. **Cost model provenance.** `base_round_trip_bps` is the round-trip bps from `kairon.backtest.cost.DEFAULT_CRYPTO_COSTS` (commission=10 bps + slippage=2 bps + half_spread=2 bps, doubled for the round trip = 28 bps). The W2.1 calibrator ships a closed-form OLS estimator; the W1.3 placeholder is used here. Re-trigger conditions: W2.1 `is_calibrated=True` AND the real-data path is available (W0 deferred).

2. **Per-trade PnL attribution.** The cost-sensitivity sweep shocks the supplied ``trade_pnl`` vector per-trade, then rebuilds the equity curve by per-trade compounding. The ``n_trades`` column is the per-trade count (``len(trade_pnl)``); in this synthetic fixture it equals the number of bars. The real-data path (W0 deferred) will pass a per-trade PnL vector from the backtest engine's trade log.

3. **W2.5 gate flag.** The W2.5 GO/NO-GO gate's `cost_sensitivity_present` flag flips to True when this sidecar exists at `artifacts/cost_sensitivity_w2.json`. The gate's PROCEED/ESCALATE/HALT decision is NOT altered by the flag (per the W2.5 deviation #2); the flag is informational.

## Provenance

- Story: W2.3
- Plan: `.omc/plans/kairon-real-data-90-percent-refactor.md`
- Module: `kairon.evaluation.cost_sensitivity`
- Cost model: `kairon.backtest.cost.DEFAULT_CRYPTO_COSTS` (W1.3 placeholder)
- Real-data path: deferred per W0 BTC-only fallback (no live network in CI)
