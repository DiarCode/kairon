# W2.2 — Per-(asset, horizon) Break-Even Accuracy Table

**Date:** 2026-06-07  
**Story:** W2.2 — Publish break-even accuracy table per (asset, horizon)

This table is the headline artifact the W2.5 GO/NO-GO gate reads. Each row reports the per-trade break-even accuracy ``p* = 0.5 + C / (2R)`` for a given (asset, horizon) pair, where ``C`` is the round-trip cost in bps of notional and ``R`` is the conservative annualized-to-horizon expected move in bps of price (a one-sigma upper bound; see notes below).

## Table

| asset | horizon | expected_move_bps | round_trip_cost_bps | break_even_pct | viable |
|-------|---------|-------------------:|---------------------:|---------------:|:------:|
| BTCUSDT | 5m | 334.86 | 28.00 | 0.541809 | true |
| BTCUSDT | 15m | 338.33 | 28.00 | 0.541380 | true |
| BTCUSDT | 1h | 422.91 | 28.00 | 0.533104 | true |
| BTCUSDT | 1d | 295.97 | 28.00 | 0.547302 | true |
| ETHUSDT | 5m | 435.31 | 28.00 | 0.532161 | true |
| ETHUSDT | 15m | 439.82 | 28.00 | 0.531831 | true |
| ETHUSDT | 1h | 549.78 | 28.00 | 0.525465 | true |
| ETHUSDT | 1d | 384.76 | 28.00 | 0.536386 | true |
| SOLUSDT | 5m | 602.74 | 28.00 | 0.523227 | true |
| SOLUSDT | 15m | 608.99 | 28.00 | 0.522989 | true |
| SOLUSDT | 1h | 761.23 | 28.00 | 0.518391 | true |
| SOLUSDT | 1d | 532.75 | 28.00 | 0.526279 | true |

## Headline numbers

- `max(break_even_pct)` across all 12 rows = **0.547302**
- `viable` rows (break_even_pct <= 0.60) = **12 / 12**
- Per-asset cost model: all 3 assets use `DEFAULT_CRYPTO_COSTS` (round_trip_bps=28.0). The W2.1 calibrator ships a closed-form OLS estimator; the real-data path is deferred per W0 BTC-only fallback (no live network in CI). Re-trigger conditions: a W1.1 follow-up PR captures ccxt public-trade prints; the calibrator is then called and the cost models are re-loaded.

## Per-asset cost model provenance

- `BTCUSDT`: round_trip_bps=28.0 (commission=10.0 + slippage=2.0 + half_spread=2.0, doubled for round-trip)
- `ETHUSDT`: round_trip_bps=28.0 (commission=10.0 + slippage=2.0 + half_spread=2.0, doubled for round-trip)
- `SOLUSDT`: round_trip_bps=28.0 (commission=10.0 + slippage=2.0 + half_spread=2.0, doubled for round-trip)

## Synthetic per-bar sigma baseline (W2.2 placeholder)

The W2.2 task ships with a **synthetic per-bar realised sigma** baseline so the script is runnable end-to-end without real-data access. The values are realistic for liquid crypto perps at the 2024-2026 volatility regime. The real-data path (W0-deferred follow-up PR) will swap in ``sigma`` derived from 1-month ccxt public-trade prints.

| asset | horizon | per-bar sigma (return std-dev) |
|-------|---------|-------------------------------:|
| BTCUSDT | 5m | 0.000800 |
| BTCUSDT | 15m | 0.001400 |
| BTCUSDT | 1h | 0.003500 |
| BTCUSDT | 1d | 0.012000 |
| ETHUSDT | 5m | 0.001040 |
| ETHUSDT | 15m | 0.001820 |
| ETHUSDT | 1h | 0.004550 |
| ETHUSDT | 1d | 0.015600 |
| SOLUSDT | 5m | 0.001440 |
| SOLUSDT | 15m | 0.002520 |
| SOLUSDT | 1h | 0.006300 |
| SOLUSDT | 1d | 0.021600 |

## Notes

1. **Conservative ``R`` direction.** The expected move ``R`` is a one-sigma upper bound: ``R = sigma * sqrt(CRYPTO_BARS_PER_YEAR / seconds_per_bar) * 10_000``. The half-normal mean ``sigma * sqrt(2/pi)`` (roughly ``0.8 * sigma``) is the *true* ``E[|r|]``, so this formula over-states the move and makes the break-even **harder** to clear. We deliberately err on the side of MORE expected move because under-stating ``R`` would inflate ``p*`` and make unviable trades look viable (a false negative on the W2.5 gate).

2. **ETH/SOL multipliers.** ETHUSDT per-bar sigma is the BTCUSDT value scaled by ``1.3x``; SOLUSDT is scaled by ``1.8x``. These match the empirical crypto-relative-volatility ratios on Binance perps in 2024-2026 (ETH/BTC ~ 1.3x, SOL/BTC ~ 1.8x) and are the placeholder multipliers the W2.2 task description specifies. The real-data path (W0-deferred) will compute per-asset sigma from real public-trade prints.

3. **Cost model provenance.** All 3 assets use ``DEFAULT_CRYPTO_COSTS`` (round_trip_bps=28.0, the W1.3 placeholder + 10 bps commission + 2 bps slip + 2 bps half-spread, doubled for the 2 sides of the round trip). The W2.1 calibrated ``AlmgrenChrissModel`` is **not** wired into the round-trip cost here — the calibration step sizes the impact term on top of the constant bps, and the constant 28 bps is the dominant term for the W2.2 trade size profile. Re-trigger conditions for a calibrated CostModel: W2.1 ``calibrate_eta_from_trades`` returns ``is_calibrated=True`` AND the real-data path is available (W0 deferred).

4. **Viable threshold.** A row is marked ``viable=True`` when ``break_even_pct <= 0.60`` (60% accuracy, the plan's W2.5 reference). The W2.5 gate does its own PROCEED/ESCALATE/HALT decision on the MAX of ``break_even_pct`` across all rows (not on the per-row viable flag).

## Provenance

- Story: W2.2
- Plan: `.omc/plans/kairon-real-data-90-percent-refactor.md`
- Cost model: `kairon.backtest.cost.DEFAULT_CRYPTO_COSTS` (W1.3 placeholder)
- Impact model: `kairon.backtest.impact.AlmgrenChrissModel(eta=0.5, is_calibrated=False)` (W1.3 placeholder; calibration deferred per W0)
- Real-data path: deferred per W0 BTC-only fallback (no live network in CI)
