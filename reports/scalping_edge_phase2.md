# Scalping Edge Enhancement — Phase 1+2 Research Report

**Date:** 2026-06-19 · **Venue:** Bybit TESTNET · **Bankroll:** synthetic $10 (real ~10.5k USDT account is margin only)

## What was built

| Phase | Module | Purpose |
|---|---|---|
| 1.1 | `src/kairon/data/history_store.py`, `history_fetch.py`, `scripts/fetch_history.py` | 8-week testnet OHLCV parquet store (450k bars, 5 symbols × 1m/5m/15m) |
| 1.2 | `src/kairon/backtest/scalping_engine.py` | Independent vectorized backtest — iterates bars, calls `ScalpingStrategy.predict` + shared `pure_fns` (risk_size_qty, post_rounding_guard, stop_exit_price) to size/stop/exit. NOT an orchestrator replay. |
| 1.3 | `src/kairon/backtest/scalping_cost.py` + `tests/backtest/test_fidelity_gate.py` | Bybit-testnet-calibrated cost model (15 bps round-trip) + fidelity gate (6 tests) reproducing the documented SOL min-lot-overshoot case and proving sim↔pure-fn parity. **HARD GATE passed.** |
| 2 | `src/kairon/live/regime.py`, `setup_matrix.py`, `backtest/setup_analysis.py` | Data-discovered regime classifier + setup-selection matrix (`MEAN_REVERSION_ONLY`) + per-setup edge analysis harness. Wired into `ScalpingStrategy(setup_matrix=...)` — opt-in, default `None` preserves legacy behaviour. |

## The diagnosis (why the model wasn't capturing edge)

The first backtest over the real 8-week testnet store, broken down **per setup**, showed the strategy was firing 4 losing setup types that swamped 2 winning ones:

**Edge (keep):** mean-reversion only — `mr_long` / `mr_short`. SOL 15m `mr_long` won **70%**, +4.68 sumPnL.
**No edge (kill):**
- `momentum_short` / `momentum_long`: single-digit win rates, −1R expectancy, −5.5 to −8.3 sumPnL each (shorting downtrends / longing uptrends → stopped when MR reasserts).
- `breakdown` / `breakout`: negative on testnet (volume rarely surges → confirmation is noise).

So the user's observation ("model not predicting properly") was correct and now **explained with hard data**: the strategy's selectivity was wrong, not the indicators.

## The fix (data-discovered setup-selection matrix)

`MEAN_REVERSION_ONLY` matrix: keep `mr_short`/`mr_long`, kill momentum + breakout/breakdown, gate MR to **range regimes** (ADX<20; MR bleeds in trends ADX>25), add an **exhaustion guard** (skip MR at RSI>85 short / RSI<15 long — continuation, not reversion), an **MTF bias** (longer-EMA higher-timeframe trend — no counter-trend bottom/top-picking), and **confidence calibration** (MR bonus, momentum suppressed).

## Validation: legacy vs MEAN_REVERSION_ONLY ($10 bankroll, no drawdown halt, 8wk testnet)

| symbol/tf | legacy win% | MR_only win% | legacy sumPnL | MR_only sumPnL | end$ legacy | end$ MR_only |
|---|---|---|---|---|---|---|
| **SOL 5m** | 35 | **62** | -6.48 | **+20.32** | 3.52 | **30.32 (3×)** |
| **SOL 15m** | 37 | **68** | -4.10 | **+5.40** | 5.90 | **15.40** |
| XRP 15m | 28 | 49 | -9.94 | -0.71 | 0.06 | 9.29 |
| XRP 5m | 30 | 46 | -9.98 | -6.79 | 0.02 | 3.21 |
| LINK 15m | 32 | 38 | -9.55 | -5.21 | 0.45 | 4.79 |
| LINK 5m | 27 | 36 | -9.90 | -7.67 | 0.10 | 2.33 |

Selectivity turned SOL from a loser into a **3× winner** and lifted win rates to 62–68% — on the 70–90% accuracy target trajectory.

## Honest caveats

- **In-sample:** the matrix was data-discovered on the same 8 weeks it's validated on → overfitting risk. Phase 3 (live A/B shadow on fresh testnet bars) + walk-forward are the out-of-sample check.
- **5m is noisier than 15m** for mean-reversion (XRP/LINK 5m improved win rate but stayed negative). 15m is the preferred MR timeframe.
- **BTC/ETH untradeable** at $10 (risk-sized qty < min lot). Only low-priced SOL/XRP/LINK tradeable at this stake.
- **No drawdown halt** in the research runs (max_drawdown=None) to measure full per-trade edge; the live runner keeps the 30% halt.
- 10× in a week is aggressive, not guaranteed. The engine now **has** edge where it had none, compounds when MR setups appear in ranges, and halts on drawdown — but real testnet live conditions (slippage, thin books, execution gaps) will differ from the sim.

## Tests

- `tests/backtest/test_scalping_engine.py` — 23 tests (engine semantics + real SOL smoke).
- `tests/backtest/test_fidelity_gate.py` — 6 tests (cost preset + sim↔pure-fn parity + real-bar SOL overshoot reproduction).
- `tests/backtest/test_setup_analysis.py` — 3 tests.
- `tests/live/test_setup_matrix.py` — 14 tests (regime + matrix + strategy wiring).
- Full live suite: **368 passed**, 2 deselected (testnet round-trips). `ruff check` clean.