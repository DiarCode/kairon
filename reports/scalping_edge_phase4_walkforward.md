# Phase 4 — Walk-Forward / Out-of-Sample Hold-Out (Step 1)

**Date:** 2026-06-22
**Script:** `scripts/walkforward_sol.py` · **Raw output:** `artifacts/wf_sol.json`
**Predecessor:** `reports/scalping_edge_phase3.md` (Phase 4 resolution — LONG_ONLY preset, in-sample)

## What this checks

Every win-rate in the Phase 4 work is **in-sample** (the LONG_ONLY preset and the
mr_long thresholds were chosen from the full 8-week SOL store). The drift
kill-switch is a guardrail that fires *after* losing on fresh bars — it is not
proof the edge holds. This step holds out the most-recent 25% of bars and asks:
**does the SOL mr_long edge survive on bars the matrix was not chosen on?**

This is the one out-of-sample check the existing OHLCV store allows. It is
deterministic and offline. The split is strictly temporal (the store is sorted
ascending by `ts`), so there is no leakage.

## Method

* Store: `data/history/SOL-USDT-PERP/{5m,15m}.parquet` (8 weeks).
* Split: train = first 75% of bars, test = last 25% (most-recent held out).
  * 5m: 16,128 bars → train 12,096 / test 4,032. Boundary: train ends
    2026-06-05T13:10, test starts 2026-06-05T13:15 (≈ 2.5 weeks held out).
  * 15m: 5,376 bars → train 4,032 / test 1,344. Boundary: 2026-06-05T13:00→13:15.
* Per split, run `analyze_setups` with `LONG_ONLY` and `MEAN_REVERSION_ONLY`
  (fresh strategy state each run), record the mr_long and mr_short edge buckets.
* `analyze_setups` runs `run_scalp_backtest` with `buffer_bars=200` warmup — the
  test backtest seeds indicators on the first 200 *test* bars (standard practice;
  warmup bars are indicator seeding, not matrix calibration).

## Results

### 5m (the live-runner timeframe)

| matrix | split | setup | n | win% | TP | SL | sumPnL | avgRR |
|--------|-------|-------|----:|-----:|---:|---:|--------:|------:|
| long-only | train | mr_long | 72 | 74% | 53 | 19 | +19.238 | +0.57 |
| long-only | **test** | **mr_long** | **20** | **95%** | **19** | **1** | **+6.959** | **+1.11** |
| mean-reversion | train | mr_long | 72 | 74% | 53 | 19 | +12.599 | +0.57 |
| mean-reversion | train | mr_short | 25 | 28% | 7 | 17 | −4.370 | −0.60 |
| mean-reversion | test | mr_long | 20 | 95% | 19 | 1 | +6.959 | +1.11 |
| mean-reversion | test | mr_short | 9 | 67% | 6 | 3 | +1.570 | +0.46 |

### 15m

| matrix | split | setup | n | win% | TP | SL | sumPnL | avgRR |
|--------|-------|-------|----:|-----:|---:|---:|--------:|------:|
| long-only | train | mr_long | 11 | 55% | 6 | 4 | +0.949 | +0.25 |
| long-only | **test** | **mr_long** | **1** | **0%** | 0 | 1 | −0.186 | −1.07 |
| mean-reversion | train | mr_long | 10 | 60% | 6 | 3 | +1.044 | +0.39 |
| mean-reversion | train | mr_short | 10 | 40% | 4 | 6 | −1.000 | −0.17 |
| mean-reversion | test | mr_long | 1 | 0% | 0 | 1 | −0.186 | −1.07 |
| mean-reversion | test | mr_short | 1 | 100% | 1 | 0 | +0.144 | +0.57 |

## Verdict

**5m (live-runner timeframe): PASS.**
* Out-of-sample mr_long: n=20 (≥15 floor met), **95% win-rate** (≥65% bar),
  **+6.959 PnL** (>0 bar). The edge did not just survive — it was *stronger* on
  the held-out window than in-sample (74%→95%). The 20 trades over ~2.5 weeks
  ≈ 8 trades/week — higher than the in-sample "~2-3 trades/week" estimate,
  meaning the held-out window was range-rich (good for Step 2).

**15m: INCONCLUSIVE (under-powered, not a refutation).**
* Out-of-sample mr_long n=1 — far below the n≥15 floor. One trade, one loss.
  This says nothing about the 15m edge; it says the 2.5-week hold-out produced
  only one 15m mr_long setup. Not a FAIL (no evidence the edge broke), not a
  PASS (no evidence it held). The live runner uses 5m, so 15m being
  inconclusive does not block Step 2.

**Overall approval gate (per plan): PASS → proceed to Step 2.**
The plan's approval criterion is "out-of-sample mr_long ≥65% win + positive PnL
on the test split, on the 5m timeframe the live runner uses." 5m meets it
decisively (95%/+6.96/n=20). 15m is inconclusive, not a failure.

## Honest caveats (do not paper over)

1. **mr_short is regime-dependent, not a clean loser in every window.** On 5m,
   mr_short was a 28% loser in-sample (n=25) but a 67% winner out-of-sample
   (n=9). This is *variance*, not a robust reversal: n=9 in one 2.5-week window
   vs n=25 in-sample, and the in-sample expectancy is clearly negative
   (avgRR −0.60). Killing mr_short (LONG_ONLY) remains the right, conservative
   call — mr_long is *consistent* (74%→95%); mr_short is *inconsistent*
   (28%↔67%). The drift kill-switch protects the real margin if a mr_short-like
   regime ever favours shorts and we're flat. **But: LONG_ONLY may be leaving
   occasional short edge on the table in down-trending 2-week windows.** That
   is an acceptable price for consistency; revisit only if a sustained live
   down-regime produces many missed mr_short wins.

2. **The 5m test window was range-rich (20 mr_long trades in ~2.5 weeks).**
   The in-sample "~2-3 trades/week" estimate came from the full 8 weeks
   (including trending stretches). Step 2's trade count will depend on the live
   regime at run time — it is market-dependent, not guaranteed to match this
   hold-out.

3. **Out-of-sample n is still modest (20 on 5m).** 95% over n=20 has a wide
   confidence interval (Wilson 95% CI ≈ 76%-99%). The edge is real-directionally
   but the *exact* win-rate is uncertain. The drift kill-switch + 30% drawdown
   halt remain the live guardrails — this walk-forward raises confidence, it
   does not remove the need for them.

4. **No mainnet.** All bars are testnet; mainnet microstructure differs.

## Approval decision

**Step 1 → PASS. Proceed to Step 2 (guarded live compounding run on SOL 5m
long-only).** The first real out-of-sample evidence supports the SOL mr_long
edge; it is no longer a purely in-sample claim. The run is gated by the drift
kill-switch + 30% drawdown halt + daily-loss backstop, so a live regime that
breaks the edge halts the real margin rather than gambling it.