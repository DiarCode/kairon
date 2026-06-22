# Phase 4 — Order-Flow Forward A/B (Step 3)

**Date:** 2026-06-22
**Status:** INCONCLUSIVE — not run (prerequisite not met)

## Prerequisite (from the plan)

Step 3 requires Step 2 to have produced **≥3 closed trades**, so that an order-flow
ON vs OFF comparison has trades in both arms to compare. Step 2 produced **0 closed
trades** (testnet SOL was static at 20.89 for the entire monitored window — see
`reports/scalping_edge_phase4_compounding_20260622_065930.md`).

## Verdict: INCONCLUSIVE (skipped, documented honestly)

An A/B over two comparable ranging-SOL windows is **meaningless with zero trades**:
there is nothing to compare on either arm. Per the plan, this is recorded as
INCONCLUSIVE rather than fabricated. The plan explicitly states: "If too few
trades to compare, document INCONCLUSIVE (do not fabricate a result)."

No decision (KEEP/DEFER) is made. The `--orderflow` feature remains **off-by-default**
(as shipped in Phase 4b) — that is the safe default and needs no change here.

## Why this step exists (unchanged)

`--orderflow` is the one Phase-4 fork that **could not be backtested** on the OHLCV
store (no historical L2), so its value can only be settled by a live forward A/B.
That remains true. The Step 1 walk-forward PASS validated the *matrix* (mr_long
edge) out-of-sample; the order-flow *timing* tweak still has no live evidence either
way. This is an open research question, not a defect.

## How to run it later (follow-up, not this pass)

When a guarded compounding run catches a ranging testnet SOL window and closes
≥3 trades, run the paired comparison:

* **Arm A (control):** `uv run python scripts/run_scalping_session.py --symbols
  SOL-USDT-PERP --timeframe 5m --setup-matrix long-only --duration <D>
  --bankroll-start 10 --bankroll-stop 100 --max-drawdown 0.30 --drift-killswitch`
* **Arm B (treatment):** same, plus `--orderflow --orderflow-interval 20`.

Compare per-arm: # trades, win-rate, mean `of_imbalance` at entry (entry-timing
proxy), realised R:R. KEEP `--orderflow` as a documented opt-in if Arm B shows
better-or-equal entry timing / R:R with no win-rate degradation and the drift
kill-switch did not halt Arm B; otherwise DEFER (leave off-by-default). The
feature already ships off-by-default, so either outcome is a safe, documented
state.

This Step 3 report records the INCONCLUSIVE verdict honestly; it does not block
the commit (Step 4).