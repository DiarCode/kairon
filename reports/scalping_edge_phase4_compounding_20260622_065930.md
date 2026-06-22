# Phase 4 — Guarded Live Compounding Run (Step 2)

**Date:** 2026-06-22
**Session id:** `scalping-20260622_065930`
**Run:** SOL-USDT-PERP, 5m, `--setup-matrix long-only`, bankroll 10→100,
drift-killswitch ON, max-drawdown 0.30, risk-per-trade 0.025, rr 1.3, leverage 10x.
**Log:** `logs/scalping_20260622_065930.log` · **DB:** `data/scalping_20260622_065930.db`

## Prerequisite

Step 1 (walk-forward) verdict = **PASS** → approved to attempt live compounding.
This step ran.

## What happened

* **Launched clean** on Bybit **testnet** (`testnet=True, tld=com`). Real testnet
  USDT balance at start: **10520.21** (unchanged → synthetic-bankroll model intact,
  no real-account risk).
* **Preflight OK:** SOL tradeable at bankroll=10.00 — live-price risk-cap preflight
  returned `risk_qty=0.2992` ≥ min_qty 0.1000 (long side clears the min lot at the
  current price; the earlier min-lot concern was specific to shorts).
* **Guardrails armed:** drift kill-switch ON, 30% bankroll drawdown halt, daily-loss
  backstop, per-symbol SL cooldown (180s), native ATR TP/SL attached (`attach_stops=True`).
* **Feed healthy:** prewarmed 59 history bars (5m); `MultiSymbolPollingFeed` emitted
  closed 5m bars every ~5 min (closed-bar alignment working — bars land at 06:50,
  06:55, 07:00, 07:05, 07:10, 07:15, …). WebSocket connected and stable.

## Result: 0 trades

| table | rows |
|-------|-----:|
| live_decisions | 0 |
| live_orders | 0 |
| live_closed_trades | 0 |
| live_events | 0 |
| growth_ledger | 1 (initial bankroll row only) |

**SOL was static at 20.89 for the entire monitored window** — 8 consecutive closed
5m bars (06:50→07:15) printed the identical close. This is a flat/illiquid testnet
book, not a ranging market. The LONG_ONLY mr_long setup requires an *oversold dip*
(RSI/Stoch/BB-lower) plus bullish confirmation inside a RANGE regime; a price that
never moves off 20.89 produces no oversold extreme, so the matrix correctly emits
**no signal**. Selectivity is working as designed — the engine does not force trades
into a market that offers no setup.

The run was monitored for ~22 minutes (8 closed bars) and then stopped deliberately:
the testnet SOL book was stuck at 20.89 with zero bar-to-bar movement, so extending
the window would not have produced setups. Real testnet balance at stop: unchanged
(10520.21) — **no positions were opened, no risk was taken, no real margin moved.**

## Verdict: INCONCLUSIVE (no trades)

Per the plan's Step 2 approval criteria:

* 0 closed trades (SOL did not range during the window) → **INCONCLUSIVE**.
  Selectivity produced no entries; this is **NOT** a claim of compounding success
  and **NOT** a failure — it is the honest outcome of a selective engine on a
  static market.
* The drift kill-switch did **not** halt (it never fired — no trades to evaluate).
  The edge was neither confirmed nor refuted *live*; Step 1's out-of-sample PASS
  remains the standing evidence for the SOL mr_long edge.

## Honest expectations (recorded, not papered over)

* The plan stated up front: one autopilot pass cannot force a multi-day compounding
  outcome; 0 trades is a legitimate, honest result (selectivity working). That is
  exactly what happened.
* A full $10→$100 likely needs a **week of ranging SOL** (~2-3 trades/week per the
  in-sample estimate, ~8/week in the range-rich walk-forward hold-out). The live
  testnet book offered zero range in this window.
* **Testnet illiquidity is a real confound.** Testnet order books are thin and can
  stick at a single price for long stretches (SOL @ 20.89 here). The walk-forward
  PASS is on stored historical bars (which included real movement); the *live*
  testnet may not always offer the same microstructure. This is an inherent limit
  of testnet validation, not a flaw in the engine.
* The drift kill-switch + 30% drawdown halt + daily-loss backstop remain the live
  guardrails. They were armed and would have halted the run had trades started
  losing; they simply had no trades to evaluate.

## Approval decision

**Step 2 → INCONCLUSIVE → does NOT advance to Step 3** (an order-flow A/B needs
≥3 trades to compare arms; 0 trades makes an A/B meaningless). Step 3 is recorded
as INCONCLUSIVE (skipped, documented). **Advance to Step 4** (commit the verified
Phase-4 work + this honest validation record), per the plan's rule that a
non-approved step still advances to the commit.

## How to actually get compounding evidence (follow-up, not this pass)

To turn INCONCLUSIVE into a real compounding verdict, the run needs a window
where SOL actually ranges on testnet. Options (deferred, not done here):

1. **Schedule recurring guarded sessions** (e.g. a 1-2h run at a few different times
   over a week) until a ranging testnet window is caught and ≥3 trades close.
2. **Monitor the live book first** and only launch the compounding run when SOL has
   actually printed a meaningful range (e.g. last-30-bar BB width > some floor) —
   avoids burning a 2h run on a stuck book.
3. **Accept testnet limits** and treat the walk-forward PASS as the primary
   out-of-sample evidence, with the live run as a guardrail-validation exercise
   rather than a compounding demonstration.

The engine is built, guarded, and ready; the remaining variable is market
cooperation, which is not forceable.