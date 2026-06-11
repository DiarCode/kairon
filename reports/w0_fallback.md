# W0 — Blocking Inputs Fallback Decision

**Date:** 2026-06-07
**Decision-maker:** Ralph engineer
**Status:** ACTIVE FALLBACK

## The 3 BLOCKING items per the plan

| # | Item | Owner | Resolved? | Fallback consequence |
|---|------|-------|-----------|----------------------|
| W0.1 | Polygon / Tiingo API key for real-time US equity candles | USER | **NOT RESOLVED** | W1.2 (Polygon WS adapter) ships as a **stub** that mocks a real Polygon endpoint via `respx`. W8's US-equity headline (SPY 1h) is dropped; W8 ships BTC-only. |
| W0.2 | `ccxt.pro` license or decision to use public WS (Binance/Bybit free tier) | USER | **NOT RESOLVED** | W1.1 (CcxtCandleFeed) implements the **public WS** path first because (a) it does not require a license, (b) it works for BTCUSDT 1m candles on Binance and Bybit, (c) the implementation can be upgraded to `ccxt.pro` later by swapping the import. If the user later provides `ccxt.pro`, the upgrade is a 1-PR change. |
| W0.3 | 2y history source (Polygon paid / CryptoDataDownload free / Tardis trial) | USER | **NOT RESOLVED** | W1.1 captures a small **1-month** BTCUSDT 1m sample from the public Binance REST endpoint (free, no key required) for the smoke test. W2's cost calibration uses ccxt public-trades prints (also free). W8's headline is **6-month BTC** (not 12-month) to stay within the public-REST rate-limit budget. |

## Decision

**BTC-only headline. W1 proceeds. W2-8 follow the plan with the SPY/AAPL leg marked deferred.**

The plan's W0 contingency was explicit: *"if any of the BLOCKING items are not confirmed by W0 start, the plan adapts: drop SPY/AAPL from W8 and ship BTC-only as the headline deliverable."* This document is the engineer-side recording of that decision.

## What is in scope (BTC-only)

- W1.1: Binance + Bybit public WS for BTCUSDT, ETHUSDT, SOLUSDT
- W1.3: Almgren-Chriss placeholder (no data dependency)
- W1.4: KaironSettings (all 5 fields added, regardless of whether they are populated)
- W1.5: Partitioned parquet writer (synthetic 70M-row test fixture suffices)
- W1.6: Leakage fixture (synthetic + 1-month real sample)
- W2: Cost calibration from 1mo ccxt public trades (free) + break-even table for BTC/ETH/SOL
- W3-4: Meta-labeling on BTCUSDT
- W5: Cost-aware loss wiring
- W6: Stacked meta + multi-head
- W7: Latency-aware simulator
- W8: 6mo BTCUSDT 1h + 6mo BTCUSDT 5m backtest (no SPY leg)
- W9: BOCPD regime fix
- W10: Honest report

## What is deferred to Phase 2 (or until W0.1/W0.3 resolved)

- W1.2 US-equity WebSocket adapter (Polygon or Tiingo) — stub only in W1
- W8 SPY/AAPL/US-equity backtest
- Any cross-asset (BTC+SPY) portfolio analysis
- L2 microstructure features (Tardis/LOBSTER) — already Phase 2

## What the engineer can proceed with RIGHT NOW

All W1 tasks except W1.2 (which is the US-equity adapter). W1.2 is reduced to a stub that demonstrates the typed interface and a mocked test, deferred until W0.1 is resolved.

## Re-trigger conditions

If at any point in W1-W8 the user resolves one of the BLOCKING items, the plan re-broadens:
- W0.1 resolved → activate W1.2 (Polygon or Tiingo WS), re-add SPY/AAPL to W8
- W0.2 resolved (ccxt.pro) → swap the import in W1.1's CcxtCandleFeed
- W0.3 resolved (2y history) → extend W1.1 to capture full 2y, extend W8 to 12mo

No code is wasted: the BTC-only path is also the right path for a BTC-trading product, and the US-equity path adds on top without rearchitecting.

## Sign-off

This decision is consistent with the ralplan consensus (Architect round 2 + Critic round 2, both APPROVE) and the plan's documented W0 contingency. No architect re-review required for the BTC-only fallback; the plan explicitly accounts for it.
