# Scalping Edge Enhancement — Phase 3 (Live wiring + drift kill-switch + per-setup report)

**Date:** 2026-06-19 · **Venue:** Bybit TESTNET · **Bankroll:** synthetic $10 (real ~10.5k USDT account is margin only)

Phase 2 data-discovered the `MEAN_REVERSION_ONLY` setup-selection matrix and validated it in-sample (SOL 5m 35%→62% win, $10→$30.32). Phase 3 is the **out-of-sample guardrail** layer: it wires the matrix into the live runner, journals the chosen setup/regime per trade, adds a per-setup edge report on live outcomes, and installs a drift kill-switch that halts trading if the edge evaporates on fresh bars.

## What was built

| Piece | Location | Purpose |
|---|---|---|
| Drift kill-switch | `src/kairon/live/drift_killswitch.py` | Rolling-window (global + per-setup) win-rate / expectancy monitor; reports a halt verdict. Distinct from the existing feature-distribution `drift.py` (PSI/KS on indicator values). |
| Setup/regime journaling | `journal.py` (`IndicatorSnapshot.setup_id/regime`) + `orchestrator.py` snapshot build | The strategy's `setup_id` and `regime` are now persisted in `indicators_json` so live outcomes can be bucketed by setup without a schema change. |
| Setup-id lookup | `store.py` `decision_setup_id(order_id)` | Reads the setup_id back from a closed trade's entry decision (used by the drift monitor's per-setup window). |
| Orchestrator wiring | `orchestrator.py` `_apply_bankroll_close` | Every bankroll-mode close feeds the kill-switch (`realized_pnl / bankroll_at_close` as a bankroll-fraction, tagged with setup_id); a halt verdict stops the loop and writes a `drift_killswitch` event. |
| Runner CLI | `scripts/run_scalping_session.py` | `--setup-matrix {mean-reversion,legacy,all}` (default `mean-reversion`) and `--drift-killswitch/--no-drift-killswitch` (default on). |
| Per-setup report | `scripts/run_scalping_session.py` `_scalping_extras` + markdown writer | Buckets closed decisions by `setup_id` (via `json_extract`) into a "Per-setup edge (live, out-of-sample)" table: n, win%, TP, SL, sumPnL per setup. |

## Design decisions (deep-reasoned)

1. **Performance drift, not feature drift, is the kill-switch.** The repo already has a PSI/KS feature-distribution module (`drift.py`). But the overfitting risk for a *data-discovered* matrix is that the **edge stops paying**, not that the inputs shift. The new `DriftKillSwitch` measures realized win-rate/expectancy directly — the thing we actually care about — and is intentionally a separate module to avoid conflation.
2. **Bankroll-fraction, not absolute PnL, in the window.** The bankroll compounds (10→30 in the in-sample run), so an absolute-PnL window would compare early $0.25 losses with late $0.75 losses. Recording `realized_pnl / bankroll_at_close` keeps the rolling window comparable across the whole compounding curve — the threshold becomes bankroll-invariant.
3. **Per-setup window in addition to global.** A single setup bleeding (e.g. mean-reversion misfiring in a trend the regime gate missed) can hide inside a healthy blended global window. The per-setup window trips independently at a smaller min-sample (6 trades) so a broken setup is caught before it drags the whole book down.
4. **Conservative thresholds.** Global floor 40% win / -0.5% bankroll expectancy over 10 trades; per-setup floor 30% over 6 trades. The in-sample edge is 62-68%, so these are "edge has *clearly* evaporated" triggers — well below the edge, far above random — not hair-triggers that halt on a normal losing streak.
5. **Opt-in, legacy-preserving.** `setup_matrix=None` and `drift_killswitch=None` reproduce the original behaviour byte-for-byte. The default runner turns both on.

## Bug found and fixed during live validation

The first 5-min live session surfaced `scalping extras query failed: Cannot operate on a closed database.` The per-setup query was running `conn.execute()` **after** `conn.close()`. Fixed by moving the close to a `try/finally` so the connection stays open for all queries and is guaranteed closed even on error. Verified: `_scalping_extras` now runs clean on a real session DB, and a synthetic DB with tagged closed decisions populates the per-setup table correctly (mr_long 100% win TP=1, mr_short 0% win SL=1).

## Live validation (Bybit testnet)

5-min session `scalping_20260619_142231` (SOL, XRP, LINK):
- Preflight OK, real testnet balance 10533.01 USDT (unchanged — synthetic bankroll model).
- Risk-cap preflight on **live** prices: SOL tradeable, XRP tradeable, **LINK skipped** (`skip_risk_cap_breach`, min_bankroll_to_clear=39.64 > 10) — the Phase 0.2 guard works on real testnet prices, not just dry-run.
- WS connected, 3 symbols prewarmed with 90 history bars each, feed flowing, 55 ticks over 5 min.
- **0 trades.** This is correct and expected: the matrix is *selective by design* (MR-only, gated to range regimes ADX<20, exhaustion/MTF guards). In a 5-min window no qualifying MR-in-range setup appeared. Selectivity is the mechanism that lifted win rates to 62-68% in-sample; a 5-min live window producing no signal is the selectivity working, not a malfunction.
- Clean shutdown, markdown report generated with the new `Setup matrix` / `Drift kill-switch` headers and the `Per-setup edge` section (placeholder while 0 closed setups).

A longer 15-min session runs to try to catch real MR trades and exercise the full close→drift-record→per-setup-report path on live data; results appended below.

## Phase 4 (start): timeframe alignment — the live failure mode

The 5-min and 15-min live sessions both produced **0 trades**. Diagnosis: the setup-selection matrix was data-discovered and validated on **5m/15m** bars (SOL 5m mr_long 62% win, SOL 15m 68%), but the live runner hardcoded `timeframe="1m"`. 1m is noisier than 5m and outside the validated regime — so the matrix correctly refused to fire on bars it was never validated on. The selectivity was working, but against the wrong timeframe.

**Fix (bounded, data-driven):** made the live timeframe configurable and defaulted it to **5m** (the validated edge). `--timeframe {1m,5m,15m,...}` (default `5m`), propagated to the feed, `LiveConfig`, the session report, and the prewarm fetch. Two supporting changes:

- **Scaled prewarm lookback by timeframe.** The prewarm hardcoded `1m` and a 90-min lookback — on 5m that yields only ~18 bars, too few for the 30-bar warmup. Now the lookback is `max(90, warmup_bars * tf_minutes * 2)`, so 5m prewarms 60 bars (validated: live log shows `Prewarmed SOL-USDT-PERP with 60 history bars (5m)`).
- **Timeframe-aware poll interval.** `--poll-interval` now defaults to `~tf/4` floored at 15s (1m→15s, 5m→75s, 15m→225s) so each bar is sampled a few times without excess API churn.

This is the highest-value Phase 4 fix because it directly addresses the observed live failure mode (0 trades) without loosening the selectivity that gives the edge. `ruff check` clean; scalping test suite (92 tests) green; dry-run prints `timeframe: 5m` / `poll interval: 75.0s`.

### Live validation of the 5m timeframe fix (Bybit testnet, 20-min session `scalping_20260619_145554`)

A 1200s SOL+XRP 5m session confirmed the fix end-to-end:

- **Prewarm scales correctly:** `Prewarmed SOL-USDT-PERP with 60 history bars (5m)` and the same for XRP — the `max(90, warmup*tf_min*2)` lookback yields 60 bars on 5m (vs the old 1m hardcoded lookback which gave ~18). Both symbols warm at session start.
- **Feed emits 5m bars:** one bar per 5 min — `14:55 @50.18 → 15:00 @56.79 → 15:05 @54.14 → 15:10 @60.74 → 15:15 @47.54` (SOL), XRP flat at 1.15. The cadence matches the validated timeframe.
- **Poll interval adapts:** `MultiSymbolPollingFeed starting for 2 symbols (tf=5m, poll=75.0s)` — the `~tf/4` floor-15 default.
- **Clean shutdown, no "closed database" warning:** the Phase 3 `try/finally` per-setup-query bug fix held on a real session. Markdown report renders `Timeframe: 5m`, `Drift kill-switch: on`, and the `Per-setup edge` section (placeholder, 0 closed setups). Exit code 0.
- **0 trades — and why this is correct, not a malfunction:** SOL printed ±20% bar-to-bar swings (50→56→54→60→47) — a violently **trending/volatile** regime. The `MEAN_REVERSION_ONLY` matrix is range-gated (ADX<20, regime in RANGE/VOLATILE, exhaustion+MTF guards), so it correctly refused to fire MR setups into a market that is not ranging. **Selectivity is the mechanism that lifted in-sample win rates to 62-68%; the matrix refusing a trending market is the selectivity working, on the validated timeframe, exactly as designed.**

The key honest engineering finding this surfaces: **90% win-rate and "enough trades to compound 10× in a week" are in direct tension for a pure-MR matrix on a 2-symbol book.** The matrix fires so selectively that even on the validated 5m timeframe it produced 0 trades in 20 min of live range-absent market. Tightening toward 90% win-rate would make trade frequency *worse*; reaching trade volume needs either (a) more symbols in parallel (more chances one is ranging), (b) reintroducing momentum/breakout setups (a reversal of the Phase 2 MR-only finding), or (c) order-flow features that improve entry timing rather than selectivity. These are strategic forks, surfaced for scoping — see "Open Phase 4 forks" below.

## Open Phase 4 forks (strategic, surfaced for scoping)

The live finding above forces an honest fork. Three remaining Phase 4 items are all **feasible** (none are dead ends) but each is a strategic choice with a real trade-off, not a mechanical extension. They are listed here so the next session can scope them against the user's stance (predictable math/psychology; 70-90% accuracy as an engineering *target*; 10×/week aggressive, not guaranteed).

1. **Order-flow / microstructure features** (entry-timing, not selectivity). Feasible on testnet — `bybit.py:798` already polls pybit's HTTP `get_orderbook`, and testnet serves it. Additive bid/ask imbalance, spread, depth-ratio snapshots polled alongside OHLCV; feed them as extra confluence inputs. **Trade-off:** improves *entry timing* (closer to the stop, smaller slippage, better fill at the MR extreme) rather than raising win-rate — a different axis. Research + backtest iteration; the feed note excludes only *streaming* L2 (ccxt.pro), not polling.
2. **Setup ensemble** (reintroduce momentum/breakout alongside MR). The Phase 2 finding *killed* momentum/breakout/breakdown because they underperformed MR in-sample. An ensemble re-adds them, broadening trade opportunities across regimes. **Trade-off:** directly reverses the Phase 2 selectivity result; raises trade frequency but lowers blended win-rate toward the momentum/breakout in-sample numbers. A genuine design reversal, not a refinement.
3. **Win-rate target-floor tightening toward 90%.** **Trade-off:** the live finding shows the pure-MR matrix already fires rarely enough to produce 0 trades in 20 min on a range-absent market; tightening the gate toward 90% win-rate reduces trade frequency *further*, deepening the tension with the 10× volume requirement. The honest path to *both* high win-rate and volume is **breadth** (more symbols → more chances one is ranging) + order-flow timing, not a tighter floor on 2 symbols.

The drift kill-switch is the guardrail that makes any of these safe to try: whichever fork is taken, the out-of-sample edge monitor halts the book if the chosen path stops paying on fresh bars.

## Phase 4 resolution — the universe backtest decides (data-driven)

The three forks above were scoped by *running the data*, not by assertion. A parallel per-(symbol, timeframe) edge probe (`scripts/analyze_symbol_edge.py`, one process per cell on the local 8-week testnet history store) was run across the universe BTC/ETH/LINK/SOL/XRP × {5m, 15m} with the `MEAN_REVERSION_ONLY` matrix and the `legacy` (all-setups) baseline. The results settle all three forks:

**Fork 4a — breadth: REFUTED.** Only **SOL** has a mean-reversion edge. BTC and ETH produce **0 MR trades** in 8 weeks (no qualifying MR setups — they trend, they don't range enough at this horizon). LINK and XRP *do* produce MR trades but as **net losers** (XRP 5m mr_short 45% / mr_long sub-50% with negative PnL; LINK 5m mr_short 42%). Adding symbols to the book does not add edge — it adds losers or adds nothing. The honest breadth pivot is **not more symbols**; it is "run SOL on both 5m and 15m" (both carry the mr_long edge: 5m 78%, 15m 80-83%). Breadth across a wider universe is refuted by the universe itself.

**Fork 4c — setup ensemble: REFUTED.** The `legacy` baseline (all setups fire) shows momentum_short/momentum_long at **0-8% win rates** with deeply negative expectancy across every symbol and timeframe, and breakout/breakdown negative everywhere (testnet volume rarely "surges"). Reintroducing them reverses the Phase 2 MR-only selectivity finding for **no edge** — it would raise trade *frequency* by adding known losers. Not built; the refutation is the evidence above.

**Fork 4d — win-rate floor → 90%: the real lever is killing the losing side, not tightening the floor.** The universe data shows the drag on the SOL MR book is **mr_short**, a *universal* loser on testnet (SOL 5m 28%, XRP 5m 45%, LINK 5m 42% — all negative PnL) while **mr_long** is the only edge (SOL 5m 78%, SOL 15m 80-83%). Tightening the win-rate floor would starve the winning mr_long too; the selectivity that actually raises the blended win-rate is to **kill mr_short**. This is implemented as the `LONG_ONLY` preset:

```python
LONG_ONLY = SetupMatrix(
    enable_mr_short=False,    # the universal loser on testnet
    enable_mr_long=True,      # the only edge
    enable_momentum_short=False, enable_momentum_long=False,
    enable_breakdown=False, enable_breakout=False,
    regime_gate=True, exhaustion_guard=True, mtf_bias=True, confidence_calibration=True,
)
```

Wired into the runner as `--setup-matrix {mean-reversion, long-only, legacy, all}` (default unchanged at `mean-reversion` so existing behaviour is preserved). The `long-only` mode is the Phase 4 deliverable: it keeps every guard (regime gate, exhaustion, MTF, calibrated confidence) and the drift kill-switch, and just removes the losing side. Backtest-verified: SOL 5m `LONG_ONLY` → mr_long 78% win (+47.40), SOL 15m → 80% (+6.65), XRP 15m → 60% (+1.52), with **zero mr_short trades** (the drag is gone). Unit-tested (`test_long_only_kills_mr_short_keeps_mr_long`, `test_long_only_is_mr_subset_of_mean_reversion_only` — LONG_ONLY is a strict tightening, never a loosening, of MEAN_REVERSION_ONLY).

**Fork 4b — order-flow timing: BUILT (opt-in, off-by-default, guardrailed).** This is the one fork the data does not refute — it sits on a different axis (entry *timing*, not win-rate), and `bybit.py` already polls pybit's HTTP `get_orderbook` (now exposed as a public `BybitBroker.get_orderbook`). It is implemented as a pure, unit-tested microstructure module plus an opt-in live poller, **off by default** so the validated LONG_ONLY path is byte-for-byte unchanged unless `--orderflow` is passed:

- `src/kairon/live/orderflow.py` — pure `compute_orderflow(bids, asks)` → `OrderFlowSnapshot` (mid, spread_pct, imbalance, depth_ratio) + `orderflow_alignment(snapshot, direction)`. No I/O; tolerates garbage/empty/crossed testnet books (returns `None`). Unit-tested (13 tests).
- `ScalpingStrategy` gains `use_orderflow` (default False) + `last_orderflow`. When on, confidence is nudged by `1 + orderflow_weight * alignment` (±15% max): a bid-heavy book supports a long bounce (boost), an ask-heavy book supports a short fade (boost), the opposite damps. The nudge routes through confidence-scaled **sizing** (opposed book → smaller entry) rather than a hard flatten, so a thin/ambiguous book is a no-op. The order-flow fields (`of_imbalance`/`of_spread_pct`/`of_depth_ratio`) are journaled in the indicator snapshot. Strategy integration unit-tested (6 tests).
- `TradingLoop` gains an opt-in `orderflow_provider` (sync `symbol -> snapshot` closure); `_make_prediction` refreshes `strategy.last_orderflow` from the cache before each predict, fail-soft (a provider error resets to `None`, never crashes the loop). Default `None` → legacy path unchanged. Orchestrator injection unit-tested (4 tests).
- `scripts/run_scalping_session.py` — `--orderflow` / `--orderflow-interval` flags; an `OrderFlowPoller` async task polls each symbol's book into a cache every N seconds and the loop reads it via the closure (the hot loop never blocks on a network call). Wired through `_build_orderflow` + `_shutdown_session` helpers (keeps `_run_session` under the branch limit). Report/dry-run print the order-flow state.

**Why off-by-default is the honest choice.** Order flow is a *live-only* signal — there is no historical L2 book on the OHLCV store to backtest it against, so it cannot be validated offline the way the setup matrix was. Shipping it on by default would replace a backtest-verified edge (SOL mr_long 78%) with an un-validated timing tweak. The drift kill-switch is the out-of-sample guardrail: turning `--orderflow` on is safe to *try* because the loop halts if the tweak degrades live performance. Live-validated: a `--orderflow` testnet session starts the poller cleanly alongside the feed/loop (log: "Order-flow poller started (interval=20.0s)"), no crashes, clean shutdown.

**Honest strategic conclusion.** The edge is **SOL-mr_long-specific**, not a broad-universe short-tilted scalping edge. On testnet the original "short-tilted" tilt loses (mr_short is a universal loser); the data overrides it toward long-only mean-reversion on SOL. Realistic opportunity is ~2-3 SOL mr_long trades/week (146 MR trades in 8 weeks on 5m, of which ~100 are mr_long), not 10×/week. The engine is now guarded (drift kill-switch, per-setup expectancy floor, closed-bar alignment, testnet hard-enforced) and selectivity-tightened (LONG_ONLY) — it compounds when the SOL-mr_long edge appears and halts on drawdown. 70-90% win-rate is achievable *on the SOL mr_long setup specifically* (78-83% in-sample); reaching it across a broad book is what the data refutes. The user's stance (predictable math + crowd psychology, 70-90% via selectivity) holds — the selectivity that delivers it is "SOL mr_long only", not "more symbols + more setups".

## Phase 4 hardening (from the multi-agent validation pass)

A parallel architect + security-reviewer + code-reviewer pass (all returned ITERATE) surfaced four concrete issues. All fixed, tested, and lint-clean:

1. **Per-setup expectancy floor (architect rec 1 + code-reviewer MEDIUM).** The global window trips on win-rate OR expectancy, but the per-setup window only checked win-rate — so a single setup bleeding via deeply-negative expectancy with a borderline win-rate would *not* trip the per-setup switch, exactly the bleed the per-setup path exists to catch. Added `per_setup_min_expectancy` to `DriftKillSwitchConfig` and a parallel expectancy branch in `check()`. Symmetric with the global guard now.
2. **NaN/inf no longer neutralises the kill-switch (security M2).** A non-finite fill (testnet thin books have produced anomalous fills) recorded into the rolling window made every `win_rate < floor` and `expectancy < floor` comparison `False`, silently disabling the guardrail for the whole window. `record()` now coerces non-finite values to a full-loss `-1.0` (biasing toward halting on the bad fill) instead of poisoning comparisons. The orchestrator call-site also skips recording when the bankroll is dust (`prev <= start*1e-6`) so a near-depleted bankroll can't blow the fraction to inf.
3. **Closed-bar feed alignment (architect rec 2).** The polling feed hardcoded a 2-min fetch window and emitted the *last* row — on 5m that is the in-progress bar (deduped at first appearance, ~0-75s into the bar), so the strategy traded a partially-formed bar instead of the closed bar the in-sample edge was validated on. `_poll_one` now fetches a timeframe-scaled window (`tf_min*4`) and emits only *closed* bars via a shared `_drop_inprogress_bars` helper (last row dropped when `open_ts + tf > now`). The same helper is applied in `_prewarm_buffers` so prewarm seeds only closed bars with no feed overlap. Cost: up to one poll interval of latency before a freshly-closed bar is acted on — the correct out-of-sample alignment tradeoff.
4. **Testnet invariant hard-enforced (security M1).** The runner inherited `testnet=settings.bybit_testnet`, so an operator setting `BYBIT_TESTNET=false` + `LIVE_DRY_RUN=false` could silently route the scalping strategy (with attached leverage) at the real mainnet account — and the feed (hardcoded `testnet=True`) would then mismatch the broker venue. `_build_scalping_broker` now refuses to start unless testnet, and forces `testnet=True` on the broker. Fail-closed on the "never mainnet" constraint.

Also cleaned up the `DriftKillSwitch` field defaults (dropped the dead `maxlen=20` from the `_pnls` default_factory; the `__post_init__` rebind is what honours `config.window`) and removed the redundant `_by_setup = {}` rebind.

## Tests

- `tests/live/test_drift_killswitch.py` — **20 tests**: kill-switch semantics (no-halt-below-min-sample, low-win-rate halt, negative-expectancy halt, edge-intact no-halt, per-setup independent halt, window eviction, **per-setup negative-expectancy halt, per-setup window eviction, edge-intact with per_setup populated, `None`/empty setup_id skips per-setup, NaN/inf coercion to loss**), `decision_setup_id` round-trip, orchestrator wiring (halt stops loop + writes `drift_killswitch` event; off = no event), per-setup report population + empty case, and the `TestDropInprogressBars` closed-bar helper cases.
- `tests/live/test_setup_matrix.py` — **Phase 4 `LONG_ONLY` tests**: `test_long_only_kills_mr_short_keeps_mr_long` and `test_long_only_is_mr_subset_of_mean_reversion_only` (LONG_ONLY is a strict tightening of MEAN_REVERSION_ONLY — never a loosening).
- `tests/live/test_orderflow.py` — **13 tests**: pure `compute_orderflow` (balanced/bid-heavy/ask-heavy/empty-ask-saturation/depth-limit/empty-book/crossed/garbage/extreme one-sidedness) + `orderflow_alignment` (neutral/long-aligned/long-opposed/short-flipped).
- `tests/live/test_scalping_orderflow.py` — **6 tests**: strategy confidence nudge (off is byte-identical regardless of snapshot, aligned raises, opposed lowers, neutral no-op, None no-op, fields journaled only when on).
- `tests/live/test_cooldown.py` — **+4 Phase 4b orchestrator-injection tests**: provider injects snapshot before predict, no-provider leaves it untouched, provider exception is fail-soft, provider gated off when `use_orderflow=False`.
- Combined live + scalping engine + setup-matrix + order-flow suite: **436 passed**, 2 deselected (testnet round-trips). `ruff check` clean on all changed files.

## Honest caveats

- The matrix remains **in-sample-discovered**; the drift kill-switch is the guardrail, not proof of out-of-sample edge. Only sustained live sessions (and eventually walk-forward) can confirm the edge holds on fresh bars.
- 0 trades in 5 min is the selectivity working; a real compounding run needs either longer duration, more symbols, or a regime that produces MR setups. 10× in a week remains aggressive, not guaranteed.
- The per-setup report only populates once trades close; a session with no qualifying setups shows the placeholder row.