# User Flows — Kairon

**Date:** 2026-06-05
**Format:** Step-by-step flows with explicit decision points, evidence surfacing, and the honest "no signal" path.

## 1. First-time onboarding

1. Land on the app → empty state with "Add 3 symbols to start".
2. User types `BTC-USDT` (autocomplete suggests canonical `BASE-QUOTE`).
3. Symbol added → tile shows "warming up" skeleton.
4. Within 30s, the tile populates: last close, regime, last 3 signals (or "no signals yet").
5. Tooltip tour: "Tap a tile for the full picture. The chip shows the regime. The dot shows data freshness."

**Decision point:** *If user adds <3 symbols*, the home screen stays in onboarding mode and explicitly says "3 is a good start for comparison."

## 2. Inspecting a single asset

1. Tap `BTC-USDT` tile.
2. Asset detail screen loads in <500ms with cached state.
3. Forecast column: p(up)=0.62, confidence band [0.55, 0.70], expected magnitude +0.4%, expected vol 1.1%.
4. Evidence column: top 3 drivers with mini-sparklines (e.g., "MVRV z=-1.2, 24h volume z=+2.1, FinBERT sentiment 0.7").
5. Risk column: cost-aware Sharpe 1.1, max DD 9%, expected holding 6h, no-signal probability 35%.
6. "Why this signal?" button → LLM explanation card with cited inputs.

**Decision point:** *If the model is in calibration drift*, the risk column shows a yellow banner and suggests "consider lowering your position size by 50% today".

## 3. Acting on a signal (paper trade)

1. From the asset detail, user clicks "Place paper trade".
2. Modal pre-fills: side from signal, size from user's risk budget, stop/target from policy.
3. User adjusts: size, stop, target, time-stop.
4. Confirms → paper trade logged with full provenance (config hash, model version, regime, drivers).
5. Trade shows up in "Open positions" with live mark-to-market.

**Decision point:** *If user sets size > risk budget*, modal warns but does not block (advanced users may want this).

## 4. Setting an alert

1. From any signal, user clicks "Alert me on similar".
2. Rule builder:
   - symbol (single or list)
   - horizon (5m / 15m / 1h / 1d)
   - threshold T (slider 1.0-2.0)
   - top-K consensus (slider 1-9)
   - regime filter (multi-select)
   - cooldown (default 15m)
3. Live preview: "expected ~X alerts per day" using the calibration fold.
4. Channels: in-app / email / webhook.
5. Save → alert appears in Alerts screen with last-fired and stats.

## 5. Comparing two assets

1. From a tile, user clicks "Compare" → picks a second symbol.
2. Side-by-side screen:
   - Same horizon, same calibration period.
   - Aligned columns: p(up), expected magnitude, vol, regime, cost-aware Sharpe, drivers.
3. "Why the difference?" → LLM explanation grounded in the diff of inputs.
4. Save comparison → "My comparisons" list.

## 6. Reviewing the day

1. Daily digest at user's local close: top 3 signals, top 3 misses, PnL attribution.
2. User can click any signal/miss to jump to the full asset detail.
3. "Explain why this missed" → LLM-grounded analysis.

## 7. Diagnosing a model

1. From "Research" screen (Diego persona), user picks (model, asset, horizon).
2. Reproducible run starts: config + data version hash → exact outputs.
3. Outputs include ablation table, calibration curve, PBO, DSR.
4. "Diff vs last release" → shows exactly what changed and whether it improved or regressed.

## 8. The "no signal" path (explicit)

When the system has no edge:
1. Asset detail shows "no signal" with a confidence threshold slider pre-set.
2. Tooltip: "Lower the threshold to see more signals at lower accuracy, or raise it for fewer but stronger signals."
3. The screen never *invents* a signal. Honest silence is the product.

## 9. Error and recovery flows

| Error | UX |
|-------|-----|
| Data feed down for symbol | Stale chip + retry; click for status |
| Calibration drift detected | Yellow banner on home + "consider risk-off" |
| Model retrain failed | Last good model continues; banner explains; retry scheduled |
| Cost model invalid for symbol | Block the trade; explain why |
| LLM call timeout | Show static evidence card; defer LLM explanation |
| User exceeds risk budget | Modal blocks; require explicit override |
| LLM output doesn't cite inputs | Reject the response; show evidence card only |

## 10. Accessibility

- All flows keyboard-navigable.
- Color is never the only signal (regime chips have icons).
- Charts have data tables.
- All LLM explanations have a "plain English" toggle.
