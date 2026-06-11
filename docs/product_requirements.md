# Product Requirements — Kairon

**Date:** 2026-06-05
**Audience:** Product, design, engineering.

## 1. Target users (personas)

### 1.1 "Retail Quant Sam" — primary
- Trades crypto and US equities on 5-min to daily horizons.
- Has a basic understanding of technical indicators but not ML.
- Wants signals that **explain themselves** and **respect costs**.
- Cares about: confidence, regime, why now, what could go wrong.
- Pain points: most platforms either oversell certainty or hide the math.

### 1.2 "Swing-trader Aisha" — primary
- Trades US equities on 1h-1d horizons.
- Wants compare-across-assets, alerts, calendar overlays (earnings, macro events).
- Pain points: alert fatigue, dashboards that don't update fast enough.

### 1.3 "Pro quant team lead Diego" — secondary
- Runs a small quant pod, has Python skills.
- Wants a research API + reproducible runs + a UI for the non-coders on the team.
- Pain points: vendor lock-in, opaque backtests, no DSR / PBO.

### 1.4 "Crypto-native Chris" — secondary
- Lives on 5-min charts, perpetual funding, on-chain.
- Wants order book + CVD + on-chain integrated.
- Pain points: signal-to-noise of social sentiment; needs a calibrated view.

## 2. Jobs to be done

| When I… | I want to… | So I can… |
|---------|------------|-----------|
| Open the app | See my watchlist with live state, regime, and last 3 signals | Decide where to look first |
| Pick an asset | See forecast, evidence, risk, and a confidence slider | Set a position size I trust |
| Compare two assets | See them side-by-side with the same metrics | Choose the better trade |
| Review my week | See hit rate, Sharpe, drawdown vs my baseline | Adjust my approach |
| Get an alert | Be pinged when a *new* signal meets *my* threshold | Catch opportunities without noise |
| Drill into a signal | See drivers, regime, similar past bars, the LLM explanation | Decide whether to act |
| Read the docs | Get a scientifically honest framing of the system's limits | Trust the system |

## 3. Product principles

1. **Honest before impressive.** Show coverage next to accuracy. Show DSR, PBO. Show "no signal" often.
2. **Fast first, deep later.** Default screen answers in <500 ms; deeper screens load on demand.
3. **One signal, one screen.** Never bury the decision in a 12-widget dashboard.
4. **Calibrated, not confident.** "We have a 62% calibrated p(up)" beats "AI is bullish."
5. **Evidence on tap.** Every signal is one click from its drivers, regime, and a human-readable explanation.
6. **Costs are first-class.** A signal card always shows the cost model and the post-cost Sharpe.
7. **Replayable.** Every recommendation is reproducible from a config + a date.

## 4. Core screens

### 4.1 Home / Watchlist
- Tiles: symbol, last close, 24h % change, regime chip, last-signal chip, sparkline, badge (`● live ● stale ● no-data`).
- Sort: by confidence, by alert, by my notes.
- Pinned top: 3 most-traded.

### 4.2 Asset detail
Three columns, scroll on mobile:

- **Forecast:** primary direction, calibrated probability, confidence band (5-95%), expected magnitude, expected vol, target/stop.
- **Evidence:** top 3 drivers (SHAP), regime context, similar past bars, sentiment, on-chain (if crypto).
- **Risk:** cost-aware Sharpe, max drawdown in this config, expected holding time, "no-signal" probability.

### 4.3 Backtest / Compare
- Pick (asset, horizon, model, cost model).
- Outputs: equity curve, drawdown, hit rate, accuracy-at-coverage table, PBO, DSR.
- Side-by-side: compare up to 3 configs.

### 4.4 Alerts
- Define a rule: "BTC-USDT, 1h, top-K ≥ 5, T ≥ 1.4, regime in {Trending, Volatile}".
- Channel: in-app, webhook, email.
- Cooldown: user-set (default 15 min).
- Audit: every alert logged with all inputs and outputs.

### 4.5 Research / Notebooks (for "Diego")
- Reproducible runs: config + data version hash → exact outputs.
- Pulls features, runs models, displays backtest + ablations.

### 4.6 Settings
- Risk budget, default horizons, cost model, calibration toggle, regime filter toggle.
- Data sources: enable/disable per source (with a clear "this disables X" warning).

## 5. Interaction states (every screen)

| State | Behavior |
|-------|----------|
| **Empty** | Onboarding card: "Add 3 symbols to your watchlist to begin." |
| **Loading** | Skeleton + progress meter; max 500ms perceived. |
| **Live** | Green dot, last update timestamp, auto-refresh on by default. |
| **Stale** | Yellow chip "data delayed 3m" + last-good timestamp. |
| **No-signal** | "Our models don't see an edge here right now." Honest. |
| **Error** | Retry button, error code, link to status page. |
| **Permission denied** (e.g., not subscribed) | Clear CTA, no dark patterns. |

## 6. Notification policy

- In-app + email digest by default; webhook optional.
- Default cooldown 15m per (symbol, horizon) pair.
- Always include: calibrated probability, regime, drivers link.
- Never "Buy now!" — always "Signal: long, p=0.62 (calibrated)".

## 7. Reporting behavior

- Daily digest at user's local close: top 3 signals, top 3 misses, PnL attribution.
- Weekly: cost-aware Sharpe, drawdown, regime breakdown.
- Monthly: DSR, PBO, calibration drift, model health.

## 8. Pricing posture (suggested)

- Free: paper trading + 3 symbols + 1d horizon.
- Pro: paper + 50 symbols + 1h/1d + alerts.
- Team: research API + multi-user + custom cost models.

## 9. Out of scope for v1

- Live trade execution against real money (paper trade only).
- Social/copy-trading features.
- Options/derivatives pricing engines (we consume Greeks, not build them).
- Mobile native apps (responsive web only).
