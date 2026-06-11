# Information Architecture — Kairon

**Date:** 2026-06-05

## 1. Top-level IA

```
Kairon
├── Home (Watchlist)
├── Markets
│   ├── Crypto
│   │   ├── BTC-USDT
│   │   ├── ETH-USDT
│   │   └── ...
│   ├── Stocks
│   │   ├── AAPL
│   │   ├── SPY
│   │   └── ...
│   └── Indices / FX / Commodities
├── Compare
│   ├── New comparison
│   └── Saved comparisons
├── Alerts
│   ├── Active rules
│   ├── History
│   └── Channels
├── Backtest
│   ├── Templates
│   ├── My backtests
│   └── Templates library
├── Research
│   ├── Notebooks
│   ├── Models
│   ├── Datasets
│   └── Diagnostics (DSR / PBO / calibration)
├── Trade journal (paper)
└── Settings
    ├── Account
    ├── Risk
    ├── Data sources
    ├── Notifications
    └── About / limits (honest ceiling table)
```

## 2. Information hierarchy on a tile (home)

```
[● live]  BTC-USDT                       [regime: Volatile]
$64,231   +1.2% (24h)                    [last signal: long, p=0.62]
─────▲─────                               cooldown 12m
  sparkline
```

Priority order: freshness dot → symbol → price → change → regime → last signal.

## 3. Information hierarchy on the asset detail

```
┌────────────────────────── HEADER ──────────────────────────────┐
│ ● live  BTC-USDT  1h horizon                  [regime chip]   │
│ $64,231 +1.2% 24h   vol 1.1%                                  │
└────────────────────────────────────────────────────────────────┘
┌── FORECAST ───────┬── EVIDENCE ──────────┬── RISK ───────────┐
│ p(up) = 0.62      │ top drivers:          │ cost-aware Shar = │
│ band [0.55, 0.70] │  • MVRV z=-1.2        │   1.1             │
│ mag  +0.4%        │  • 24h vol z=+2.1     │ max DD 9%         │
│ vol   1.1%        │  • FinBERT 0.7        │ no-signal 35%     │
│                   │ regime: Volatile      │                   │
│ [sliders: T, K]   │ similar past bars (3) │ [risk budget]     │
└───────────────────┴───────────────────────┴───────────────────┘
┌─────────────────────────── WHY THIS? ─────────────────────────┐
│ LLM explanation card (grounded; click to expand citations)    │
└────────────────────────────────────────────────────────────────┘
┌─────────────────────────── ACTION ────────────────────────────┐
│ [Place paper trade]   [Alert me on similar]   [Add to compare]│
└────────────────────────────────────────────────────────────────┘
```

## 4. Card patterns

| Card type | Anatomy |
|-----------|---------|
| **Signal** | symbol, horizon, side, p(calibrated), confidence band, regime, drivers (top 3), cost-aware Sharpe, max DD, "no-signal" prob, LLM "why" link, action row |
| **Miss** | same anatomy, side mirrored, "why did this miss?" link |
| **No-signal** | "Our models don't see an edge here right now." + threshold slider link |
| **Calibration drift** | yellow banner, drift score, recommended action |
| **Backtest result** | equity curve, drawdown, PBO, DSR, accuracy-at-coverage table |

## 5. Color & iconography

- Regime chips always have an icon, not only color:
  - Trending ▲, Ranging ↔, Volatile ⚡, Stressed ⚠
- Status dots: live (green), stale (yellow), down (red), no-data (gray).
- Confidence is a band, not a point — shown as `[0.55, 0.70]`, never a single number alone.
- Money is shown with thousands separators; deltas with explicit sign and color.

## 6. Empty / loading / error

| State | Pattern |
|-------|---------|
| Empty | onboarding card + a single primary CTA |
| Loading | skeleton + 500ms perceived target |
| Error | retry button + status link + auto-retry schedule |
| Permission | clear CTA + explain what's gated |

## 7. Settings IA

```
Settings
├── Account (profile, plan, billing)
├── Risk (default position size, max DD, calibration toggle)
├── Data sources (per-source enable/disable, with warnings)
├── Notifications (channels, cooldown, digest time)
├── Display (theme, density, plain English toggle)
├── API (for Diego: keys, webhooks, rate limits)
└── About (release notes, DSR of current build, honest ceiling table)
```

## 8. Honest ceiling table (settings → About)

| Asset | Horizon | Achievable direction accuracy (evidence) |
|-------|---------|------------------------------------------|
| Crypto | 5m (5h ahead) | 60-75% (R-002 with conf T) |
| Crypto | 1h | 55-62% |
| Crypto | 1d | 52-58% |
| Equity index | 1d | 55-60% (R-001) |
| Equity stock | 1d | 52-57% |
| Equity stock | 1h | 53-58% |

These are the realistic numbers. The system will not pretend otherwise.
