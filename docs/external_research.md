# External Research — Kairon

**Date:** 2026-06-05
**Method:** Topic-decomposed web searches across market data, on-chain, sentiment, libraries, backtesting, evaluation, and LLM layers. Each section ends with a *Use / Skip / Defer* recommendation and a justification.

---

## 1. Market data APIs

### 1.1 Crypto exchanges (via CCXT)

| Provider | Latency (REST) | Rate limit (per-exchange) | WebSocket | Coverage | Cost | License | Recommendation |
|----------|----------------|---------------------------|-----------|----------|------|---------|----------------|
| **Binance** | ~50-200 ms | Leaky bucket, `rollingWindow` mode | yes | Spot + Perp + options | free tier, pay-as-you-go for high freq | commercial | **Primary** for crypto OHLCV + order book + execution |
| **Bybit** | ~80 ms | 20ms base; per-endpoint weights | yes | Spot + linear + inverse + options | free | commercial | **Primary for paper trading** (R-002 used it; good docs) |
| **Coinbase Advanced Trade** | ~50-200 ms | 30/s private, 10/s public | yes | Spot | free | commercial | **Include** for USD pairs and US-compliance |
| **OKX / Kraken / Bitfinex** | similar | similar | yes | Spot + derivatives | free / paid | commercial | **Include via CCXT** for cross-venue arbitrage features |

**Why CCXT:** Unified async API across 100+ exchanges, built-in rate limiting, WebSocket support via `watchOrderBook`, supported by `orjson` and `coincurve` for performance. *Source: ccxt GitHub README.*

**Use:** Binance + Bybit + Coinbase as primary; rest via CCXT. **Skip direct exchange SDKs** unless we need native order types not exposed by CCXT.

### 1.2 Stocks / ETFs

| Provider | Coverage | Historical depth | Latency | Cost | License | Recommendation |
|----------|----------|------------------|---------|------|---------|----------------|
| **Polygon.io** | US stocks, options, forex, crypto | 15y intraday | real-time WS | $29-199/mo (free delayed tier) | commercial | **Primary** for live US equities |
| **Tiingo** | US + global, fundamentals, news | 20y+ | daily EOD free; intraday paid | $10-50/mo | commercial | **Include** for EOD + fundamentals + news |
| **Alpha Vantage** | US + global | 20y+ | 5/min free tier | free / $50/mo | commercial | **Include** for free-tier prototyping |
| **Yahoo Finance (yfinance)** | global, daily + 1m/5m/15m/60m | varies | scraping | free | unofficial scraping, no SLA | **Dev only** |
| **Databento** | institutional-grade US futures/equities | full depth | very low latency | per-symbol | commercial | **Defer** to v2 |
| **TWRR Kaggle** | 10k+ assets, dividends, splits, corp actions | 1950-2024 | EOD | one-off | research-only | **Use as historical bulk** |

**Use:** TWRR for backtest bulk, Tiingo for live EOD + fundamentals + news, Polygon for intraday + tick + live US.

### 1.3 Macro / economic

| Provider | Coverage | Latency | Cost | License | Recommendation |
|----------|----------|---------|------|---------|----------------|
| **FRED (St Louis Fed)** | US macro, 800k+ series | 1-day | free | public | **Primary** for macro features |
| **World Bank API** | global | 1-day | free | public | **Include** for global macro |
| **ECB / BoE / IMF** | regional | varies | free | public | **Include** for FX & rates |
| **TradingEconomics** | comprehensive | varies | freemium | commercial | **Optional** |

**Use:** FRED as the default macro source (free, reliable, huge catalog).

### 1.4 News / sentiment

| Source | Coverage | Latency | Cost | License | Recommendation |
|--------|----------|---------|------|---------|----------------|
| **CryptoPanic** | crypto news + sentiment tag | minutes | free / pro | commercial | **Primary for crypto news** |
| **Tiingo News** | US stocks + tickers | minutes | included | commercial | **Include for stocks** |
| **GDELT** | global news in 100+ langs | 15 min | free | public | **Include** for global signal |
| **Bloomberg/Refinitiv** | institutional | real-time | $$$ | commercial | **Defer** to v2 |
| **Reddit / X / Telegram** | social | varies | scraping | platform ToS risk | **Include via FinGPT** (Stanford-grade sentiment) |

**Use:** CryptoPanic + Tiingo News + GDELT for textual signals. **Skip** scraping Twitter/X without legal review.

### 1.5 On-chain (crypto-only)

| Provider | Coverage | Free tier | Cost | License | Recommendation |
|----------|----------|-----------|------|---------|----------------|
| **Glassnode** | BTC, ETH, top alts — MVRV, NUPL, SOPR, exchange flows | limited | Pro $29+/mo | commercial | **Primary** (entity-adjusted metrics) |
| **CryptoQuant** | same families + exchange wallet clustering | limited | Professional tier required for advanced | commercial | **Alternative** |
| **Bitquery** | multi-chain, free tier for traces | yes | freemium | commercial | **Include for on-chain traces** |
| **Dune Analytics** | SQL on public chains | yes | freemium | commercial | **Include for bespoke queries** |
| **Coin Metrics** | institutional grade | paid | $$$ | commercial | **Defer** |

**Use:** Glassnode Studio (free API key) for MVRV/NUPL/SOPR/exchange flows. **Defer** CryptoQuant unless we need wallet-clustering granularity. Bitquery for free trace data.

---

## 2. Python libraries

### 2.1 Data manipulation

| Library | Why | Justification |
|---------|-----|---------------|
| **polars** | Fast, multi-threaded, strict-typed schema | 5-10× faster than pandas, Arrow-native, eager+lazy API, strict types via `schema` |
| **pandas** | Ubiquitous | Compatibility for exchange APIs that return pandas-like frames |
| **numpy** | Core numerics | Required by everything |

**Decision:** **polars as default** (strict types, fast, modern). **pandas as compat layer** when an API requires it. Justified by the 5-10× speed gain on multi-million-row crypto frames and stronger typing story.

### 2.2 ML / DL

| Library | Why |
|---------|-----|
| **scikit-learn** | Linear models, RF, GBT, calibration, pipelines — the workhorse |
| **xgboost** | GBT, well-typed, fast, GPU optional |
| **lightgbm** | GBT, faster on some features, well-typed |
| **pytorch** | LSTM, Transformer, iTransformer, PatchTST — strict typing via `beartype` or `jaxtyping` |
| **pytorch-forecasting** | Production wrappers for time-series DL |
| **statsmodels** | ARIMA, GARCH, regime tests (HMM) |
| **arch** | GARCH family, vol forecasting |
| **river** | Online learning for streaming features |

**Decision:** scikit-learn + xgboost + lightgbm + pytorch. Use **pytorch-forecasting** for production wrappers where it helps.

### 2.3 Backtesting

| Library | Speed | Strengths | Recommendation |
|---------|-------|-----------|----------------|
| **backtesting.py** | very fast, ~50-200ms | easy, interactive HTML, single-asset | **Primary for quick iteration** |
| **vectorbt** | fastest for grid sweeps | vectorized, Numba JIT | **Primary for parameter sweep + ML backtest** |
| **zipline-reloaded** | medium | institutional, multi-asset, live | **Defer to v2** |
| **bt (backtest)** | medium | multi-strategy portfolio | **Include** for portfolio construction |
| **backtrader** | medium | event-driven, live | **Defer** |

**Decision:** `backtesting.py` for the user-facing research app, `vectorbt` for batch sweeps. (Source: 2025 community consensus; both well-maintained.)

### 2.4 Portfolio optimization

| Library | Strengths | Recommendation |
|---------|-----------|----------------|
| **PyPortfolioOpt** | Mean-variance, Black-Litterman, HRP, easy | **Primary for classical MVO** |
| **Riskfolio-Lib** | 24 risk measures, factor models, robust opt, drawdown | **Primary for production + drawdown-aware** |
| **empyrical** | Risk metrics only | **Include for risk stats** |
| **quantstats** | Tearsheets | **Include for backtest reports** |

**Decision:** **Riskfolio-Lib** as the production default (drawdown measures, robust optimization); **PyPortfolioOpt** for quick prototyping.

### 2.5 Type / quality

| Library | Use |
|---------|-----|
| **pydantic v2** | IO schemas, config, request/response |
| **pyright** | strict static type checking |
| **mypy** | alt static type checking |
| **ruff** | lint + format |
| **tenacity** | retry/backoff for external APIs |
| **httpx** | async HTTP for APIs |
| **orjson** | fast JSON |
| **loguru** | structured logging |
| **pydantic-settings** | typed env-driven config |
| **SQLModel** | typed SQL (if needed) |
| **duckdb** | in-process OLAP for parquet / CSV |
| **pyarrow** | parquet IO |
| **mlflow** | experiment tracking |
| **optuna** | hyperparameter optimization |

### 2.6 LLM

- **ollama (Python client)**: `Client().chat(model="gpt-oss:120b-cloud", ...)` — see Section 5.

---

## 3. DL architectures for time series (2024 SOTA)

| Architecture | Use | Where to use it in Kairon |
|--------------|-----|---------------------------|
| **LSTM** | Sequential short-horizon | 5-min crypto, 1d equity |
| **Transformer (vanilla)** | Mixed | Baseline |
| **Decision Transformer** | Conditional sequence | Multi-horizon return target (R-001 found competitive) |
| **PatchTST** | Long-horizon univariate | Daily+ equity, daily crypto |
| **iTransformer** | Multivariate | Cross-asset, portfolio-level heads |
| **N-BEATS / N-HiTS** | Long-horizon, hierarchical | Multi-scale vol forecasting |
| **TFT (Temporal Fusion Transformer)** | Multi-horizon with attention | Vol + direction + magnitude jointly |

**Decision:** LSTM + Decision Transformer + PatchTST + iTransformer + N-HiTS. (Source: ICLR 2024 papers, Time-Series-Library, NeuralForecast.) **Defer:** Informer, Autoformer (outperformed by PatchTST/iTransformer on most benchmarks).

---

## 4. Backtesting / evaluation

- **Walk-forward + purging + embargo**: gold standard (López de Prado 2018).
- **Combinatorial Purged Cross-Validation (CPCV)**: for probability-of-backtest-overfitting (Bailey et al. 2017).
- **Deflated Sharpe Ratio (DSR)**: required for any headline Sharpe claim (Bailey & López de Prado 2014).
- **Realistic costs**: commission + bid/ask spread + market impact (square-root model).
- **PBO audit**: see `When Alpha Disappears` (Zhang et al. 2026) and `Illusion of Alpha` (Bhand & Joshi 2026).

**Tools:**
- `mlfinlab` (or its open re-implementations) for CPCV, purging, embargo.
- `vectorbt` for high-speed sweep.
- `quantstats` for tearsheets.
- **In-house** DSR + PBO calculator (typed pydantic + numpy).

---

## 5. Ollama cloud LLM

**Confirmed facts** (from `docs.ollama.com/cloud` + community benchmarks, 2025-11):

- `gpt-oss:120b-cloud` is a 116B MoE with 5.1B active, **131k context**, **MXFP4 quantization**.
- Three modes: local, hybrid (local proxy), cloud (direct `https://ollama.com`).
- Cloud p50 ~1.8s end-to-end for typical prompts; streaming supported.
- Free tier: generous cap, **1 concurrent model**, 5h session window, weekly reset.
- Pro $20/mo: 50× free quota, 3 concurrent; Max $100/mo: 250× free, 10 concurrent.
- Privacy: cloud does not retain prompts; no training on user data.
- API surface: same as local Ollama (`/api/chat`, `/api/generate`); OpenAI-compat `/v1/*` available.
- Errors: 429 has `Retry-After`; 402 = payment required.

**Kairon rules:**
1. Never use the LLM for numeric prediction or numeric evaluation.
2. Use it for: research synthesis, hypothesis generation, feature ideation, summarization, evidence-grounded explanation, anomaly commentary, decision support.
3. Always cite the inputs (computed model output, retrieved evidence) that grounded any LLM-produced recommendation.
4. The client is wrapped in a typed `OllamaClient` pydantic schema; all calls are logged and replayable.

---

## 6. UX patterns for traders

Patterns we want to use (from research + trader-platform conventions):

1. **Pinned watchlist with sparkline + confidence badge** — at-a-glance state.
2. **Single-asset deep-dive screen** with three columns: forecast | evidence | risk.
3. **Confidence slider** (R-002's T threshold) — user trades coverage for accuracy live.
4. **Regime chip** (Trending / Ranging / Volatile / Stressed) — always visible.
5. **Explain-this-signal** (LLM-driven, evidence-grounded).
6. **Compare-this-asset** — side-by-side with same horizon/horizon.
7. **Alert manager** — typed alert rules; never spammy.
8. **No fake precision**: show probability bands, not point estimates; show "no signal" honestly.

**Avoid:** dashboard bloat, 12 widgets on one screen, jargon without tooltips, fake "AI confidence" scores without calibration, "Buy now" red/green screaming buttons.

---

## 7. Summary: Kairon's chosen external stack

| Layer | Choice | Why |
|-------|--------|-----|
| Crypto data | **CCXT** (Binance, Bybit, Coinbase primary) | Unified async, WebSocket, rate-limit aware |
| Stock data | **Tiingo + Polygon + TWRR (bulk)** | EOD + intraday + 1950-2024 |
| Macro | **FRED** | Free, reliable, 800k+ series |
| News | **CryptoPanic + Tiingo News + GDELT** | Crypto, stocks, global |
| On-chain | **Glassnode Studio (free API)** | MVRV/NUPL/SOPR |
| Social sentiment | **FinBERT (ProsusAI) + FinGPT v3.x for fine-tune** | Best F1 / cost ratio |
| Storage | **parquet (pyarrow) + duckdb** | Fast, typed, local-first |
| ETL | **polars + pydantic v2** | Strict types, fast |
| ML | **scikit-learn + xgboost + lightgbm + pytorch** | Standard, typed, well-supported |
| DL TS | **PatchTST + iTransformer + N-HiTS + LSTM + Decision Transformer** | 2024 SOTA |
| Backtest | **backtesting.py + vectorbt** | Fast, flexible |
| Portfolio | **Riskfolio-Lib + PyPortfolioOpt** | Drawdown-aware, classical MVO |
| Eval | **walk-forward + purged/embargo + DSR + PBO** | Scientifically defensible |
| LLM | **Ollama cloud (`gpt-oss:120b-cloud`)** | Long context, generous free tier |
| Type | **pyright --strict + ruff + mypy (defense-in-depth)** | Required by user |
| Config | **pydantic-settings + YAML** | Typed env, deterministic |
| Test | **pytest + hypothesis** | Property-based for invariants |
| Experiment | **mlflow + optuna** | Tracking + HPO |
| Logs | **loguru + structlog** | Structured |
| API | **fastapi + pydantic v2** | Typed, async |
| UI | (TBD) | Tauri or web — see Phase 7 |
