# API / Dataset / Library Catalog — Kairon

**Date:** 2026-06-05

A condensed reference index. Detailed evaluations are in `external_research.md`; this file is the *at-a-glance* roster.

## A. Data sources

### A.1 Market OHLCV (crypto)
- **Binance** — `data.binance.vision` (free historical), `https://api.binance.com` (live)
- **Bybit v5** — `https://api.bybit.com`
- **Coinbase Advanced Trade** — `https://api.coinbase.com`
- **CoinGecko** (free tier, 2y)
- **CryptoDataDownload** — `cryptodatadownload.com`
- **CCXT aggregator** — unifies all of the above

### A.2 Market OHLCV (stocks/ETFs)
- **Polygon.io** — real-time + historical US
- **Tiingo** — EOD + fundamentals + news
- **Alpha Vantage** — free tier
- **TWRR** (Kaggle) — bulk historical 1950-2024
- **yfinance** — dev-only

### A.3 Macro
- **FRED** — `api.stlouisfed.org`
- **World Bank** — `api.worldbank.org`
- **ECB / BoE / IMF** — direct

### A.4 News & sentiment
- **CryptoPanic** — `cryptopanic.com/developers/api/`
- **Tiingo News** — `api.tiingo.com/tiingo/news`
- **GDELT** — `api.gdeltproject.org`
- **ProsusAI/finbert** — `huggingface.co/ProsusAI/finbert`
- **AI4Finance-Foundation/FinGPT** — `github.com/AI4Finance-Foundation/FinGPT`

### A.5 On-chain
- **Glassnode Studio** — `api.glassnode.com/v1` (free key, limited metrics)
- **CryptoQuant** — `api.cryptoquant.com/v1` (paid advanced)
- **Bitquery** — `graphql.bitquery.io` (free tier)

## B. Core Python libraries

| Domain | Primary | Secondary |
|--------|---------|-----------|
| DataFrames | **polars** | pandas |
| IO / storage | **pyarrow / parquet** | duckdb |
| Numerics | **numpy / scipy** | — |
| Plotting | **plotly** | matplotlib |
| ML | **scikit-learn** | — |
| GBT | **xgboost, lightgbm** | catboost |
| DL | **pytorch** | pytorch-forecasting, NeuralForecast |
| Stats | **statsmodels, arch** | scipy.stats |
| Backtest | **backtesting.py** | vectorbt, bt |
| Portfolio | **Riskfolio-Lib** | PyPortfolioOpt |
| Backtest reports | **quantstats, empyrical** | — |
| Web/API | **fastapi, httpx, orjson** | — |
| Validation | **pydantic v2** | attrs (compat) |
| Settings | **pydantic-settings** | — |
| Type check | **pyright --strict** | mypy --strict |
| Lint/format | **ruff** | — |
| Test | **pytest, hypothesis** | — |
| Experiment | **mlflow, optuna** | — |
| Logging | **loguru** | structlog |
| Scheduling | **APScheduler** | celery (v2) |
| HTTP retry | **tenacity** | — |
| Async DB | **SQLAlchemy 2 (async)** | SQLModel |
| WebSocket | **websockets / ccxt.pro** | — |
| LLM | **ollama (Python SDK)** | — |
| Repro | **uv** | — |

## C. Models to ship in v1

| Model | Use | Horizon |
|-------|-----|---------|
| Logistic Regression (regularized) | Baseline | all |
| Random Forest (2-model conf ensemble) | R-002 style | 5m / 15m / 1h / 1d |
| XGBoost | Strong tabular | 1d / 1h |
| LightGBM | Strong tabular, fast | 1d / 1h |
| LSTM (small) | Sequential | 5m / 1h |
| Decision Transformer | Conditional sequence | 1d |
| PatchTST | Long-horizon univariate | 1d+ |
| iTransformer | Multivariate | 1d portfolio |
| N-HiTS | Multi-scale | 1d vol |
| GARCH (via `arch`) | Vol baseline | 1d |

## D. Models explicitly NOT in v1

| Model | Why deferred |
|-------|--------------|
| Hybrid quantum (VQC) | Cost-benefit poor; classical ML gives same gain |
| Reinforcement learning | High variance, hard to evaluate |
| GANs / VAEs for finance | Marginal evidence |
| Heavy LLMs (BloombergGPT-scale) | Cost; we use FinBERT + Ollama cloud instead |
| Informer / Autoformer | Outperformed by PatchTST/iTransformer on most benchmarks |
