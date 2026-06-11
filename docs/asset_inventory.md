# Asset Inventory — Kairon (AI Trading Research Platform)

**Date:** 2026-06-05
**Scope:** Local `@researches` and `@datasets` folders + user-supplied indicator table.

## 1. Local research materials

### R-001 — "Hybrid Quantum-Classical Ensemble Learning for S&P 500 Directional Prediction" (Weinberg, AI-WEINBERG)

- **Type:** Markdown paper
- **Size:** ~71 KB
- **Domain:** Daily S&P 500 directional classification (binary)
- **Period:** 2020-01 to 2023-12 (1006 trading days)
- **Instruments:** S&P 500, VIX, Gold, XLF, XLK, HYG, IWM
- **Architectures compared:** LSTM, Decision Transformer, XGBoost, Random Forest, Logistic Regression, plus a 4-qubit variational quantum feature extractor
- **Features (27):** price-based, rolling volatility (5/10/20d), momentum (ROC), SMA/EMA, Bollinger, RSI, quantum-sentiment (4 features)
- **Split:** Single 70/30 temporal split (704 train / 286 test)
- **Headline result:** Top-7 architecture-diverse ensemble → **60.14%** accuracy (95% CI [56.84, 63.44]); McNemar p<0.05; Sharpe ~1.2 with 6+/7 consensus filtering
- **Top individual:** VIX-LSTM 57.04%
- **Naive 35-model ensemble:** 51.20% (worst than random — proves filtering is essential)
- **Quantum contribution:** +0.82% to ensemble (avg +1.09% per model)
- **Regime behavior:** best at VIX 25-35 (66.67%), degrades at VIX>35 (54.55%) — limits generalization under extreme stress

**Quality concerns:**
1. Single temporal 70/30 split → no walk-forward, risk of regime overfitting.
2. 35 models then reported best subset — selection bias.
3. Quantum features simulated classically → no genuine quantum advantage.
4. Sharpe 1.2 reported from preliminary backtest without slippage or live impact.
5. No stocks other than index; no crypto.

**Use in Kairon design:** Validate "architecture diversity > dataset diversity" as a principle; justify 52% quality floor; use as benchmark ceiling for S&P 500 daily.

### R-002 — "Improving Cryptocurrency Price Direction Forecasting via Confidence Score Thresholds on Random Forest Ensembles" (Csanadi & Lennartsson, BTH 2024)

- **Type:** Master thesis
- **Size:** ~96 KB
- **Domain:** 5-minute crypto OHLCV directional classification
- **Periods:** 2017-07-17 → 2018-01-17 (BTC only); 2022-01-01 → 2023-12-31 (10 coins)
- **Method:** Two parallel RF classifiers (RF1, RF2) on the same features, retrained on a rolling 7-day window with a 5-hour test window, N=2016 / K=60, threshold T on summed probability.
- **Indicators used:** ATR, ADX, BOP, SMA, EMA, %return, log return, Lyapunov exponent, MACD, RSI, volatility, volume (out of 43 considered)
- **Target:** Direction at t + 60 bars (= 5h on 5-min bars, 1h on 1-min bars)
- **Results (multi-coin average 2022-23):** 71.30% @ T=1.3 (56.5% data remaining), 77.40% @ T=1.5 (32.1% remaining), 88.50% @ T=1.8 (6.7% remaining), 93.0% @ T=1.9 (1.8% remaining)
- **Justification frame:** Adaptive Markets Hypothesis (Lo, 2004) — appropriate for less-mature crypto markets.

**Quality concerns:**
1. RF1+RF2 share architecture; correlation likely ≥0.6 (limited diversity vs. R-001 finding).
2. High accuracy at high T is conditional on extreme selectivity — not a generalizable skill claim.
3. Threshold T is not cross-validated; selection bias risk.
4. Only 2 horizons tested.
5. No transaction-cost-aware trading; no live paper trading.

**Use in Kairon design:** Concrete confidence-thresholding pattern (sum-of-probabilities), the rolling retraining cadence, the explicit accuracy-vs-coverage trade-off as a UX primitive, the AMH framing for crypto, and a realistic 60-75% accuracy ceiling on 5-min crypto.

## 2. Local dataset assets

### D-001 — `DATASETS.md`: 10 public crypto dataset index

The only local dataset file is a catalog of 10 external public crypto datasets (no files downloaded). It is the recommended starting set for download during Phase 1 of the implementation.

| # | Dataset | Size | Since | Best for |
|---|---------|------|-------|----------|
| 1 | Binance Full History (tick) | ~28 GB | 2021 | HFT / microstructure |
| 2 | Bitcoin Historical 2018-2026 (Binance) | ~50-200 MB | 2018 | TA / DL forecasting |
| 3 | BTCUSD 1-Min (7 exchanges) | ~585 MB | 2017 | arbitrage / multi-venue |
| 4 | Integrated Crypto Data (80+ coins) | ~500 MB-1 GB | 2012 | multi-asset |
| 5 | CryptoDataDownload OHLCV | ~100-500 MB/coin | 2015 | multi-timeframe |
| 6 | Bitcoin Full-History Network (graph) | ~10-50 GB | 2009 | on-chain / GNN |
| 7 | TWRR Financial Market Dataset (10k+ assets) | ~2-3 GB | 1950 | stocks + cross-asset |
| 8 | Bitcoin Arbitrage (multi-exchange) | ~100-300 MB | varies | arbitrage |
| 9 | Crypto Twitter/Reddit Sentiment | ~1-5 GB | 2019 | sentiment |
| 10 | CoinGecko Historical API | on-demand | 2014 | broad market |

## 3. User-supplied indicator matrix

The user supplied a 20-row technical-indicator table with `category / indicator / best timeframe / crypto? / stocks? / strength`. This is treated as the **initial indicator shortlist** for the feature library. Notable points:

- **Trend:** EMA (5, 50, 200), SMA Crossover, MACD, ADX, Ichimoku Cloud
- **Momentum:** RSI, Stochastic, Williams %R, CCI
- **Volatility:** Bollinger Bands, ATR
- **Volume:** Volume Profile, VWAP/AVWAP, OBV, CVD/Delta
- **Pattern/Structure:** Fibonacci, Candlestick Patterns, BOS/CHoCH (SMC), Wyckoff
- **On-chain:** Glassnode/MVRV (crypto only)

## 4. Gaps detected in local assets

| Gap | Implication | Mitigation in Kairon |
|---|---|---|
| No LLM / RAG layer in local research | The user requires Ollama cloud | We add an LLM-as-reasoning layer for explanation, evidence synthesis, agent planning |
| No purged/walk-forward evaluation | Risk of inflated metrics | We mandate walk-forward + purging + embargo from Phase 1 |
| No slippage model | Backtest overstates profitability | We add a realistic cost & slippage model in `evaluation` |
| No stocks data file | TWRR is the candidate; download required | Add a data-ingest stage that downloads TWRR (Phase 1) |
| No live on-chain data pipeline | On-chain signals not currently wired | Build `data/onchain.py` with CryptoQuant/Glassnode adapters |
| No UX / persona work | Product has to be designed from scratch | Add `product/` design phases with personas and primary screens |
| No ADR-style architecture record | Decisions need to be auditable | Add `docs/adr/` with Nygard-style records |

## 5. Coverage check vs. required phases

| Phase | Local asset coverage | Additional work required |
|-------|----------------------|---------------------------|
| 1. Inventory & parse | ✅ This document | Maintain `asset_manifest.json` |
| 2. Research synthesis | ✅ Both papers digested | Build `literature_matrix.md` |
| 3. External research | 🟡 Partial (from local + indicators) | Run external research (this design) |
| 4. Dataset audit | 🟡 Catalog only | Per-dataset schema/coverage audit |
| 5. Objective reformulation | ❌ Not in local | We design it from first principles |
| 6. Best-practice system design | 🟡 Two papers | Synthesize with external evidence |
| 7. UX | ❌ Not in local | Design from scratch |
| 8. Architecture | ❌ Not in local | Design from scratch (uv + pyright strict + pydantic v2) |
| 9. Evaluation framework | 🟡 Implicit | Build explicit purged/walk-forward + DSR framework |
| 10. Implementation roadmap | ❌ Not in local | Design from scratch |

Legend: ✅ = done, 🟡 = partial, ❌ = missing
