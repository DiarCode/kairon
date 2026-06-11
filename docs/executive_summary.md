# Kairon — Executive Summary

**Date:** 2026-06-05

## What Kairon is

A strictly-typed, scientifically honest, cost-aware AI market analysis and prediction platform for short-interval crypto and US equity trading, with:
- an architecture-diverse ML ensemble (LR + RF + XGB + LGBM + LSTM + Decision Transformer + PatchTST + iTransformer + N-HiTS + GARCH),
- confidence-thresholded inference with a user-tunable accuracy/coverage trade-off,
- a real walk-forward + purging + embargo + DSR + PBO evaluation pipeline,
- a typed Ollama-cloud LLM layer that explains signals without ever producing a number that drives a trade,
- and a fast, honest product UX that puts coverage next to accuracy and shows "no signal" as a valid answer.

## What Kairon is NOT

- A "90% accurate" oracle. That number is not defensible under valid evaluation.
- A system that uses random k-fold on time series (we forbid it in code).
- A system that lets the LLM predict prices (we forbid it in code and test).
- A system that quotes accuracy without coverage, regime, and cost-aware Sharpe.

## Honest verdict on the 90% objective

**90% raw directional accuracy is not a defensible target** for any horizon or asset under scientifically valid evaluation. The local research shows 60% on S&P 500 daily (R-001), 71-77% on 5-min crypto *with strict confidence filtering and ~30-60% coverage* (R-002), and external evidence converges on 53-62% as the realistic ceiling for daily equity and 1h crypto. Kairon replaces the 90% target with a multi-metric objective: cost-aware Sharpe (DSR-corrected), Brier/ECE, drawdown, regime breakdown, and a coverage-conditioned accuracy table.

## Top 5 ranked best-practice findings

1. **Architecture diversity beats dataset diversity** (R-001, p<0.05). Combine LR + tree + RNN + Transformer; do not just scale the same model.
2. **Walk-forward + purging + embargo is the only valid backtest harness** (López de Prado 2018; Bhand & Joshi 2026; Wang & Ruf 2022).
3. **Cost-aware by default.** A 60% accurate model is *unprofitable* with naive daily trading (R-001 V.F).
4. **Confidence thresholding is a first-class product primitive** (R-002). Show coverage next to accuracy; let the user trade the two.
5. **LLM is a reasoning layer, not a numeric oracle.** Every LLM call is typed, cited, and forbidden from producing a number that drives a trade.

## Top 5 ranked external picks

1. **CCXT** for crypto data (unified, async, WebSocket).
2. **TWRR + Polygon + Tiingo** for stocks (bulk historical + live + EOD + news).
3. **Riskfolio-Lib** for portfolio optimization (drawdown measures, robust opt).
4. **vectorbt + backtesting.py** for backtest (sweep + iteration).
5. **Ollama cloud `gpt-oss:120b-cloud`** for the LLM layer (131k context, fast, generous free tier, p50 ~1.8s).

## Stack summary

`uv` · `pyright --strict` · `pydantic v2` · `ruff` · `polars` + `pyarrow` + `duckdb` · `scikit-learn` + `xgboost` + `lightgbm` + `pytorch` (PatchTST, iTransformer, N-HiTS, Decision Transformer) · `arch`/`statsmodels` (GARCH) · `ccxt` (crypto) · `httpx` + `tenacity` + `orjson` (network) · `fastapi` (API) · `mlflow` + `optuna` (experiment) · `quantstats` (reports) · `ollama` (LLM) · `pytest` + `hypothesis` (tests) · `loguru` (logs).

## Phased roadmap at a glance

| Phase | Duration | What ships |
|-------|----------|------------|
| 0 Skeleton | 1-2 d | uv + pyright strict + CI green |
| 1 Data | 3-5 d | Real crypto + stock ingestion, QC, parquet |
| 2 Features | 5-7 d | All 20 indicators, regime, on-chain, sentiment |
| 3 Labels & splits | 3-4 d | Leakage-tested, walk-forward + purging + embargo |
| 4 Models v1 | 7-10 d | LR + RF + XGB + LGBM + LSTM + GARCH ensemble |
| 5 Backtest + eval | 5-7 d | Cost-aware, DSR, PBO, ablation, regime breakdown |
| 6 API | 5-7 d | fastapi with typed DTOs |
| 7 UI | (parallel) | Watchlist, asset detail, compare, alerts, settings |
| 8 LLM layer | 3-4 d | Ollama cloud, cited explanations, no-numeric guardrail |
| 9 Live inference | 5-7 d | WS data, drift detection, alerts |
| 10 Diego UX | 3-5 d | Reproducible runs, run-vs-run diff |
| 11 Deep TS | 7-10 d | PatchTST, iTransformer, N-HiTS, Decision Transformer |
| 12 Paper trade | 2-4 wk | 30-day paper trade, drift gate |
| 13 Canary live | (optional) | Tiny size, conservative risk, weekly review |
| 14 Docs & community | ongoing | ADRs, public benchmarks |

## The 5 unresolved research questions Kairon will track

1. Do **iTransformer** / **PatchTST** generalize from standard benchmarks to finance? (Plausible, unconfirmed.)
2. Does **conformal prediction** on top of an architecture-diverse ensemble give stronger calibration than either alone? (Plausible.)
3. Do **on-chain + sentiment** features reduce error for crypto beyond what technicals alone achieve? (Plausible, modest.)
4. Does **Decision Transformer** as a sequence modeler beat purpose-built volatility models? (R-001 found competitive on VIX; needs replication.)
5. Does the **iTransformer** cross-asset encoder beat per-asset models on portfolio-level forecasting? (Plausible but unconfirmed.)

## Bottom line

Kairon is built to be the strongest *honest* system, not the loudest. The single biggest design decision is to **replace the 90% accuracy target with a multi-metric, cost-aware, regime-aware, coverage-aware objective hierarchy**, and to enforce walk-forward + purging + embargo + DSR + PBO in code. Every other choice flows from that.
