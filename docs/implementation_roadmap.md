# Implementation Roadmap — Kairon

**Date:** 2026-06-05
**Format:** Numbered phases with explicit exits. Each phase ships a runnable artifact and is gated by the prior phase's tests.

## Phase 0 — Skeleton (1-2 days)
- `uv init` → `pyproject.toml` with strict deps.
- `pyrightconfig.json` set to `strict`.
- `ruff.toml` set to project standards.
- `pre-commit` with ruff + pyright.
- `AGENTS.md` with build/test/lint recipes.
- Empty `src/kairon/` package with `py.typed`.
- CI workflow file.
- Empty `tests/` with a passing sanity test.

**Exit:** `uv run pytest` and `uv run pyright` pass on an empty tree.

## Phase 1 — Data layer (3-5 days)
- `kairon.data.symbols` (typed canonical symbol).
- `kairon.data.io` (parquet + duckdb).
- `kairon.data.adapters.ccxt` (Binance, Bybit, Coinbase).
- `kairon.data.adapters.tiingo` + `polygon` (skeleton).
- `kairon.data.adapters.fred` (real).
- `kairon.data.diagnostics` (QC checks).
- `kairon.data.ingestion` (orchestration).
- Reference historical load: TWRR for stocks; CryptoDataDownload for crypto.

**Exit:** Can download a single crypto symbol, write to parquet, run diagnostics, and pass.

## Phase 2 — Features (5-7 days)
- `kairon.features.technical` — implement every indicator in the user table (EMA, SMA, MACD, ADX, Ichimoku, RSI, Stochastic, Williams %R, CCI, Bollinger, ATR, OBV, VWAP).
- `kairon.features.technical.structure` — BOS/CHoCH, Fibonacci, candlestick patterns.
- `kairon.features.regime` — HMM + rule-based regime.
- `kairon.features.onchain` — Glassnode adapter.
- `kairon.features.sentiment` — FinBERT wrapper.
- `kairon.features.pipeline.FeaturePipeline` (typed, deterministic).

**Exit:** Each indicator has a unit test against a known reference (e.g., TA-Lib).

## Phase 3 — Labels & splits (3-4 days)
- `kairon.labels.direction`, `.magnitude`, `.volatility`, `.triple_barrier`.
- Leakage tests for each label spec.
- `kairon.splits.walkforward` (typed).
- `kairon.splits.purged` (typed).
- `kairon.splits.embargo` (typed).
- `kairon.splits.cpcv` (typed, for PBO).

**Exit:** A small fixture runs through the splits layer and all leakage tests pass.

## Phase 4 — Models v1 (7-10 days)
- `kairon.models.linear.LogisticRegression` (anchor).
- `kairon.models.tree.RandomForest`, `.XGBoost`, `.LightGBM`.
- `kairon.models.lstm.LSTM`.
- `kairon.models.garch.GARCH`.
- `kairon.models.ensemble.ArchitectureDiverseEnsemble` (top-K majority vote + conf weighting).
- `kairon.calibration.isotonic` (typed).

**Exit:** Ensemble produces calibrated probabilities on a small fixture, with mlflow tracking.

## Phase 5 — Backtest + eval (5-7 days)
- `kairon.backtest.engine` (backtesting.py wrapper).
- `kairon.backtest.vector_engine` (vectorbt wrapper).
- `kairon.backtest.cost_model.CostModel` (pydantic).
- `kairon.backtest.execution` (slippage, market impact).
- `kairon.backtest.reports` (quantstats + JSON).
- `kairon.evaluation.metrics` (typed).
- `kairon.evaluation.dsr` (typed).
- `kairon.evaluation.pbo` (typed).
- `kairon.evaluation.regime_breakdown`.
- `kairon.evaluation.ablation` (typed record).

**Exit:** Walk-forward backtest of the v1 ensemble produces a DSR, PBO, CAS, regime breakdown, and ablation JSON on a small fixture.

## Phase 6 — API + persistence (5-7 days)
- `kairon.api.app` (fastapi).
- Routes: market, signal, backtest, alert, explain.
- pydantic v2 DTOs.
- DuckDB for ad-hoc.
- Auth (simple API key for v1).

**Exit:** `GET /signal/{symbol}?horizon=1h` returns a typed signal JSON with full provenance.

## Phase 7 — UI (separate timeline, depends on API)
- Tauri or React web app consuming the API.
- Reuse `docs/information_architecture.md` and `docs/user_flows.md`.
- Honest "no signal" path; confidence slider; compare screen.

**Exit:** A user can browse the watchlist, open an asset, see forecast/evidence/risk, place a paper trade, and set an alert.

## Phase 8 — LLM reasoning layer (3-4 days)
- `kairon.llm.client.OllamaClient` (typed).
- `kairon.llm.prompts` (versioned).
- `kairon.llm.grounding` (citation enforcement).
- `kairon.llm.explainer` (signal explanation).
- `kairon.research.agent` (read-only planner).
- CI tests for prompt contracts and "no-numeric" guardrail.

**Exit:** `GET /signal/{symbol}?explain=true` returns a grounded, citation-bearing explanation.

## Phase 9 — Live inference + alerting (5-7 days)
- Live data adapters (CCXT WebSocket, Polygon WS).
- Live feature computation.
- Live model server.
- Alert engine.
- Notification channels (in-app, email, webhook).
- Stale-data detection.
- Drift detection (PSI on top features, ECE on calibration).

**Exit:** A user can subscribe to "BTC-USDT, 1h, T≥1.4, regime in {Trending, Volatile}" and receive alerts.

## Phase 10 — Research notebooks + Diego UX (3-5 days)
- Reproducible runs (config + data hash → exact outputs).
- Compare-across-runs UI.
- Diagnostic dashboards (DSR, PBO, ECE).
- JSON diff between runs.

**Exit:** Diego persona can re-run a backtest, see the diff vs the prior version, and export a JSON.

## Phase 11 — Models v2: deep time-series (7-10 days)
- PatchTST, iTransformer, N-HiTS, Decision Transformer.
- Per-horizon model selection via walk-forward.
- On-chain encoder integration.

**Exit:** PatchTST, iTransformer, N-HiTS, Decision Transformer are in the ensemble and pass the same CI gates.

## Phase 12 — Paper trading + drift (2-4 weeks)
- Run paper trading for 30 days.
- Compare live PnL vs backtest expectations.
- Auto-quarantine models that drift.

**Exit:** All "strong" milestone models pass the paper-trade gate (CAS within 1.5 std, ECE drift < 0.03).

## Phase 13 — Canary live (optional, user-driven)
- Tiny size, conservative risk.
- Weekly review.
- Auto-pause on drawdown > 12% or DSR < 0.95.

**Exit:** Sustained 12 months of CAS ≥ 1.5, drawdown < 12%.

## Phase 14 — Documentation & community (ongoing)
- ADR backlog groomed.
- `docs/` kept current.
- Public benchmark suite so external researchers can reproduce.

---

## Cross-cutting work (continuous)

- **Test hardening:** add a new leakage test for every leak pattern found in literature.
- **Dataset coverage:** keep adding new exchanges / sources as we can integrate them safely.
- **Performance:** profile, vectorize, parallelize.
- **Security:** secret rotation, key custody, redaction review.
