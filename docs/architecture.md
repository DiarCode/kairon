# Technical Architecture — Kairon

**Date:** 2026-06-05
**Stack:** Python 3.12+, `uv`, `pyright --strict`, `ruff`, `pydantic v2`, fastapi, polars, pyarrow/duckdb, scikit-learn, xgboost, lightgbm, pytorch, mlflow, optuna, ollama.

## 1. Guiding principles

1. **Strict typing everywhere.** All public APIs are typed; `pyright --strict` is the gate.
2. **No hidden magic.** Every component has explicit inputs, outputs, and contracts.
3. **Determinism where it matters.** All randomness is seeded and recorded.
4. **Replayable.** Every prediction and every backtest is reproducible from a config + data version hash.
5. **LLM is not a numeric oracle.** LLMs reason, summarize, and explain. They never own a number that drives a trade.
6. **Methodological rigor is enforced by code.** Walk-forward, purging, embargo, DSR, PBO are not optional — they are CI gates.
7. **Cost-aware by default.** Every backtest and every trade is cost-aware.
8. **Typed schemas at every boundary.** Network IO, file IO, env, DB — all pydantic.

## 2. Repository layout

```
kairon/
├── pyproject.toml            # uv-managed, all deps
├── uv.lock
├── README.md
├── AGENTS.md                 # build/test/lint recipes
├── .python-version           # 3.12
├── .gitignore
├── .pre-commit-config.yaml   # ruff + pyright hooks
├── configs/                  # typed YAML configs (pydantic-loaded)
│   ├── base.yaml
│   ├── data/
│   │   ├── crypto.yaml
│   │   └── stocks.yaml
│   ├── features/
│   │   ├── trend.yaml
│   │   ├── momentum.yaml
│   │   ├── vol.yaml
│   │   └── sentiment.yaml
│   ├── models/
│   │   ├── lstm.yaml
│   │   ├── patchtst.yaml
│   │   ├── itransformer.yaml
│   │   └── ensemble.yaml
│   ├── backtest/
│   │   └── walkforward.yaml
│   ├── eval/
│   │   ├── dsr.yaml
│   │   └── pbo.yaml
│   └── llm/
│       └── ollama.yaml
├── data/                     # local-only; parquet, partitioned
├── models/                   # local-only; trained weights + meta
├── reports/                  # generated artifacts
├── artifacts/                # generated JSON / manifests
├── evals/                    # CI-evals + benchmarks
├── docs/
│   ├── architecture.md       # this file
│   ├── adr/                  # architecture decision records
│   ├── repo_structure.md
│   └── ...                   # all design docs
├── src/
│   └── kairon/               # the actual package
│       ├── __init__.py
│       ├── py.typed          # PEP 561 marker
│       ├── config/           # typed config loader
│       │   ├── __init__.py
│       │   ├── settings.py   # pydantic-settings
│       │   ├── loader.py
│       │   └── schema.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── symbols.py
│       │   ├── calendar.py
│       │   ├── io.py         # parquet / duckdb
│       │   ├── adapters/
│       │   │   ├── __init__.py
│       │   │   ├── ccxt_adapter.py
│       │   │   ├── polygon.py
│       │   │   ├── tiingo.py
│       │   │   ├── fred.py
│       │   │   ├── glassnode.py
│       │   │   ├── cryptopanic.py
│       │   │   └── gdelt.py
│       │   ├── diagnostics.py
│       │   └── ingestion.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── technical/
│       │   │   ├── trend.py
│       │   │   ├── momentum.py
│       │   │   ├── volatility.py
│       │   │   ├── volume.py
│       │   │   └── structure.py
│       │   ├── cross_asset.py
│       │   ├── regime.py
│       │   ├── onchain.py
│       │   ├── sentiment.py
│       │   └── pipeline.py
│       ├── labels/
│       │   ├── __init__.py
│       │   ├── direction.py
│       │   ├── magnitude.py
│       │   ├── volatility.py
│       │   └── triple_barrier.py
│       ├── splits/
│       │   ├── __init__.py
│       │   ├── walkforward.py
│       │   ├── purged.py
│       │   ├── embargo.py
│       │   └── cpcv.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── linear.py
│       │   ├── tree.py
│       │   ├── lstm.py
│       │   ├── decision_transformer.py
│       │   ├── patchtst.py
│       │   ├── itransformer.py
│       │   ├── nbeats.py
│       │   ├── garch.py
│       │   └── ensemble.py
│       ├── calibration/
│       │   ├── __init__.py
│       │   └── isotonic.py
│       ├── policy/
│       │   ├── __init__.py
│       │   ├── sizer.py
│       │   ├── stops.py
│       │   └── rules.py
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py        # backtesting.py wrapper
│       │   ├── vector_engine.py # vectorbt wrapper
│       │   ├── cost_model.py
│       │   ├── execution.py
│       │   └── reports.py       # quantstats
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── dsr.py
│       │   ├── pbo.py
│       │   ├── calibration.py
│       │   ├── regime_breakdown.py
│       │   └── ablation.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py        # ollama wrapper
│       │   ├── prompts.py
│       │   ├── grounding.py
│       │   └── explainer.py
│       ├── research/
│       │   ├── __init__.py
│       │   ├── agent.py         # ollama-driven planner
│       │   └── synthesizer.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py           # fastapi
│       │   ├── deps.py
│       │   ├── routes/
│       │   │   ├── market.py
│       │   │   ├── signal.py
│       │   │   ├── backtest.py
│       │   │   ├── alert.py
│       │   │   └── explain.py
│       │   └── schemas/         # pydantic DTOs
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   ├── tracing.py
│       │   └── metrics.py
│       ├── experiment/
│       │   ├── __init__.py
│       │   ├── tracker.py       # mlflow wrapper
│       │   └── registry.py
│       └── cli/
│           ├── __init__.py
│           └── main.py          # typer / click
└── tests/
    ├── conftest.py
    ├── data/
    ├── features/
    ├── labels/
    ├── splits/                  # leakage tests
    ├── models/
    ├── calibration/
    ├── backtest/
    ├── evaluation/
    ├── llm/
    └── api/
```

## 3. Domain model (typed, pydantic v2)

```python
# src/kairon/data/symbols.py
from typing import Literal, NewType
from pydantic import BaseModel, Field

Symbol = NewType("Symbol", str)
Venue = Literal["binance", "bybit", "coinbase", "polygon", "tiingo", "yfinance"]

class CryptoSymbol(BaseModel):
    base: str = Field(min_length=1, max_length=16)
    quote: str = Field(min_length=1, max_length=16)
    venue: Venue
    market: Literal["spot", "perp"] = "spot"
    @property
    def canonical(self) -> str:
        return f"{self.base}-{self.quote}" + ("-PERP" if self.market == "perp" else "")
```

```python
# src/kairon/labels/schema.py
from datetime import datetime
from pydantic import BaseModel, Field

class LabelSpec(BaseModel):
    horizon: str = Field(pattern=r"^[0-9]+(m|h|d|w)$")  # 5m, 1h, 1d, 1w
    kind: Literal["direction", "magnitude", "volatility", "triple_barrier"]
    params: dict[str, float] = Field(default_factory=dict)

class LabeledBar(BaseModel):
    symbol: str
    ts: datetime  # UTC
    horizon: LabelSpec
    y: int | float
    y_class: int | None = None
    meta: dict[str, float] = Field(default_factory=dict)
```

```python
# src/kairon/llm/schema.py
from pydantic import BaseModel, Field
from typing import Literal

class LLMRequest(BaseModel):
    model: str = Field(default="gpt-oss:120b-cloud")
    task: Literal["explain", "summarize", "synthesize", "plan"]
    system: str
    user: str
    citations_required: bool = True
    max_tokens: int = 1024
    temperature: float = 0.2

class LLMResponse(BaseModel):
    text: str
    cited_inputs: list[str] = Field(default_factory=list)
    raw: dict
```

## 4. Service boundaries

| Service | Owns | Talks to |
|---------|------|----------|
| `kairon.data` | Ingestion, normalization, QC | file system + external APIs |
| `kairon.features` | Pure compute over bars | `kairon.data` |
| `kairon.labels` | Label construction, leakage tests | `kairon.features` |
| `kairon.splits` | Walk-forward, purging, embargo, CPCV | `kairon.labels` |
| `kairon.models` | Model zoo + ensemble | `kairon.features`, `kairon.splits` |
| `kairon.calibration` | Probability calibration on a held-out fold | `kairon.models` |
| `kairon.policy` | Trade construction (sizing, stops) | `kairon.calibration`, `kairon.features` |
| `kairon.backtest` | Simulation with cost model | `kairon.policy` |
| `kairon.evaluation` | DSR, PBO, calibration, regime breakdown | `kairon.backtest` |
| `kairon.llm` | Ollama client + prompt contracts | `kairon.models`, `kairon.evaluation` |
| `kairon.research` | LLM-driven planner (read-only) | `kairon.llm`, all read-only services |
| `kairon.api` | fastapi HTTP surface | all services |
| `kairon.observability` | Logging, tracing, metrics | cross-cutting |
| `kairon.experiment` | mlflow + optuna | `kairon.models` |

## 5. Configuration strategy

- `pyproject.toml` for dependencies (`uv`).
- Typed YAML configs in `configs/`, loaded via pydantic v2 into immutable config objects.
- Environment / secrets via `pydantic-settings` (`KAIRON_*` env vars).
- No silent defaults: every config field has either a YAML value, an env value, or an explicit `Field(...)` required flag.

## 6. Training pipeline (typed, reproducible)

1. **Resolve config** → typed `RunConfig`.
2. **Load data** via `kairon.data` (with QC + version hash).
3. **Build features** via `kairon.features.pipeline.FeaturePipeline` (typed, deterministic, seedable).
4. **Build labels** via `kairon.labels` (typed, with leakage tests).
5. **Build splits** via `kairon.splits.walkforward` (typed, with purging+embargo).
6. **Train models** via `kairon.models` (each model has a `fit(features, labels, splits) -> TrainedModel` contract).
7. **Calibrate** via `kairon.calibration` on a held-out fold.
8. **Record** every run in `mlflow` (config, hash, metrics, model, feature importance).
9. **Export** to `models/{name}/{version}/` (parquet features + model weights + meta.json).
10. **CI gates** run on the calibration fold: PBO, DSR, regime breakdown, leakage audit.

## 7. Backtest pipeline

1. **Build a `BacktestConfig`** from a run + a cost model.
2. **Walk-forward:** for each fold, use the model from that fold; never the final.
3. **Apply cost model** (commission, spread, market impact, funding).
4. **Output:** equity curve, drawdown, hit rate, accuracy-at-coverage, per-trade PnL.
5. **Evaluate:** cost-aware Sharpe, Sortino, Calmar, CVaR, DSR, PBO, regime breakdown.
6. **Tear sheet:** quantstats HTML + Kairon-native JSON for the UI.

## 8. Inference pipeline

1. **Resolve config** + model version.
2. **Pull live data** (CCXT for crypto, Polygon for US).
3. **Compute features** (same code as training — no drift).
4. **Predict per horizon** (5m / 15m / 1h / 4h / 1d).
5. **Calibrate** probabilities.
6. **Apply regime filter** if requested.
7. **Build signal** (direction, magnitude, vol, confidence band).
8. **Optional LLM explanation** (grounded; never numeric).
9. **Log + alert** if rule fires.

## 9. API layer (fastapi)

- Typed request/response via pydantic v2 (`model_config` + `Field`).
- Async-first (`httpx` upstream, async DB if added).
- All endpoints return a `Response[T]` envelope: `{"data": T, "meta": {...}}`.
- OpenAPI generated and committed.
- Rate-limited per user.
- WebSocket endpoint for live signals.
- Health endpoint with per-source status.

## 10. Storage

- **Raw** data: parquet, partitioned by venue/symbol/timeframe/yyyy/mm.
- **Processed** features: parquet, partitioned by symbol/horizon/yyyy.
- **Model registry:** filesystem under `models/{name}/{version}/` + mlflow.
- **Experiment metadata:** mlflow (local file backend for dev, server for prod).
- **DuckDB** as the in-process OLAP for ad-hoc queries.

## 11. Caching

- Feature cache: in-process LRU + on-disk parquet (versioned by config hash).
- HTTP cache: per-(endpoint, params, ts-range) with TTL.
- WebSocket: rotating buffer for L2 order book.

## 12. Observability

- **Logging:** `loguru` JSON, with `run_id`, `model_version`, `config_hash` injected.
- **Tracing:** OpenTelemetry spans per request, per pipeline stage.
- **Metrics:** Prometheus format, scraped per service.
- **Alerts:** on (a) data feed stale > 5m, (b) calibration drift > threshold, (c) retrain failure, (d) DSR drops below 0.95 on the canary fold, (e) order book depth below floor.

## 13. Secrets strategy

- `.env` for local dev (gitignored).
- `pydantic-settings` reads `KAIRON_*` env vars.
- CI: GitHub OIDC → cloud secret store.
- No secrets in code; no secrets in logs (redact in `kairon.observability`).

## 14. Test strategy

- **Unit tests:** every public function, every model, every label rule, every cost model, every metric.
- **Property tests:** `hypothesis` for invariants (e.g., `direction_label` is in {-1, 0, 1}).
- **Leakage tests:** `tests/splits/test_no_future_in_features.py`, `tests/splits/test_no_train_test_overlap.py`, `tests/splits/test_embargo_respects_serial_correlation.py`.
- **Regressions:** every shipped model has a baseline number in `evals/baselines/`; CI fails on regression.
- **Snapshot tests:** pydantic schemas, prompts, signal cards.

## 15. LLM integration rules

- `kairon.llm.client.OllamaClient` is the only thing that calls the SDK.
- Every LLM call has a typed `LLMRequest` and a `LLMResponse`.
- `citations_required: bool` is True for any user-facing call.
- The LLM never has access to a "make a trade" tool; it is read-only.
- A response without the required citations is **rejected**, not silently passed through.
- Latency budget: 2s p50; 10s p99 — fallback to static evidence card on timeout.

## 16. Cross-cutting concerns

- **Error handling:** typed exceptions (`KaironError`, `DataError`, `ModelError`, `EvalError`); no bare `except`.
- **Concurrency:** `asyncio` for IO, `concurrent.futures` for CPU-bound. No global state.
- **Time:** all timestamps UTC; no naive `datetime`; `pydantic` enforces `tzinfo`.
- **Randomness:** every random op takes a `Seed` and records it.
- **Reproducibility:** every output has a `provenance` field: `config_hash`, `data_hash`, `model_version`, `seed`.
