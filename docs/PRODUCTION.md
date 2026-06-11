# Production Runbook

This document describes how to **deploy, operate, and debug** a
production Kairon instance. It complements the architecture doc
(``docs/architecture.md``) and the ADRs (``docs/adr/``).

## Architecture summary

```
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   ──▶  │  data/      │  │ features/   │  │ labels/     │
        │  CCXT+parq  │─▶│  20 ind +   │─▶│  direction  │
        │             │  │  GMM regime │  │  magnitude  │
        └─────────────┘  └─────────────┘  └──────┬──────┘
                                                │
                                                ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │ splits/     │  │ models/     │  │  trainer    │
        │ WF+purge    │─▶│ LR/RF/XGB/  │─▶│  + mlflow   │
        │ CPCV        │  │ LGBM/LSTM/  │  │  + DSR/PBO  │
        │             │  │ N-BEATS/    │  │             │
        │             │  │ DeepEns     │  │             │
        └─────────────┘  └──────┬──────┘  └──────┬──────┘
                                │                 │
                                ▼                 ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │ api/        │  │ paper/      │  │ backtest/   │
        │ FastAPI +   │◀▶│  broker     │  │  engine +   │
        │  typed DTOs │  │  simulator  │  │  metrics    │
        └──────┬──────┘  └──────┬──────┘  └─────────────┘
               │                │
               ▼                ▼
        ┌─────────────┐  ┌─────────────┐
        │ ui/         │  │ live/       │
        │ static      │  │  drift +    │
        │ dashboard   │  │  alerting   │
        └─────────────┘  └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │ llm/        │
                       │  Ollama     │
                       └─────────────┘
```

## Deployment

### Local dev

```bash
uv sync
uv run pytest         # 394 tests
uv run ruff check     # clean
uv run pyright        # 0 errors
uv run kairon api     # boots FastAPI on :8000
```

### With ML / deep extras

```bash
uv sync --extra ml       # xgboost, lightgbm, torch
uv sync --extra api      # fastapi, uvicorn
uv sync --extra experiment  # mlflow, optuna
```

### With LLM (Ollama cloud)

```bash
export KAIRON_OLLAMA_HOST=https://ollama.com
export KAIRON_OLLAMA_MODEL=gpt-oss:120b-cloud
export OLLAMA_API_KEY=...  # Ollama reads this itself
```

The LLM client is a hard opt-in: if ``ollama`` is not installed the
client returns a graceful "unavailable" response with an error
string, never an exception.

## Configuration

Every setting is overridable via ``KAIRON_*`` env vars or a ``.env``
file. See ``KaironSettings`` in ``src/kairon/config/__init__.py`` for
the canonical list. Important ones:

| Setting | Default | Purpose |
|---|---|---|
| ``KAIRON_ARTIFACT_ROOT`` | ``./artifacts`` | Where ``ModelStore`` writes |
| ``KAIRON_LOG_LEVEL`` | ``INFO`` | DEBUG for traces |
| ``KAIRON_MAX_POSITION_EQUITY_FRACTION`` | ``0.20`` | Per-position cap |
| ``KAIRON_MAX_TOTAL_LEVERAGE`` | ``1.0`` | Total-leverage cap |
| ``KAIRON_DRIFT_METHOD`` | ``psi`` | ``psi`` or ``ks`` |
| ``KAIRON_API_PORT`` | ``8000`` | HTTP port |

## API surface

```
GET  /healthz                       liveness
GET  /v1/models                     list backends
POST /v1/models/train               fit a model
POST /v1/models/predict             inference
POST /v1/backtest                   cost-aware backtest
GET  /ui/                           static dashboard
```

The DTOs in ``src/kairon/api/dto.py`` are the contract. Every
response field is pydantic-typed; every input is pydantic-validated.

## Model store

A trained model is a directory:

```
{artifact_root}/{run_name}/
  meta.json    # backend, features, classes, metrics, timestamps
  state.pkl    # pickled fitted artifact
```

Load by ``run_name``:

```python
from kairon.store import ModelStore
store = ModelStore("./artifacts")
trained = store.load("r1")
```

## Live runtime

```python
from kairon.live import LivePredictor
predictor = LivePredictor(model, trained)
result = predictor.predict(features)
# latency_ms tracked, errors counted
```

Drift detection:

```python
from kairon.live.drift import check_drift_table
scores = check_drift_table(ref, live, feature_names)
```

Alerting:

```python
from kairon.live.alerts import AlertEngine, DriftSeverityRule, InMemoryChannel
eng = AlertEngine(rules=[DriftSeverityRule()], channels=[InMemoryChannel()])
eng.evaluate(score)  # score is a DriftScore
```

## Paper trading

```python
from kairon.paper import PaperTrader, Order, OrderSide
trader = PaperTrader()
trader.on_price("BTC/USDT", 50_000)
trader.submit_order(Order(symbol="BTC/USDT", side=OrderSide.BUY, size=0.1),
                    fill_price=50_000)
state = trader.snapshot()
```

The paper trader applies the same ``CostModel`` as the backtester, so
paper and back are directly comparable.

## Risk & portfolio

```python
from kairon.portfolio import (
    SizingConfig, size_position, ExposureLimits, check_exposure, aggregate_signals,
)
size = size_position(equity=10_000, price=100, config=SizingConfig(method="kelly", kelly_cap=0.1),
                     win_rate=0.6, avg_win=100, avg_loss=100)
```

## CI gates

Three commands must all pass:

```bash
uv run pytest        # 394 passed, 15 skipped (optional deps)
uv run ruff check    # All checks passed!
uv run pyright       # 0 errors, 222 warnings (informational)
```

The warnings from pyright are intentional — they live in
``pyrightconfig.json`` as demoted errors so we see the signal without
the false positives in test code.

## Failure modes & recovery

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: ollama` | Optional dep missing | `uv sync` (it's in core deps already) or skip LLM features |
| `ModelError: requires optional package 'xgboost'` | Tree-boost backend called without ML extras | `uv sync --extra ml` |
| `torch not installed` | LSTM/N-BEATS test skipped | Tests skip; runtime refuses to construct |
| Slow backtest | Cost model has high `impact_coefficient` | Set to 0 for static slippage |
| Drift alert storms | `confidence_floor` too tight | Raise to 0.5+ |
| Paper vs back mismatch | Different cost models | Pass the same ``CostModel`` to both |
| Equity curve NaN | Division by 0 in realised PnL | Check ``initial_equity`` and size |

## Observability

- **MLflow** — every fit/predict call logs params, metrics, and
  artifacts to ``{artifact_root}/mlruns/``. View with
  ``mlflow ui --backend-store-uri ./artifacts/mlruns``.
- **Events** — the paper trader records every order/fill/mark with
  a monotonic ``event_id`` in its in-memory log.
- **Alerts** — every alert has a ``created_at`` and a unique
  ``extras`` payload; pipe the ``InMemoryChannel`` to your log
  aggregator.

## Security

- API is bound to ``127.0.0.1`` by default. Override via
  ``KAIRON_API_HOST`` *and* put it behind a real reverse proxy
  (nginx, Caddy) for production.
- CORS origins are explicit; do not use ``*`` in production.
- The LLM prompt forbids buy/sell recommendations; the prompt and
  parser are unit-tested.
- The paper trader refuses to short by default; flip
  ``allow_short=True`` only after you understand the cost.

## Roadmap

- Production-grade adapter: pluggable broker interface (CCXT,
  IBKR, Alpaca).
- WebSocket data adapter: subscribes to OHLCV ticks and pushes to
  the predictor.
- Multi-symbol portfolio backtest with margin accounting.
- Model registry served over the API.
