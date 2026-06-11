# Repository Structure — Kairon

**Date:** 2026-06-05
**See also:** `docs/architecture.md` for the full architecture; this file is the directory-level map with brief annotations.

## Top level

| Path | Purpose |
|------|---------|
| `pyproject.toml` | uv-managed project metadata, deps, tool configs (pyright, ruff, mypy) |
| `uv.lock` | lockfile, committed |
| `.python-version` | 3.12 |
| `.pre-commit-config.yaml` | ruff + pyright hooks |
| `README.md` | overview, quick start, links to docs |
| `AGENTS.md` | recipes: `uv run pytest`, `uv run ruff`, `uv run pyright`, `uv run kairon` |
| `.gitignore` | standard + data/, models/, .env, __pycache__ |
| `configs/` | typed YAML configs |
| `data/` | local-only raw + processed (gitignored) |
| `models/` | local-only trained weights + meta (gitignored) |
| `reports/` | generated backtest tearsheets |
| `artifacts/` | generated manifests, evidence tables |
| `evals/` | CI benchmarks and regression baselines |
| `docs/` | design docs, ADRs, IA, user flows |
| `src/kairon/` | the package |
| `tests/` | unit, property, leakage, regression, snapshot |

## `src/kairon/`

| Module | Owns |
|--------|------|
| `config/` | typed config loader, settings, env |
| `data/` | adapters (CCXT, Polygon, Tiingo, FRED, Glassnode, CryptoPanic, GDELT), IO, diagnostics, ingestion |
| `features/` | technical, cross-asset, regime, on-chain, sentiment, pipeline |
| `labels/` | direction, magnitude, volatility, triple-barrier |
| `splits/` | walk-forward, purged, embargo, CPCV |
| `models/` | LR, RF, XGB, LGBM, LSTM, Decision Transformer, PatchTST, iTransformer, N-HiTS, GARCH, ensemble |
| `calibration/` | isotonic, Platt, temperature |
| `policy/` | sizer, stops, rules |
| `backtest/` | engine (backtesting.py + vectorbt), cost model, execution, reports |
| `evaluation/` | metrics, DSR, PBO, calibration, regime breakdown, ablation |
| `llm/` | ollama client, prompts, grounding, guardrails |
| `research/` | LLM-driven planner (read-only), synthesizer |
| `api/` | fastapi, routes, schemas |
| `observability/` | logging, tracing, metrics |
| `experiment/` | mlflow + optuna |
| `cli/` | typer / click entry points |

## `tests/`

| Path | Tests |
|------|-------|
| `tests/data/` | adapters, IO, diagnostics, ingestion |
| `tests/features/` | each indicator vs a known reference (TA-Lib), pipeline determinism |
| `tests/labels/` | each label spec, leakage invariants |
| `tests/splits/` | **leakage tests**: no future in features, no train/test overlap, embargo respects serial correlation, walk-forward is monotonic |
| `tests/models/` | each model, save/load, determinism with seed |
| `tests/calibration/` | calibration improves Brier / ECE on synthetic |
| `tests/backtest/` | cost model math, execution, pnl reconciliation |
| `tests/evaluation/` | DSR formula, PBO simulation, calibration metrics |
| `tests/llm/` | prompt contracts, citation enforcement, no-numeric guardrail |
| `tests/api/` | endpoint contracts, validation, auth |

## `evals/`

| Path | Purpose |
|------|---------|
| `evals/baselines/` | JSON baselines per (model, asset, horizon, version) |
| `evals/datasets/` | small deterministic fixtures for fast CI |
| `evals/runner.py` | runs a model on a fixture, diffs vs baseline |
| `evals/llm_safety/` | adversarial prompts that must be rejected |
