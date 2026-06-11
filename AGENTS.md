# AGENTS.md — recipes for working in this repository

## Toolchain
- Python 3.12 (managed by `uv`)
- `uv` (Astral) — install, sync, run, lock
- `pyright --strict` — static type checker (CI gate)
- `mypy --strict` — defense-in-depth type checker
- `ruff` — linter + formatter (replaces flake8, isort, black)
- `pytest` — test runner
- `hypothesis` — property-based tests
- `pre-commit` — local pre-commit hooks

## Common commands

```bash
# install (creates .venv, syncs lockfile)
uv sync

# install with optional groups
uv sync --extra dev --extra ml --extra backtest --extra api --extra experiment

# run python in the managed venv
uv run python --version

# run tests
uv run pytest

# run tests with coverage
uv run pytest --cov=kairon --cov-report=html

# run only leakage tests
uv run pytest -m leakage

# run a specific test
uv run pytest tests/test_smoke.py::test_imports

# lint + format check
uv run ruff check .
uv run ruff format --check .

# auto-fix lint
uv run ruff check . --fix
uv run ruff format .

# type check (strict)
uv run pyright

# type check (defense in depth)
uv run mypy src tests

# install pre-commit hooks
uv run pre-commit install

# run all pre-commit hooks on all files
uv run pre-commit run --all-files
```

## Repository conventions

1. **All public functions have type annotations.** `pyright --strict` is the gate.
2. **All IO boundaries use `pydantic v2` models** with `model_config = ConfigDict(frozen=True, extra="forbid", strict=True)`.
3. **No `Any`, no `cast`, no `# type: ignore`** without a written justification comment.
4. **All timestamps are UTC and timezone-aware.** `pydantic` enforces `tzinfo`.
5. **All randomness is seeded.** Use `kairon.utils.random.Seed` (added in Phase 4).
6. **No silent defaults.** Every config field has either a YAML value, an env value, or a required `Field(...)`.
7. **Reproducibility.** Every output has a `provenance` field with `config_hash`, `data_hash`, `model_version`, `seed`.
8. **Tests must be hermetic.** No live network calls. Mock or use `evals/datasets/`.
9. **Leakage tests are mandatory.** `tests/splits/` is non-negotiable.

## Adding a new dependency

```bash
# add to a specific group
uv add --optional ml xgboost
uv add --optional dev pytest-mock

# add a dev-only dependency
uv add --dev ruff

# then re-lock
uv lock
```

## Adding a new model

1. Add a config YAML in `configs/models/{name}.yaml`.
2. Implement the class in `src/kairon/models/{name}.py` inheriting `kairon.models.base.Model`.
3. Add unit tests in `tests/models/test_{name}.py` (incl. determinism test).
4. Add a baseline entry to `evals/baselines/{name}.json` after a run.
5. Update the architecture document if the model is in v1.

## Adding a new feature

1. Implement in `src/kairon/features/{category}/{name}.py`.
2. Add a unit test against a known reference (TA-Lib, manual calculation).
3. Register the feature in `kairon.features.registry`.
4. Add it to the appropriate `configs/features/{category}.yaml`.

## Running an LLM call

The LLM is wrapped at `kairon.llm.client.OllamaClient`. Never call the SDK directly.
See `docs/adr/0005-llm-as-reasoning-not-numeric-oracle.md` and `0008-no-llm-for-numeric-prediction-or-evaluation.md`.
