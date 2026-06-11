# Kairon

> Strictly-typed, cost-aware, walk-forward-validated AI market analysis and prediction platform.

Kairon is a research-grade ML system for short-interval trading in crypto and US equities. It is built around four invariants:

1. **No metric theatre.** Direction accuracy is reported *with coverage*; Sharpe is reported *with DSR and PBO*; backtests are cost-aware.
2. **No leakage.** Walk-forward + purging + embargo is the only allowed harness; random k-fold is forbidden by code.
3. **No LLM hallucinations of numbers.** The LLM (Ollama cloud) is a typed reasoning layer that never produces a number that drives a trade.
4. **Strict typing.** `pyright --strict` is the CI gate; `pydantic v2` guards every IO boundary.

## Status

This repository is in **Phase 0: Skeleton**. The full design is in `docs/` (19 design docs + 8 ADRs) and was produced via a deep-research workflow against the local `researches/` and `datasets/` folders. The implementation roadmap is in `docs/implementation_roadmap.md`.

## Quick start

```bash
# install
uv sync --extra dev

# tests
uv run pytest

# type check (strict)
uv run pyright

# lint + format
uv run ruff check .
uv run ruff format .
```

See `AGENTS.md` for the full build/test/lint recipe.

## Where to start reading

- `docs/executive_summary.md` — one-page overview
- `docs/architecture.md` — full technical architecture
- `docs/adr/0001-uv-pyright-strict-pydantic-v2.md` — foundation
- `docs/adr/0002-walkforward-purged-embargo-evaluation.md` — evaluation rigor
- `docs/objective_and_metrics.md` — what "good" looks like
- `docs/best_practice_blueprint.md` — what is in, what is out

## License

MIT.
