# ADR-0001: Use `uv`, `pyright --strict`, and `pydantic v2` as the foundation

**Status:** Accepted
**Date:** 2026-06-05
**Deciders:** Kairon core team

## Context
Kairon is a financial ML system that requires strict typing for safety, fast reproducible installs, and a typed IO boundary. Defaults must be unambiguous and CI-enforceable.

## Decision
- Package management: `uv` (Astral). Single binary, lockfile-driven, fast, drop-in for pip/poetry/pipx.
- Static type checking: `pyright --strict` in CI, with `mypy --strict` as a defense-in-depth check.
- Lint/format: `ruff`.
- Runtime validation: `pydantic v2` for all IO boundaries, configs, and DB rows.

## Consequences
- One tool (`uv`) for env + deps + lockfile + run.
- Strict typing catches more bugs at PR time.
- `pydantic v2` adds ~µs per object at runtime — acceptable.
- The `py.typed` marker is shipped; downstream consumers can rely on types.

## Alternatives considered
- Poetry: slower, separate toolchain.
- Conda: heavier; not needed for our stack.
- dataclasses only: no runtime validation.
- attrs: less ecosystem.
