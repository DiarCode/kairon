# ADR-0007: Strict static typing at every boundary

**Status:** Accepted
**Date:** 2026-06-05

## Context
A financial system that silently coerces a string to a number, or a UTC-naive datetime to a local one, can produce wrong decisions. We want the type system to be a guard.

## Decision
- `pyright --strict` in CI; PRs fail on any new warning.
- All public APIs have explicit return types.
- All IO boundaries use `pydantic v2` models with `model_config = ConfigDict(frozen=True, extra="forbid", strict=True)`.
- `datetime` fields are timezone-aware UTC by default; naive datetimes are rejected.
- Numerics: prefer `Decimal` for money, `float` for ML, `int` for counts; never mix.
- No `Any`, no `cast`, no `# type: ignore` without a written justification comment.

## Consequences
- Higher friction in PRs.
- Far fewer runtime surprises in production.
- The codebase is self-documenting through types.

## Alternatives considered
- mypy --strict only: slower; Pyright is the editor + CI standard in 2025.
- Relaxed typing in research code: rejected (it eventually becomes production).
