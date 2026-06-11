# ADR-0002: Walk-forward + purging + embargo as the only valid backtest harness

**Status:** Accepted
**Date:** 2026-06-05

## Context
Standard ML evaluation is invalid for time series. Random k-fold breaks temporal dependencies; single 70/30 splits overfit to one regime; backtest overfitting is rampant in our domain (Bailey & López de Prado 2014; Bhand & Joshi 2026).

## Decision
- The only allowed backtest harness is **walk-forward with purging and embargo** (López de Prado 2018).
- All claims of "model skill" must come from walk-forward + DSR + PBO.
- `tests/splits/` enforces this with explicit leakage tests.
- CI gates any new model on PBO ≤ 0.10 and DSR ≥ 0.95 on the canary fold.

## Consequences
- Slower experimentation (cannot reuse a model across folds).
- Forces discipline in label construction (must be leakage-free).
- Honest results.

## Alternatives considered
- Random k-fold: rejected (proven invalid by Bhand & Joshi 2026).
- Single temporal split: rejected (selection bias).
- Expanding window walk-forward without purging: rejected (label overlap).
