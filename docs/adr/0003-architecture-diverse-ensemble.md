# ADR-0003: Architecture-diverse ensemble as the default modeling layer

**Status:** Accepted
**Date:** 2026-06-05

## Context
R-001 empirically demonstrated that architecture diversity dominates dataset diversity (60.14% vs 52.80%, p<0.05). Ballings 2015 and Kuncheva 2003 support the theoretical underpinning.

## Decision
- The default model is an **architecture-diverse ensemble**: LR, RF, XGBoost, LightGBM, LSTM, Decision Transformer, PatchTST, iTransformer, N-HiTS, GARCH.
- Aggregation: Top-K majority vote with confidence weighting; K is chosen per-asset-per-horizon via walk-forward.
- Quality floor of 52% on the calibration fold (R-001 finding); weak members are excluded.

## Consequences
- Higher engineering cost (10 models, not 1).
- Higher confidence in out-of-sample performance.
- Better robustness to regime change.
- More compute per training run (mitigated by shared feature store).

## Alternatives considered
- Single best model per asset: simpler but more brittle.
- Same-architecture deep ensemble: lower diversity, lower ensemble gain.
- Dataset-diverse ensemble: rejected by R-001 evidence.
