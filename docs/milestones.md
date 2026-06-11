# Milestones — Kairon

**Date:** 2026-06-05
**Format:** Numbered milestones with explicit pass criteria. Tied to the implementation roadmap.

## M0 — Skeleton (after Phase 0)
- `uv` project builds, tests pass, pyright is green on strict.
- Pre-commit hooks wired.
- CI green on an empty tree.

## M1 — Data is real (after Phase 1)
- We can fetch, normalize, QC, and store BTC-USDT 1m and SPY 1d in production-quality parquet.
- Diagnostics page renders a clean report.

## M2 — Features are real (after Phase 2)
- Every indicator in the user table has a unit-tested implementation.
- A `FeaturePipeline` produces a deterministic feature frame.

## M3 — Labels & splits are leak-free (after Phase 3)
- Direction / magnitude / volatility labels are typed and leakage-tested.
- Walk-forward + purging + embargo is the only allowed harness; CI enforces it.

## M4 — Baseline ensemble works (after Phase 4)
- Architecture-diverse ensemble of LR + RF + XGB + LSTM + GARCH produces calibrated probabilities.
- Beats LR on the calibration fold.

## M5 — Honest backtest works (after Phase 5)
- Cost-aware backtest with DSR, PBO, CAS, regime breakdown, ablation JSON.
- Random k-fold cannot beat walk-forward by a wide margin (Bhand & Joshi audit passes).

## M6 — API is live (after Phase 6)
- FastAPI exposes market, signal, backtest, alert, explain endpoints.
- OpenAPI committed and reviewed.

## M7 — UI ships (after Phase 7)
- Watchlist, asset detail, compare, backtest, alerts, settings all functional.
- Honest "no signal" path is the default.

## M8 — LLM layer is real (after Phase 8)
- Ollama cloud wired.
- All user-facing calls return cited explanations.
- "No-numeric" guardrail is tested.

## M9 — Live inference is real (after Phase 9)
- WebSocket-based live signals.
- Drift detection fires when expected.
- Alerts go out via in-app / email / webhook.

## M10 — Diego UX is real (after Phase 10)
- Reproducible runs.
- Run-vs-run diff.
- Exportable JSON.

## M11 — Deep TS is real (after Phase 11)
- PatchTST, iTransformer, N-HiTS, Decision Transformer in the ensemble.
- Each passes the same CI gates.

## M12 — Paper trading passes (after Phase 12)
- 30+ days of paper trading.
- Live CAS within 1.5 std of backtest.
- ECE drift < 0.03.

## M13 — Strong milestone (product)
- DSR ≥ 0.95
- PBO ≤ 0.10
- ECE ≤ 0.05
- Cost-aware Sharpe ≥ 1.0
- Max drawdown ≤ 15%
- Hit rate × avg-win / (avg-loss × commission mult) > 1.0
- Regime breakdown non-degrading

## M14 — Breakthrough milestone (product)
- Cost-aware Sharpe ≥ 1.5 sustained 12 months
- Drawdown < 12%
- DSR ≥ 0.97
- PBO ≤ 0.08
- Paper trade + canary live

## Pass criteria for every milestone

1. **CI green:** all tests, pyright strict, ruff clean, no `# type: ignore` added in this milestone.
2. **Benchmarks green:** `evals/runner.py` shows no regression vs `evals/baselines/`.
3. **Docs updated:** ADR added if a major decision was made.
4. **Demo:** a recorded walkthrough of the milestone on a real fixture.
5. **Sign-off:** at least one reviewer outside the author.
