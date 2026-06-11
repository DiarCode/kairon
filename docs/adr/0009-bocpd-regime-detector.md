# ADR-0009: BOCPD (Adams & MacKay 2007) for online regime detection

**Status:** Accepted
**Date:** 2026-06-08

## Context

The W8 honest report (`reports/w8_honest_report.md`) and the four audit
panels (glm.md, kimi.md, qwen.md, perplexity.md) all converge on a
single load-bearing diagnosis of `src/kairon/features/regime.py:37`:
the current `RegimeModel` is a static, 4-component diagonal-covariance
GMM on `(adx, atr_z)`, fit **once per (symbol, timeframe)** and used
unchanged through the lifetime of the model. The 4 audit panels call
this a HIGH-severity structural flaw with the same recommendation:

> "Replace the static GMM with Bayesian Online Changepoint Detection
> (BOCPD) to detect regime shifts in real-time without retraining."
> (glm.md, Module A)
>
> "Replace the GMM regime classifier with ... BOCPD." (kimi.md, Module A)
>
> "Replace the static GMM with BOCPD" (qwen.md, Module A)

The W8 pipeline already documents forward-compat with the W9 BOCPD
detector (see `scripts/run_e2e.py:343-347` and the W8 honest report
§5, §11).

The W9 plan section §9.1 explicitly says "decide, don't ship both":
the choice is between BOCPD and an HMM (Hidden Markov Model), and we
must pick one and document why.

## Decision

Adopt **Bayesian Online Changepoint Detection (BOCPD)** (Adams & MacKay,
2007) as the W9 regime detector. The detector is implemented as a
plain Python class `BOCPDRegimeDetector` in
`src/kairon/features/regime.py`, additive on the existing
`RegimeModel`. It operates on **realized volatility + spread** (bar-level
data, no L2 required), uses a Normal-Inverse-Gamma conjugate
run-length prior, and emits a per-bar run-length posterior
`P(r_t = r | x_1:t)` plus a derived `Regime` label.

**No HMM in this phase.** The HMM was considered and rejected; see
"Alternatives considered" below.

## Why BOCPD over HMM

| Dimension | BOCPD (Adams & MacKay 2007) | HMM (Baum-Welch / forward-backward) |
|---|---|---|
| Training pipeline | None — closed-form conjugate update per bar | Baum-Welch EM on a labelled corpus, plus a state-count selection step (BIC / Viterbi-constrained search) |
| Latency budget per bar | O(1) amortised via the truncation threshold `S_MAX=200` (constant memory) | O(N_states^2) per bar (forward + backward lattice) |
| Detects novel regimes | Yes — the run-length posterior peaks at the most recent changepoint; the model makes no commitment to a fixed state alphabet | No — the model is committed to a fixed state alphabet at fit time; novel regimes are silently mis-assigned to the closest learned state |
| Calibrates to new symbols | No retraining — the conjugate prior learns the per-bar signal scale online | Requires re-fit on a labelled window |
| Fits 10-week budget | Yes — ~200 lines of plain Python, no external dependency beyond numpy | No — needs labelled regime data per symbol, plus a fit/select/validate pipeline that does not exist in the project today |
| Addresses all 4 audit panels' recommendation | Yes (all four recommend BOCPD by name) | No (none of the four recommend HMM) |

The decisive factors are:

1. **No training pipeline dependency.** BOCPD has no fit step — it
   updates its posterior one bar at a time. The HMM path would need
   labelled regime data per (symbol, timeframe), which we do not have
   in the project (we only have the static GMM labels today).

2. **Detects novel regimes.** A static GMM (and a fixed-alphabet HMM)
   cannot detect a regime the model has never seen. BOCPD's run-length
   posterior peaks at the most recent changepoint regardless of
   whether the new regime matches anything in the prior. This is the
   load-bearing property the four audit panels flagged.

3. **Fits the 10-week budget.** The HMM path is a 1-2 week project on
   its own (labelling, fit, validation, calibration). BOCPD is one
   short class with the conjugate update derived in the paper.

## Why not HMM

The HMM was considered and rejected for three reasons:

1. **No labels.** The project has no labelled regime corpus. Fitting an
   HMM on unsupervised data (EM) gives a state alphabet the model
   invents; that is not better than the static GMM we have today.
2. **Fixed state alphabet.** An HMM committed to K states cannot
   detect a K+1-th regime. BOCPD has no such commitment.
3. **Audit-panel consensus.** All four audit panels recommend BOCPD.
   None recommends HMM. Picking the method the panels did NOT
   recommend would re-open the same discussion at W8.5 decision-fork
   time.

## Plan alignment

- The W9 plan §9.1 says "decide, don't ship both". This ADR is the
  explicit decision.
- The W9 plan §9.2 says "BOCPDRegimeDetector on realized vol + spread
  (additive on existing RegimeModel); no L2 required". The
  implementation in `src/kairon/features/regime.py` follows this
  exactly.
- The W9 plan §9.3 (regime-conditional cost model) is the natural
  consumer of the BOCPD detector: a detected regime shift triggers a
  step in `regime_eta_multipliers` and the cost-aware backtester
  scales `eta` by the multiplier.

## Consequences

- `BOCPDRegimeDetector` ships as a plain Python class (not
  `torch.nn.Module`) so the `pyright --strict` gate stays clean and
  no torch dependency is added.
- The existing `RegimeModel` is **not removed**; the W9 path is
  additive. Consumers (backtest engine, evaluation harness, paper
  simulator) continue to use `RegimeModel` until a downstream story
  migrates to the BOCPD path.
- The `BOCPDRegimeDetector` operates on **realized volatility and
  spread** — no L2 data is required, which is the BTC-only-fallback
  constraint (W0).
- The detector's run-length posterior is exposed alongside the
  derived `Regime` label so the W9.3 cost-regime coupling can read
  the posterior mean (or the posterior entropy) as a soft regime
  signal, not just the hard argmax.

## Alternatives considered

- **HMM (rejected)** — see "Why not HMM" above.
- **CUSUM / Page-Hinkley** — considered; simpler than BOCPD but does
  not produce a per-bar run-length posterior and is not recommended
  by any of the 4 audit panels.
- **Hidden Semi-Markov Model (HSMM)** — explicit duration modelling
  is a future story; the v1 BOCPD truncates the run-length
  distribution at `S_MAX=200` to bound memory, which is a
  deliberately simpler proxy.
- **Online neural change-point detection (e.g. ChangePointGAN,
  BOCPD-Transformer)** — out of scope for the 10-week budget and
  not recommended by the audit panels.

## References

- Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online
  Changepoint Detection". arXiv:0710.3742.
- Kairon audit panels (2026-06-08): `enhance/glm.md`, `enhance/kimi.md`,
  `enhance/qwen.md`, `enhance/perplexity.md`.
- Kairon W8 honest report (2026-06-08): `reports/w8_honest_report.md`.
- Kairon W9 plan (2026-06-07): `.omc/plans/kairon-real-data-90-percent-refactor.md`
  §9.1–§9.3.
