# Literature Review — Kairon (AI Trading Research Platform)

**Date:** 2026-06-05
**Sources synthesized:** R-001 (Weinberg, S&P 500), R-002 (Csanadi & Lennartsson, crypto), plus external SOTA from 2024-2025 searches.

## 1. Method family taxonomy

| Family | Representative works | Verdict for Kairon |
|--------|----------------------|--------------------|
| Tree ensembles (RF, XGBoost, LGBM) | R-001, R-002, Jaquart 2022 | **Include.** Fast, robust, interpretable, probability-calibrated. |
| Linear/Logistic baselines | R-001 | **Include as anchor.** If you cannot beat LR, nothing works. |
| Recurrent (LSTM, GRU) | Fischer & Krauss 2018, McNally 2018, Lahmiri 2019 | **Include as one of the architectures.** Modest gain over trees. |
| Attention / Transformers | Li 2021, Informer, PatchTST, iTransformer (ICLR 2024) | **Include Decision Transformer and iTransformer.** Promising for cross-asset / long sequences. |
| Hybrid quantum-classical (VQC) | R-001 | **Defer.** PennyLane simulation cost-benefit is poor for live trading. |
| Conformal / confidence-thresholded RF | R-002 | **Include as the production inference gate.** Strong and cheap. |
| Reinforcement learning (PPO, DQN) | Not in local | **Defer to v2.** High variance, hard to evaluate. |
| FinBERT/FinGPT sentiment | Stanford 2024, AI4Finance | **Include for news text layer (crypto + high-impact stocks).** |
| Bayesian / GP | Jang & Lee 2017 | **Defer.** Diminishing returns. |

## 2. Pattern matrix — what actually helps, what doesn't

### Patterns that survive scrutiny
1. **Confidence-thresholded inference** (R-002) — trades coverage for accuracy in a controlled way. The product can surface this explicitly to the user.
2. **Architecture diversity in ensembles** (R-001) — combining different inductive biases (LSTM + Transformer + GBT + LR + RF) > scaling same-architecture models or scaling data sources.
3. **Rolling retraining** (R-002) — 7-day window / 5-hour forward test is a sensible cadence.
4. **VIX/volatility context** (R-001) — predicting **volatility** direction is more learnable than predicting **price** direction; treat VIX as a primary head.
5. **Quality filter ≥52%** (R-001) — adding weaker learners to a majority vote is *worse* than not including them. Top-K > All.

### Patterns that look good but are weak
1. **Quantum sentiment features** (R-001) — +0.82% on ensemble. Cost of simulation is not justified at scale; the value likely comes from a non-linear feature basis that an MLP could provide.
2. **Naive dataset diversity** (R-001) — same architecture on different assets underperforms; the market dynamics are too correlated.
3. **Higher-frequency is not always better** (R-002) — accuracy rises with horizon in some regimes; Nyquist consideration matters.
4. **Static hyperparameters** — paper R-001 picks 52% threshold once; should be walk-forward-validated.

### Patterns that are hype / fail to generalize
1. **Claimed 90% directional accuracy** is unachievable in a tradable, fee-aware, slippage-aware setting. (See R-001 V. F "expected return is negative" warning.)
2. **Single random k-fold on time series** — the 2024 Bhand & Joshi *Illusion of Alpha* paper shows random k-fold on a target nothing can predict reports R² ≈ 0.83-0.91.
3. **Datasets containing 100% crypto retail data** — selection bias and survivorship bias are baked in.

## 3. Reproducibility risk ranking

| Paper / pattern | Risk | Why |
|---|---|---|
| R-001 60.14% ensemble | **Medium-High** | Single 70/30 split, no walk-forward, "Top-7" selected from 35 |
| R-002 71% on 10 coins @ T=1.3 | **Medium** | Two RF models with limited diversity; threshold not cross-validated |
| Generic "we got 90%" claims | **Critical** | Almost always leakage, survivorship, or overfitting |

## 4. Implications for Kairon design

1. Treat **R-001's 60% on S&P 500 daily** as a *legitimate directional ceiling* for that target/horizon.
2. Treat **R-002's 60-70% on 5-min crypto with confidence filtering** as a *legitimate ceiling* for that target/horizon.
3. **90% accuracy is not a meaningful objective.** The replacement is "maximum risk-adjusted PnL after costs, with a calibrated probability."
4. The system must make the **accuracy-coverage trade-off** user-visible (a slider in the product).
5. The system must make **regime** user-visible (volatility bucket, trend strength, structural break flag).
6. The product must **always show evidence** behind any signal: feature drivers, top neighbors, regime context, sentiment.

## 5. Unresolved academic questions worth tracking

- Whether **iTransformer** (ICLR 2024) or **PatchTST** generalize from standard time-series benchmarks to finance.
- Whether **conformal prediction** (Johansson 2013) on top of an architecture-diverse ensemble gives stronger calibration than either alone.
- Whether **on-chain + sentiment** features meaningfully reduce error for crypto beyond what technicals alone achieve.
- Whether **Decision Transformer** as a sequence modeler (R-001 found 56.99% on VIX) is competitive with purpose-built volatility models.
