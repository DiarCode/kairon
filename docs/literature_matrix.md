# Literature Matrix — Kairon

**Date:** 2026-06-05
**Format:** tabular, machine-readable extension in `artifacts/evidence_table.json`.

| ID | Method | Problem | Dataset | Horizon | Target | Features | Split | Reported metric | Reproducibility | Practical relevance | Verdict for Kairon |
|----|--------|---------|---------|---------|--------|----------|-------|-----------------|------------------|---------------------|--------------------|
| R-001 | LSTM + DT + XGBoost + RF + LR + 4-qubit VQC ensemble | S&P 500 daily direction | 7 instruments, 2020-23, 1006d | 1d | binary up/down | 27 (TA + quantum) | Single 70/30 temporal | Acc 60.14% (CI [56.84, 63.44]) | Medium-High (no public code) | Daily equity / macro overlay | Use as **ceiling reference** for daily equity direction |
| R-002 | 2× RF + conf-threshold | 5-min crypto direction | 10 coins, 2022-23, 210k bars/coin | 5h (60 bars) | binary up/down | 12 (TA + Lyapunov) | Rolling 7d train / 5h test | Acc 71.3% @ T=1.3, 77.4% @ T=1.5 | Medium (code referenced) | Intraday crypto signals | Use as **canonical confidence-thresholded pipeline** |
| Fischer & Krauss 2018 | LSTM | S&P 500 daily direction | 1992-2015 | 1d | binary | TA | Walk-forward | Acc 56% | High (code public) | Baseline LSTM | Use as **lower bound reference** |
| Krauss et al. 2017 | DNN ensemble | S&P 500 stat arb | 1992-2015 | 1d | returns | TA + fundamentals | Walk-forward | Sharpe > 1.0 (claimed) | High | Long-only equity stat arb | Use for **sizing and trade construction** |
| Sezer et al. 2020 | CNN on images | Turkish stocks | 2007-2017 | 1d | binary | Image of price chart | Random k-fold ⚠️ | Acc 57.3% | Medium | Visual-feature replication is brittle | **Skip** |
| Li et al. 2021 | Multi-head attention ensemble | Chinese A-shares | 2010-2020 | 1d | binary | OHLCV + TA | Walk-forward | Acc 58.1% | Medium | Asian market direction | Replicate with `iTransformer` |
| Zhou et al. 2021 (Informer) | Transformer | Commodity futures | 2015-2020 | mixed | regression | OHLCV | Standard | MSE improvement | High | Generic time series | Replicate with `PatchTST` / `iTransformer` |
| PatchTST (Nie et al. 2023) | Patched transformer | Long-horizon TS | 7 benchmarks | long | regression | univariate | Standard | 21% MSE reduction | High | Long-horizon forecasting | **Include** for daily+ horizons |
| iTransformer (Liu et al. ICLR 2024) | Inverted transformer | Multivariate TS | 20 benchmarks | varied | regression | multivariate | Standard | SOTA on most | High | Cross-asset, transaction load | **Include** for portfolio heads |
| N-BEATS / N-HiTS | Basis expansion | Long-horizon TS | M3/M4 | long | regression | univariate | Standard | SOTA on M3/M4 | High | Long-horizon baseline | **Include** as classical DL baseline |
| Jaquart 2022 | RF + GBC + LSTM + GRU ensemble | Crypto | multiple | 1d | direction | TA + sentiment | Walk-forward | Sharpe 3.23 (LSTM) / 3.12 (GRU) | Medium | Intraday crypto | Use for **trading rules** |
| Dudek 2023 | SVR + MLP + Ridge | Crypto volatility | multiple | 1d/1w | regression | lagged returns | Walk-forward | lower RMSE | High | Volatility modeling | Use for **vol head** |
| Bhand & Joshi 2026 ("Illusion of Alpha") | Quantifies leakage effects | synthetic & real | various | mixed | Sharpe | n/a | A/B on contamination | Sharpe 0.17 → 1.75 on random k-fold | High (replicable) | **Diagnostic only** | **Mandate** this audit for any model |
| Wang & Ruf 2022 ("Information Leakage in Backtesting") | Theoretical + empirical | synthetic | n/a | n/a | n/a | n/a | A/B | random splits overfit | High | **Methodological** | **Mandate** purged + walk-forward |
| Bailey & López de Prado 2014 (Deflated Sharpe) | Theoretical | n/a | n/a | n/a | n/a | n/a | n/a | DSR formula | High | **Methodological** | **Mandate** DSR in any headline result |
| Stanford 2024 FinBERT tutorial | FinBERT fine-tune | Financial PhraseBank | small | n/a | 3-class sentiment | financial text | 80/20 | F1 97.4% | High (data + model) | Sentiment signal | **Include** for news layer |
| FinGPT v3.3 (AI4Finance 2024) | Llama-2-13B + LoRA | FPB, FiQA-SA, TFNS, NWGI | medium | n/a | 3-class sentiment | financial text | standard | F1 0.882 on FPB | High (open weights) | News + social sentiment | **Include** as alternative / upgrade path |
| SentimentPulse (Eomaxl 2024) | FinBERT + Kafka + FastAPI | Bloomberg-like | streaming | n/a | sentiment | RSS + EDGAR | continuous | F1 92%, p50 38s | High | **Architecture reference** for streaming | **Reference**, not 1:1 copy |

## High-confidence conclusions (evidence score ≥ 3 independent sources)

1. **Random shuffling on time series is invalid.** Evidence: Bhand & Joshi 2026, Wang & Ruf 2022, López de Prado 2018, Zachary David 2019.
2. **Confidence thresholding is a legitimate accuracy-coverage trade-off** when applied via a held-out calibration set. Evidence: R-002, Johansson 2013 conformal.
3. **Architecture diversity beats dataset diversity in financial ensembles.** Evidence: R-001 explicit A/B; Ballings 2015 (review); Kuncheva 2003 (theoretical).
4. **Purging + embargo + walk-forward is the gold standard for backtesting ML on financial series.** Evidence: López de Prado 2018; Wang & Ruf 2022; Harding 2020.
5. **Sentiment (FinBERT/FinGPT) adds measurable signal to crypto and earnings-sensitive stocks** but is noisy on its own. Evidence: Stanford 2024, FinGPT 2024, Jaquart 2022, SentimentPulse 2024.
6. **Tree ensembles (XGBoost, LGBM, RF) are a strong default baseline** for tabular financial features. Evidence: R-001, R-002, Krauss 2017, Ballings 2015.

## Medium-confidence findings (1-2 sources, plausible but unconfirmed)

- iTransformer / PatchTST generalize to financial multivariate forecasting.
- Hybrid quantum-classical features provide meaningful gain (R-001 is the only source; result is +0.82% on ensemble).
- Lyapunov-exponent-based chaos features (R-002) help in 5-min crypto.

## Rejected claims

- **"90% directional accuracy"** — no source supports this in a fee-aware, leakage-free, tradable setting. Reject.
- **"Random k-fold works on time series"** — empirically false; reject.
- **"More features = better"** — no signal; literature shows feature selection / ablation matters.
