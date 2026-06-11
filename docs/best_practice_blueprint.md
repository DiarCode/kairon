# Best-Practice Blueprint — Kairon

**Date:** 2026-06-05
**Purpose:** The strongest defensible end-to-end design, with what is included, what is excluded, and the evidence behind each call.

## 1. Top-level system overview

```
+-----------------------------------------------------------------+
|                      Kairon Decision Engine                      |
|                                                                 |
|  +----------+   +-------------+   +----------+   +-----------+  |
|  | Ingest   |-->|  Features   |-->|  Models  |-->|  Policy   |  |
|  +----------+   +-------------+   +----------+   +-----------+  |
|       |              |                 |                |        |
|       v              v                 v                v        |
|  +----------+   +-------------+   +----------+   +-----------+  |
|  | Data QC  |   |  Ablation   |   | Calibr.  |   | Explainer |  |
|  +----------+   +-------------+   +----------+   +-----------+  |
|                                                                 |
|             +------------+   +-----------------+               |
|             |  Backtest  |-->|  Cost / Slip /  |               |
|             |  Engine    |   |  PBO / DSR      |               |
|             +------------+   +-----------------+               |
+-----------------------------------------------------------------+
                              |
                              v
                  +-----------------------+
                  |  LLM Reasoning Layer  |  (Ollama cloud)
                  |  - Evidence-grounded  |
                  |  - Explanation only   |
                  +-----------------------+
```

## 2. What is INCLUDED (with evidence)

### 2.1 Architecture-diverse ensemble
- **Members:** Logistic Regression (anchor), Random Forest, XGBoost, LightGBM, LSTM, Decision Transformer, PatchTST, iTransformer, N-HiTS, GARCH.
- **Why:** R-001 showed architecture diversity > dataset diversity (60.14% vs 52.80%, p<0.05). Ballings 2015 supports diminishing returns beyond ~7 members.
- **Aggregation:** Top-K majority vote with confidence weighting (R-001 + R-002 hybrid). K chosen by walk-forward.
- **Quality floor:** 52% directional accuracy in calibration fold (R-001 finding); below that, exclude.

### 2.2 Confidence-thresholded inference
- **Mechanism:** Sum-of-probabilities threshold T (R-002), but extended with **conformal calibration** (Johansson 2013) for distribution-free coverage guarantees.
- **UI:** User can slide T to trade coverage for accuracy. The product explains the trade-off in plain English.
- **Why:** R-002 demonstrated a real, measurable trade-off (e.g., 71.3% @ 56% coverage → 88.5% @ 6.7% coverage).

### 2.3 Multi-horizon heads
- **Targets:** 5m, 15m, 1h, 4h, 1d, 1w — depending on asset.
- **Per-horizon model** (not one model for all horizons): research consensus is horizon-specific inductive biases.
- **Why:** PatchTST + iTransformer (ICLR 2024) both win on long horizons; LSTM/GRU better on short. No free lunch.

### 2.4 Direction + magnitude + volatility
- **Three heads**:
  1. Direction (binary or 3-class).
  2. Magnitude (regression: `log return at t+H`).
  3. Volatility (GARCH + N-HiTS + realized).
- **Why:** R-001 found VIX is more learnable than price; vol is its own head.

### 2.5 Regime classifier
- **Method:** HMM + rule-based + ML hybrid. Output: {Trending, Ranging, Volatile, Stressed}.
- **Why:** R-001 found ensemble accuracy varies materially by VIX bucket (54.5% in stressed vs 66.7% in high-vol). Regime is a known confounder.
- **Use:** Filter signals, adjust confidence, surface to user.

### 2.6 Cross-asset context encoder
- **iTransformer** over a portfolio's assets: attention across instruments, FFN over time.
- **Why:** ICLR 2024 paper shows iTransformer SOTA on multivariate; aligns with the regime + cross-asset intuition.

### 2.7 Sentiment / news encoder
- **FinBERT** (ProsusAI) for news; FinGPT v3.x for fine-tune option.
- **Inputs:** CryptoPanic, Tiingo News, GDELT.
- **Why:** Stanford 2024 shows F1 0.97 on FPB; FinGPT 2024 F1 0.88 on FPB. Sentiment is incremental, not primary.

### 2.8 On-chain encoder (crypto)
- **Glassnode** features: MVRV, NUPL, SOPR, exchange netflow, dormancy.
- **Why:** Documented to add macro-cycle context for crypto.

### 2.9 Feature selection / ablation engine
- **Methods:** SHAP, permutation importance, recursive feature elimination, Optuna HPO.
- **Output:** a *ranked* shortlist per asset+horizon.
- **Why:** Literature converges on feature importance ≠ feature count; ablations are non-negotiable.

### 2.10 Policy / trade construction
- **Inputs:** calibrated probabilities, regime, vol forecast, user risk budget.
- **Output:** position size, stop, take-profit, time-stop.
- **Why:** Jaquart 2022 demonstrated tradable Sharpe from ML + vol-aware sizing.

### 2.11 Confidence calibration
- **Methods:** Platt scaling, isotonic, temperature scaling on a held-out calibration fold.
- **Why:** Brier / ECE target ≤ 0.05 in the metrics doc.

### 2.12 Explanation layer
- **Mechanism:** SHAP for tabular, attention weights for transformers, top-k neighbor retrieval for RF.
- **Why:** R-001 explicitly flags interpretability as a regulator-grade requirement.

### 2.13 Execution simulator
- **Components:** realistic commission, bid/ask spread snapshot, market-impact model (square-root of `order_size / ADV`), funding (perps), borrow (shorts), latency.
- **Why:** R-001 V.F explicitly demonstrates that **raw accuracy does not guarantee profitability**; the toy example there is **negative expected return** even at 60% accuracy.

### 2.14 LLM reasoning layer
- **LLM:** Ollama cloud `gpt-oss:120b-cloud` (131k context, MXFP4, 1.8s p50).
- **Role:** evidence-grounded explanation, anomaly commentary, research synthesis, agent planning.
- **Forbidden:** any numeric prediction or numeric evaluation.
- **Why:** User requirement + 2025 production latency and tier are usable.

### 2.15 Evaluation: walk-forward + purging + embargo + DSR + PBO
- **Walk-forward** as the default backtest harness.
- **Purging + embargo** (López de Prado 2018) for all CV.
- **DSR** (Bailey & López de Prado 2014) for any Sharpe claim.
- **PBO via CPCV** (Bailey et al. 2017) as a standard diagnostic.
- **Leakage audit** per Bhand & Joshi 2026.

## 3. What is EXCLUDED (with evidence)

| Excluded | Why |
|----------|-----|
| Random k-fold on time series | Invalid; evidence: Bhand & Joshi 2026, Wang & Ruf 2022 |
| Hybrid quantum features | R-001 +0.82% on ensemble; classical MLP gets the same; cost not justified |
| Naive dataset-only ensembles | R-001: 52.80% < 60.14% |
| Sourcing a "90% accurate" model | Not defensible under valid evaluation |
| LLM as numeric oracle | Will silently hallucinate numerics; forbidden by design |
| Future-feature leakage in labels | Strict tests; pre-commit CI checks |
| Order types not exposed by CCXT | Out of scope for v1 |
| Heavy LLMs (BloombergGPT-scale) | Cost; FinBERT/FinGPT cover sentiment adequately |
| Full HFT stack | Out of scope; we will not compete on µs latency |

## 4. Expected generalizable gains (with caveats)

| Component | Expected contribution | Evidence |
|-----------|----------------------|----------|
| Architecture-diverse ensemble | +2-3% over best single model | R-001 |
| Confidence thresholding (T) | +5-10% accuracy at lower coverage | R-002 |
| Sentiment (FinBERT) | +0.5-1% on earnings/news days | Stanford 2024, Jaquart 2022 |
| On-chain (Glassnode) | crypto-specific macro edge, not directly measurable on accuracy | Glassnode docs |
| Walk-forward + DSR | makes results **honest**, not higher | Methodology |
| Calibration | makes downstream decisions better, not accuracy higher | Brier/ECE |

## 5. Ablation plan

Every shipped model ships with a typed ablation record (JSON, mlflow-tagged):

- without architecture diversity
- without confidence threshold
- without sentiment
- without on-chain
- without regime filter
- without calibration
- without embargo
- without cost model
- with random k-fold (sanity; should be flat or worse)

The CI gates any new model on: (a) it doesn't regress on the calibration fold, (b) PBO doesn't increase, (c) DSR stays ≥ 0.95.

## 6. Risk register

| Risk | Mitigation |
|------|------------|
| Concept drift | Daily retraining cadence (R-002 style 7d window) |
| Regime change | HMM regime classifier + regime-specific model selection |
| Liquidity shock | Order book depth pre-trade gate |
| Exchange downtime | Multi-venue adapter (CCXT) + paper-trade fallback |
| LLM hallucination | Strict prompt contracts; every output must cite inputs |
| Backtest overfit | DSR, PBO, walk-forward, embargo, no in-sample tuning |
| Survivorship | Use TWRR (delisted included) + universe rebalance per fold |
| Latency (live) | Asynchronous CCXT + WebSocket; pre-compute features |

## 7. What "good" looks like

A Kairon signal card in the UI shows:
- `BTC-USDT`, 1h horizon, 62% confidence (calibrated)
- Regime: Volatile, Trend strength: 0.7
- Top drivers: 24h volume z-score, MVRV < 1.0, news sentiment (3 sources)
- Risk: 1.2% stop, 3.1% target, position size 2% of equity
- Backtest on this exact config: 58% hit rate, 1.1 cost-aware Sharpe, 9% max drawdown, DSR 0.97, PBO 0.08
- Honest caveat: "This signal is not actionable in 38% of bars; click here to see coverage and confidence trade-off."
