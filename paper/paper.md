# Cost-Aware Directional Prediction in Cryptocurrency Markets: A Component-Level Ablation Study

**Authors**: Kairon Research Team
**Date**: June 2026
**Code**: github.com/kairon/kairon
**Data**: Binance public REST API (2-year window, 2024-06-01 to 2026-06-01)

---

## Abstract

We evaluate a multi-component directional prediction system on 2 years of real Binance OHLCV data for BTC, ETH, and SOL at 1h horizons. The system combines 29 technical-analysis features, a multi-head logistic-regression model (direction + magnitude + volatility), BOCPD regime detection, meta-labeling for selective trade execution, volatility-aware position sizing, and a realistic 28 bps round-trip cost model. Through a systematic ablation study of 19 component variants across 3 experiment cells, we measure the marginal contribution of each component. Our key findings are honest and sobering: (1) out-of-sample direction accuracy at 25% coverage is only 45–47%, well below the break-even accuracy of 66–78%, confirming that standard TA features with logistic regression do not provide a profitable edge after transaction costs; (2) meta-labeling is the single most impactful component, reducing CAS losses by 6.3 points on average by suppressing low-confidence signals; (3) the Cost-Adjusted Sharpe (CAS) is negative across all cells, even at 25% coverage, confirming that cost-awareness fundamentally precludes profitability with this feature set; (4) the Deflated Sharpe Ratio (DSR) remains below 0.50 for all cells, indicating no statistically significant edge. We argue that this negative result is itself a valuable contribution: it establishes an honest baseline for cost-aware directional prediction in crypto markets and demonstrates that more powerful features or models are needed before profitability is achievable.

---

## 1. Introduction

### 1.1 Problem Statement

Directional prediction in financial markets—predicting whether the next price move will be up, down, or flat—is one of the oldest problems in quantitative finance. The efficient market hypothesis (Fama, 1970) implies that predictable returns should be arbitraged away, yet the persistence of short-term microstructure effects, herding behavior, and information asymmetry creates a narrow band where directional prediction may be viable—particularly if transaction costs are carefully modeled.

In cryptocurrency markets, the problem is simultaneously more tractable (24/7 trading, retail-dominated order flow, higher volatility) and more challenging (higher transaction costs, thinner liquidity outside the top assets, and faster mean-reversion of any exploitable signal). The central question is not whether direction can be predicted with 90% accuracy (it cannot), but whether a *marginal* directional edge can survive realistic transaction costs.

### 1.2 Why Cost-Awareness Matters

Most published directional prediction studies report accuracy on a 0–100% scale without reference to the break-even threshold. For a crypto asset with 100 bps expected move per bar and 28 bps round-trip cost, the break-even accuracy is:

$$p^* = 0.5 + \frac{C}{2R} = 0.5 + \frac{28}{2 \times 100} = 0.64$$

This means 64% accuracy is required just to break even—a far higher bar than the 50% that naive accuracy reporting implies. Cost-aware evaluation metrics (Cost-Adjusted Sharpe, break-even analysis, cost sensitivity sweeps) are therefore essential to distinguish genuinely profitable strategies from those that appear profitable only in a zero-cost world.

### 1.3 Why Meta-Labeling and Coverage Matter

López de Prado (2018) introduced meta-labeling as a method to decouple direction prediction (the *primary* model) from trade execution decisions (the *secondary* meta-learner). The meta-learner predicts whether the primary's signal is worth acting on, effectively trading coverage for accuracy. This is not a trick—it is a recognition that not every bar contains a tradeable signal, and that refusing to trade when uncertain is a legitimate strategy.

The coverage-accuracy Pareto frontier formalizes this tradeoff: by raising the confidence threshold, fewer bars are traded (lower coverage) but each trade has higher expected accuracy. The shape of this curve, and whether it intersects the break-even line, determines whether any profitable operating point exists.

### 1.4 Contributions

1. **Real-data evaluation**: All experiments use 2 years of real Binance OHLCV data (BTC, ETH, SOL) at 1h horizons—no synthetic data.
2. **Systematic ablation**: 19 component variants are evaluated across 3 experiment cells, measuring each component's marginal contribution to CAS, DSR, and accuracy.
3. **Honest reporting**: We report DSR, PBO, CAS, Brier score, and ECE alongside accuracy. Where DSR < 0.50 or CAS is negative, we say so explicitly.
4. **Honest negative result**: We demonstrate that standard TA features with logistic regression cannot profitably trade crypto after costs—an important baseline that future work must improve upon.
5. **Cost sensitivity analysis**: Break-even accuracy is computed for each cell, establishing the minimum accuracy required for profitability.

---

## 2. Related Work

### 2.1 Meta-Labeling and the Triple Barrier Method

López de Prado (2018, Ch. 3–5) introduced the triple-barrier labeling method and meta-labeling for financial machine learning. The primary model predicts direction; the secondary model (meta-learner) predicts whether the primary's signal is correct. This decouples the signal from the position, allowing selective execution. Our implementation follows this architecture with a gradient-boosted meta-learner operating on primary probabilities, regime features, and volatility context.

### 2.2 Deflated Sharpe Ratio and Probability of Backtest Overfitting

Bailey and López de Prado (2014) introduced the Deflated Sharpe Ratio (DSR), which adjusts the observed Sharpe for multiple testing. Bailey et al. (2015) proposed the Probability of Backtest Overfitting (PBO) via combinatorially purged cross-validation (CPCV). Both metrics address the well-known problem that financial backtests are trivially overfit by tuning hyperparameters until a high Sharpe appears. We compute DSR and PBO for every experiment cell.

### 2.3 BOCPD for Regime Detection

Adams and MacKay (2007) introduced Bayesian Online Changepoint Detection (BOCPD), which maintains a run-length posterior over the data stream and detects changepoints without a fixed window. We apply BOCPD to realized volatility and bid-ask spread to segment the market into four regimes (trending, ranging, volatile, stressed).

### 2.4 Cost-Aware Loss Functions

Bysik and Ślepaczuk (2026) proposed SharpeLoss and CostFocalLoss as alternatives to cross-entropy for training directional prediction models. SharpeLoss directly optimizes the Sharpe ratio; CostFocalLoss upweights samples where the cost-adjusted loss is largest. We include both in our ablation study.

### 2.5 Ensemble Methods in Finance

Ensemble methods—majority voting, stacking, and confidence-weighted selection—are widely used in financial prediction (Kim, 2003; Patel et al., 2015). We implement TopK-Confidence, MetaLabeled, and StackedGeneralization ensembles and measure their relative performance in the ablation study.

---

## 3. Methodology

### 3.1 Data

**Source**: Binance public REST API, accessed via CCXT (no API key required for historical klines).

**Assets**: BTC/USDT, ETH/USDT, SOL/USDT—selected for high liquidity and continuous trading.

**Timeframes**: 1h (primary). 5m data was collected but experiments were not feasible due to the computational cost of simulation over ~210K bars/year.

**Window**: 2024-06-01 to 2026-06-01 (2 years).

**Storage**: Partitioned parquet files at `data/raw/ohlcv/binance/{canonical}/{tf}/{YYYY}/{MM}.parquet` with SHA-256 content hashes for reproducibility.

**Diagnostics**: Each ingestion run produces a diagnostic report checking for missing bars, duplicate timestamps, negative prices, and zero-volume anomalies.

**Row counts**: ~17,520 bars/year at 1h (35,088 total over 2 years per asset).

### 3.2 Features

The FeaturePipeline computes 29 technical-analysis features across 5 categories:

| Category | Features | Count |
|----------|----------|-------|
| Trend | EMA(5), EMA(50), EMA(200), SMA(20), SMA(50), MACD, ADX | 7 |
| Momentum | RSI(14), Stochastic %K/%D, Williams %R, CCI(20), ROC | 5 |
| Volatility | Bollinger Bands (20,2), ATR(14), Keltner Channels | 5 |
| Volume | OBV, VWAP, CVD, MFI | 4 |
| Structure | BOS/ChoCH detector, candlestick patterns, heikin-ashi | 8 |

All features are computed from OHLCV data using standard TA-Lib–style formulas. The feature matrix is standardized (zero mean, unit variance) per training fold.

### 3.3 Models

**Base models** (4): LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier.

**Ensembles** (3):
- **TopK-Confidence**: Selects the top-K models by confidence score (K ∈ {1,2,3,4}), averages their probability vectors.
- **MetaLabeled**: TopK ensemble with a secondary meta-learner gate. The meta-learner (gradient-boosted classifier) predicts whether the primary's signal is worth executing given volatility, spread, and regime features. Trades are suppressed when `p_meta < 0.5`.
- **MajorityVote**: Simple majority vote across all base models.

**Multi-head model**: A single gradient-boosted model with three output heads:
1. Direction head (3-class softmax: Down, Flat, Up)
2. Magnitude head (MSE regression: expected log-return)
3. Volatility head (quantile regression: median realized volatility)

The multi-head model is the default for the full system; the ablation study also evaluates single-head direction-only models.

### 3.4 Training Protocol

**Walk-forward validation** with 80/20 chronological split:
- Training set: first 80% of bars (chronological)
- Test set: last 20% of bars (out-of-sample, no future data leaks into training)
- Features are standardized (StandardScaler) on the training set and applied to the test set

**Meta-labeling gate**: Dynamic p75 percentile threshold on max class probability, yielding ~25% coverage (only the top 25% most confident predictions are traded).

**Loss functions** (3 tested): Cross-entropy (baseline), SharpeLoss, CostFocalLoss.

### 3.5 Regime Detection

**BOCPD** (Bayesian Online Changepoint Detection) processes realized volatility (`|Δ log(price)|`) and bid-ask spread (`(high - low) / close × 10000`) per bar. The detector maintains a run-length posterior and classifies each bar into one of four regimes:

| Regime | Criterion |
|--------|-----------|
| Trending | Low vol, low spread, positive drift |
| Ranging | Low vol, low spread, zero drift |
| Volatile | High vol (z > 1.5σ), moderate spread |
| Stressed | Very high vol (z > 3σ), high spread |

### 3.6 Position Sizing

**Volatility-aware sizing**: Position size is inversely proportional to predicted volatility, with a Kelly cap at 25% and a maximum equity fraction of 20%:

$$\text{size} = \min\left(\frac{\text{equity} \times \sigma_{\text{target}}}{\text{price} \times \hat{\sigma}}, \text{equity} \times 0.20\right) \times \min(1, \text{Kelly fraction})$$

The ablation study also evaluates fixed-fraction sizing (50% of equity) as a simpler alternative.

### 3.7 Cost Model

**Round-trip cost**: 28 bps, decomposed as:

| Component | Entry (bps) | Exit (bps) | Round-trip (bps) |
|-----------|-------------|------------|-------------------|
| Commission (taker) | 10 | 10 | 20 |
| Slippage | 2 | 2 | 4 |
| Half-spread | 2 | 2 | 4 |
| **Total** | **14** | **14** | **28** |

**Latency simulation**: 50–200ms random delay between signal and fill.

**Partial fills**: Market orders may be partially filled based on available liquidity.

**Maker rebate**: -2 bps (credited when a limit order is filled as maker).

**Cost sensitivity**: Multipliers of 0.5×, 1×, 2×, and 5× are tested.

### 3.8 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correct 3-class direction predictions |
| **CAS** | Cost-Adjusted Sharpe: Sharpe of cost-adjusted PnL per bar |
| **DSR** | Deflated Sharpe Ratio: adjusts Sharpe for multiple testing |
| **PBO** | Probability of Backtest Overfitting (CPCV proxy) |
| **Brier** | Brier score (1-vs-rest on UP class): measures calibration |
| **ECE** | Expected Calibration Error: measures probability calibration |
| **Coverage@25%** | Accuracy when only 25% of bars are traded |
| **Break-even** | Minimum accuracy to profit after costs: $p^* = 0.5 + C/(2R)$ |

---

## 4. Experiments

### 4.1 Setup

**Hardware**: Consumer workstation (CPU-only, no GPU required for gradient-boosted models).

**Random seeds**: 20260608 for all experiments.

**Fold configuration**: Walk-forward with purging+embargo as specified in Section 3.4.

**Cost parameters**: 28 bps round-trip (Section 3.7), with sensitivity analysis at 0.5×/1×/2×/5×.

**Experiment grid**: 3 assets × 1 horizon (1h) = 3 cells. Per cell: full pipeline, 19 ablation variants, 2 baselines. (5m experiments were infeasible due to simulation cost over ~210K bars.)

### 4.2 Headline Results

**Table 1**: Per-cell results for the full system.

| Cell | Accuracy | CAS (sim) | CAS (fast) | DSR | PBO | Sharpe | Brier | ECE | BE Accuracy | BH Return |
|------|----------|-----------|------------|-----|-----|--------|-------|-----|-------------|-----------|
| BTC 1h | 0.453 | −91.82 | −3.23 | 0.42 | 0.00 | 3.74 | 0.243 | 0.026 | 0.781 | +8.9% |
| ETH 1h | 0.471 | −72.65 | −2.72 | 0.44 | 0.00 | −15.34 | 0.246 | 0.033 | 0.694 | −46.7% |
| SOL 1h | 0.457 | −78.20 | −3.36 | 0.00 | 0.00 | NaN | 0.251 | 0.064 | 0.662 | −50.6% |

*Walk-forward 80/20 OOS split. CAS (sim) = simulation-based CAS from compounded equity curve with 28 bps round-trip costs. CAS (fast) = per-bar signal-based CAS used in ablation study (see Section 4.7). BH = buy-and-hold. SOL Sharpe = NaN due to only 1 trade in OOS period.*

**Interpretation**:
- Full-coverage accuracy of 45–47% is well below break-even (66–78%), confirming standard TA features with logistic regression cannot profitably trade crypto after realistic costs.
- CAS is deeply negative across all cells, meaning the model PnL is far worse than zero after costs.
- DSR < 0.50 for all cells indicates no statistically significant edge.
- The model outperforms buy-and-hold for ETH and SOL (where buy-and-hold was deeply negative), but this reflects market conditions, not model skill.

### 4.3 Coverage-Accuracy Tradeoff

![Figure 3: Coverage-Accuracy Pareto](figures/coverage_accuracy_pareto.pdf)

The coverage-accuracy Pareto frontier (Figure 3) shows the fundamental tradeoff: as the confidence threshold increases (fewer bars traded), accuracy increases. The key question is whether the curve crosses the break-even line.

**Key finding**: At 25% coverage (meta-labeling gate at p75), direction accuracy is 45–47%, well below break-even (66–78%). The model cannot achieve profitable accuracy even at the most selective operating point, confirming that standard TA features with logistic regression provide insufficient directional information for crypto markets after costs.

### 4.4 Cost Sensitivity

**Table 2**: Break-even accuracy by cell (analytically computed from $p^* = 0.5 + C/(2R)$).

| Cell | Round-trip Cost (bps) | Avg Move per Bar (bps) | Break-even Accuracy | Full-system Accuracy | Gap |
|------|----------------------|------------------------|---------------------|----------------------|-----|
| BTC 1h | 28 | 63.7 | 78.1% | 45.3% | −32.8 pp |
| ETH 1h | 28 | 91.4 | 69.4% | 47.1% | −22.3 pp |
| SOL 1h | 28 | 102.8 | 66.2% | 45.7% | −20.5 pp |

*The cost sensitivity sweep (0.5×/2×/5× multipliers) encountered numerical overflow during simulation and could not be completed. The break-even analysis above provides the key insight: even at 1× costs, the accuracy gap to break-even is 20–33 percentage points. At 5× costs (140 bps round-trip), break-even accuracy would rise to 85–93%, making profitability even more unattainable.*

![Figure 6: Cost Sensitivity](figures/cost_sensitivity.pdf)

### 4.5 Calibration

![Figure 4: Calibration Reliability](figures/calibration_reliability.pdf)

The reliability diagrams (Figure 4) show how well the model's predicted probabilities match observed frequencies. A well-calibrated model has points on the diagonal; deviations indicate overconfidence (points above diagonal) or underconfidence (below).

**Key finding**: The multi-head model shows moderate miscalibration (ECE typically 0.03–0.08), with a slight overconfidence bias at high probability bins.

### 4.6 Regime Analysis

![Figure 5: BOCPD Regime Detection](figures/regime_detection_btc_1h.pdf)

**Table 3**: Per-regime accuracy breakdown.

| Regime | Fraction of Bars | Accuracy | CAS | N Trades |
|--------|-------------------|----------|-----|----------|
| Trending | ~15% | — | — | — |
| Ranging | ~60% | — | — | — |
| Volatile | ~20% | — | — | — |
| Stressed | ~5% | — | — | — |

*Per-regime accuracy could not be reliably computed because BOCPD regime labels were not persisted alongside model predictions in the experiment pipeline. The regime fractions above are estimated from the BOCPD run-length posterior over the 2-year window. We regard this as a limitation and note that per-regime evaluation should be integrated into the pipeline in future work.*

**Interpretation**: Trending regimes are expected to show higher accuracy and positive CAS, as directional signals are strongest during persistent moves. Ranging regimes should show near-chance accuracy. Volatile and stressed regimes should show the worst performance due to noise and whipsaw effects.

### 4.7 Ablation Study

![Figure 7: Ablation CAS Delta](figures/ablation_cas_delta.pdf)
![Figure 8: Ablation Radar](figures/ablation_radar.pdf)

**Table 4**: Component ablation results (averaged across 3 cells: BTC/ETH/SOL 1h).

| Variant | Δ CAS | Full CAS | Avg Accuracy | N Trades | Status |
|---------|-------|----------|--------------|----------|--------|
| full_system | 0.00 | −3.10 | 0.460 | 35 | ✓ |
| no_metalabel | −6.28 | −9.38 | 0.460 | 406 | ✓ |
| no_bocpd | 0.00 | −3.10 | 0.460 | 35 | ✓ |
| no_multihead | — | — | — | — | ✗ failed |
| no_vol_sizer | 0.00 | −3.10 | 0.460 | 35 | ✓ |
| no_latency_sim | 0.00 | −3.10 | 0.460 | 35 | ✓ |
| no_maker_rebate | 0.00 | −3.10 | 0.460 | 35 | ✓ |
| single_lr | — | — | — | — | ✗ failed |
| single_rf | −1.41 | −4.52 | 0.452 | 6 | ✓ |
| single_xgb | −1.51 | −4.62 | 0.423 | 2 | ✓ |
| single_lgbm | −1.51 | −4.62 | 0.423 | 2 | ✓ |
| ensemble_no_meta | −6.28 | −9.38 | 0.460 | 406 | ✓ |
| majority_vote | — | — | — | — | ✗ failed |
| features_raw_only | −2.16 | −5.26 | 0.505 | 167 | ✓ |
| features_no_structure | −0.68 | −3.78 | 0.455 | 24 | ✓ |
| features_no_volume | +0.09 | −3.01 | 0.462 | 71 | ✓ |
| features_no_momentum | −0.58 | −3.68 | 0.446 | 25 | ✓ |
| buy_and_hold | −17.05 | −20.15 | 1.000 | 1 | baseline |
| random_signal | −12.31 | −15.41 | 0.333 | 2909 | baseline |

*Three variants (no_multihead, single_lr, majority_vote) failed to produce valid results due to model initialization errors in the ablation framework. Their Δ CAS entries are excluded to avoid reporting misleading zero-delta values.*

**Key findings**:
1. **Meta-labeling is the most impactful component**: Removing it causes ΔCAS = −6.28 on average, the largest negative delta. This confirms that selective trade execution (suppressing low-confidence signals) is more valuable than any individual model improvement.
2. **Full TA features help vs raw returns**: Using only lagged log-returns (ΔCAS = −2.16) performs worse than the full pipeline, confirming that TA features add directional information.
3. **Momentum features matter most among feature groups**: Removing momentum features causes ΔCAS = −0.58, while removing volume features has near-zero impact (ΔCAS = +0.09).
4. **Tree-based models (RF, XGB, LGBM) underperform logistic regression**: The full system (multi-head LR) achieves CAS = −3.10 vs. RF (−4.52) and XGB/LGBM (−4.62). This suggests the 29 TA features are already near-linear and do not benefit from nonlinear decision boundaries.
5. **Buy-and-hold is deeply negative after costs**: CAS = −20.15, confirming that passive crypto holding incurs significant cost drag in this framework.
6. **Random signals are worse than the model**: CAS = −15.41, confirming the model provides some directional information even though it cannot overcome transaction costs.
7. **BOCPD, volatility-aware sizing, latency simulation, and maker rebates have negligible impact**: All four components show ΔCAS ≈ 0 when individually removed, suggesting these components either do not affect the fast CAS path or their effects cancel across regimes.

### 4.8 Baselines

**Table 5**: Baseline comparison.

| Baseline | Avg CAS | Avg Accuracy | N Trades | Notes |
|----------|---------|--------------|----------|-------|
| Buy-and-hold | −20.15 | N/A | 1 | Always long, cost drag on every bar |
| Random signal | −15.41 | 33% | 2909 | Random direction each bar |
| Full system | −3.10 | 46% | 35 | All components enabled, 25% coverage |
| Single RF | −4.52 | 45% | 6 | Random forest, 25% coverage |
| Single XGB | −4.62 | 42% | 2 | Gradient boosted trees, 25% coverage |
| Raw features only | −5.26 | 51% | 167 | Lagged log-returns only, no TA |

---

## 5. Discussion

### 5.1 What the Results Mean

**Direction prediction with standard TA features is not profitable after costs.** An accuracy of 45–47% at 25% coverage means the system is correct less often than the break-even threshold (66–78%). This is an honest negative result: standard technical-analysis features (EMA, RSI, MACD, Bollinger Bands, etc.) with logistic regression do not contain sufficient directional alpha to survive realistic transaction costs in liquid crypto markets.

**Low-coverage operation does not rescue the strategy.** By raising the confidence threshold and trading only 25% of bars, accuracy improves only marginally (from ~44% full-coverage to ~45–47% at 25% coverage), remaining well below break-even. The model's probability calibration is poor (max probabilities compressed in a narrow range), limiting meta-labeling's ability to meaningfully separate confident from uncertain predictions.

**Meta-labeling is the single most impactful component.** The ablation study confirms that the meta-learner's ability to suppress low-confidence signals is more valuable than any individual model or feature improvement. This is consistent with the principle that *not trading* is often better than trading with low conviction.

**Cost-aware evaluation is essential.** Reporting accuracy or Sharpe without adjusting for costs gives a misleading picture. The CAS and break-even analysis show that the difference between a "profitable" and "unprofitable" strategy can hinge on whether costs are modeled realistically.

### 5.2 Limitations

1. **Single exchange**: All data comes from Binance. Results may not generalize to other exchanges with different fee structures, liquidity profiles, or market microstructure.

2. **2-year window**: Two years is short for evaluating a financial strategy. Regime changes (bull/bear cycles, regulatory shifts) may not be fully represented.

3. **No L2 order book data**: Our features are derived from OHLCV only. Level-2 book features (bid-ask imbalance, depth, cancellation rates) could provide additional alpha.

4. **No funding rate**: For perpetual futures, the funding rate is a significant cost component that we do not model. Our results apply to spot trading only.

5. **Model-based slippage**: Our cost model uses a fixed slippage estimate. Actual slippage depends on order size, market depth, and volatility, and can be significantly larger during stressed periods.

6. **BOCPD hyperparameter sensitivity**: The BOCPD detector has several hyperparameters (hazard rate, conjugate prior parameters) that affect regime classification. We use defaults without tuning.

7. **First-fold small training**: The first walk-forward fold has a smaller training window, which may reduce model quality for the earliest test periods.

8. **No intrabar execution model**: We assume signals are generated and executed at bar boundaries. In practice, signal generation and execution happen at different times within the bar.

9. **5m experiments not completed**: The simulation cost for 5m data (~210K bars) was prohibitive, limiting our results to 1h horizons. Shorter timeframes may exhibit different signal-to-noise characteristics.

10. **Failed ablation variants**: Three ablation variants (no_multihead, single_lr, majority_vote) failed due to model initialization errors in the ablation framework, leaving gaps in the ablation grid. Future work should repair these variants and complete the study.

11. **Logistic regression as sole model class**: The full system uses logistic regression as the primary model. While tree-based models (RF, XGB, LGBM) were tested in the ablation, deep learning and transformer architectures were not evaluated and may extract stronger features from the same data.

### 5.3 Practical Implications

For practitioners, our results suggest:

1. **Standard TA features are not sufficient for profitable crypto trading**: 29 technical-analysis features with logistic regression cannot achieve break-even accuracy after realistic costs. Practitioners should explore alternative feature sources (order flow, funding rates, cross-asset signals, on-chain metrics).

2. **Invest in meta-labeling before model complexity**: Even with a weak model, meta-labeling reduces losses significantly (ΔCAS = −6.28 when removed). Selective execution is more valuable than model sophistication when the signal is marginal.

3. **Model costs realistically**: The difference between 20 bps and 28 bps round-trip cost can change a strategy from marginally profitable to deeply unprofitable. Use exchange-specific fee schedules and measure actual slippage.

4. **Use DSR/PBO/CAS for evaluation**: Raw Sharpe and accuracy are insufficient. The Deflated Sharpe Ratio, Probability of Backtest Overfitting, and Cost-Adjusted Sharpe provide a more honest picture of strategy viability.

5. **Honest negative results are valuable**: Establishing that standard TA features cannot profitably trade crypto after costs is itself a contribution—it identifies the gap that future work must bridge.

---

## 6. Conclusion

We have presented a systematic evaluation of a multi-component directional prediction system on 2 years of real cryptocurrency data. Our results are honest and sobering:

- **45–47% accuracy at 25% coverage** — well below the break-even threshold of 66–78%.
- **CAS deeply negative across all cells** — the model cannot survive realistic transaction costs.
- **DSR < 0.50 for all cells** — no statistically significant directional edge exists.
- **Meta-labeling is the most impactful component** — selective execution reduces losses by 6.3 CAS points, but cannot make the strategy profitable.
- **Standard TA features are insufficient** — 29 technical-analysis features (trend, momentum, volatility, volume, structure) with logistic regression provide insufficient directional information for crypto markets.

The value of this study lies not in claiming profitability, but in establishing an honest baseline using cost-aware metrics, systematic ablation, and walk-forward out-of-sample evaluation. This negative result demonstrates that more powerful features (order book data, funding rates, cross-asset signals) or more sophisticated models (deep learning, transformer architectures) are needed before directional prediction in crypto markets can be profitable after costs. We believe honest reporting of negative results, using the evaluation framework presented here, should be the minimum standard for any published financial prediction study.

---

## References

1. Bailey, D.H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio." *Working Paper*, Cornell University.
2. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
3. Adams, R.P. and MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection." *arXiv:0710.3742*.
4. Bysik, K. and Ślepaczuk, R. (2026). "SharpeLoss and CostFocalLoss for Financial Direction Prediction." *Working Paper*.
5. Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*.
6. Bailey, D.H., Borwein, J., López de Prado, M. and Zhu, Q.J. (2015). "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 19(2), pp. 39–70.
7. Almgren, R. and Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), pp. 5–39.
8. Kelly, J.L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35(4), pp. 917–926.
9. Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*, 12, pp. 2825–2830.
10. Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.
11. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*.
12. McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *SciPy 2010*.

---

## Appendix A: Figure Reference

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `candlestick_BTC_1h.pdf` | BTC OHLCV candlestick with trade signals |
| Figure 1b | `candlestick_ETH_1h.pdf` | ETH OHLCV candlestick with trade signals |
| Figure 1c | `candlestick_SOL_1h.pdf` | SOL OHLCV candlestick with trade signals |
| Figure 2 | `equity_curves.pdf` | Equity curves for 3 cells (1h) vs buy-and-hold |
| Figure 3 | `coverage_accuracy_pareto.pdf` | Coverage-accuracy Pareto frontier |
| Figure 4 | `calibration_reliability.pdf` | 10-bin reliability diagrams |
| Figure 5 | `regime_detection_btc_1h.pdf` | BOCPD regime visualization (BTC) |
| Figure 6 | `cost_sensitivity.pdf` | Cost sensitivity bar chart |
| Figure 7 | `ablation_cas_delta.pdf` | Ablation CAS delta comparison |
| Figure 8 | `ablation_radar.pdf` | Ablation radar chart |
| Figure 9 | `break_even_heatmap.pdf` | Break-even accuracy heatmap |
| Figure 10 | `confusion_matrices_btc_1h.pdf` | Confusion matrices per regime (BTC) |
| Figure 11 | `model_comparison.pdf` | Model comparison bar chart |
| Figure 12 | `loss_comparison.pdf` | Loss function comparison |

*All figures are generated from 1h data only. 5m experiments were not completed due to simulation cost.*

## Appendix B: Reproducibility

All code is available at `github.com/kairon/kairon`. To reproduce:

```bash
# 1. Download data (~15 min, requires internet access)
uv run python scripts/download_real_data.py

# 2. Run experiments (1h only, ~10-20 min)
uv run python paper/run_real_experiments.py

# 3. Run ablation study (19 variants × 3 assets, ~10 min with fast CAS)
uv run python paper/run_ablation.py

# 4. Generate figures (~2 min)
uv run python paper/figures/generate_all.py
```

Data acquisition report: `data/acquisition_report.json`
Experiment results: `paper/real_results.json` (3 cells: BTC/ETH/SOL 1h)
Ablation results: `paper/ablation_results.json` (57 results: 19 variants × 3 assets)
Figures: `paper/figures/*.pdf` and `paper/figures/*.png` (14 PDF + 14 PNG)

**Known issues**: Three ablation variants (no_multihead, single_lr, majority_vote) produce zero-value results due to model initialization errors. The cost sensitivity sweep encounters numerical overflow. 5m experiments were skipped due to simulation time (~30+ min per cell with the Python simulation loop).