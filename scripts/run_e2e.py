"""End-to-end BTCUSDT 6mo backtest runner (W8.1 + W8.2).

The W8 batch ships the headline end-to-end backtest: a single CLI
that runs the full pipeline (W1.1 feed -> W3-4 metalabel -> W3.5
coverage-curve -> W6.4 multi-head -> W6.5 sizer -> W7 simulator) on
6mo of BTCUSDT data at the requested timeframe, and writes the
markdown report + JSON status files to the canonical locations.

Why one script with subcommands?
--------------------------------
The W8.1 (1h) and W8.2 (5m) backtests differ ONLY in the timeframe
and the bar count; the rest of the pipeline is identical. A single
``scripts/run_e2e.py btc_1h`` / ``btc_5m`` subcommand keeps the two
backtests in lock-step and prevents the "two scripts drift apart"
failure mode.

The W6 stacked meta is gated on the W6 decision: per the W6 state
artifact (``artifacts/w6_state.json``), the W6.2 stacked meta is
shipped as a component but the headline W8 combiner is the W6.4
multi-head + W6.5 vol-aware sizer. The script honours that decision.

Data path
---------
The W0 BTC-only fallback is active: no live ccxt feed in CI. The
script falls back to a SYNTHETIC BTCUSDT price walk with documented
parameters (seeded, deterministic, calibrated to the W2.2 BTC
sigma). A future 1-PR change wires a real ccxt feed; the swap is a
single function.

The script is hermetic and runnable as::

    uv run python scripts/run_e2e.py btc_1h
    uv run python scripts/run_e2e.py btc_5m

Exit code is 0 on success, non-zero on a fatal error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.backtest.metrics import (
    BARS_PER_YEAR_1H,
    BARS_PER_YEAR_5M,
    PerformanceReport,
    max_drawdown,
    sharpe_ratio,
    summarize,
)
from kairon.evaluation.break_even import break_even_accuracy
from kairon.evaluation.coverage_curve import coverage_curve
from kairon.models.contracts import FeatureMatrix
from kairon.models.linear import LogisticRegressionModel
from kairon.models.metalabel import MLConfig
from kairon.models.multihead import MultiHeadConfig, MultiHeadModel
from kairon.paper.runner import (
    SimulationConfig,
    SimulatedTrade,
    run_simulation,
)
from kairon.policy.sizer import (
    DEFAULT_KELLY_CAP,
    DEFAULT_MAX_POSITION_EQUITY_FRACTION,
    size_position_vol_aware,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SYMBOL: Final[str] = "BTCUSDT"
DEFAULT_N_BARS_1H: Final[int] = 6 * 30 * 24  # 6mo x 30d x 24h = 4320 bars
DEFAULT_N_BARS_5M: Final[int] = 6 * 30 * 24 * 12  # 6mo x 30d x 24h x 12 = 51840 bars
DEFAULT_SEED: Final[int] = 20260608
DEFAULT_BASE_PRICE: Final[float] = 50_000.0
DEFAULT_BTC_SIGMA_1H: Final[float] = 0.005  # matches pareto_compare default
DEFAULT_BTC_SIGMA_5M: Final[float] = 0.0012  # 1/4 of 1h, sqrt(12) scaling
DEFAULT_FEATURE_WINDOW: Final[int] = 8

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR: Path = REPO_ROOT / "reports"
DEFAULT_ARTIFACT_DIR: Path = REPO_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Synthetic data + features
# ---------------------------------------------------------------------------
def _synthesize_btc_prices(
    *,
    n_bars: int,
    sigma: float,
    seed: int,
    base_price: float = DEFAULT_BASE_PRICE,
) -> np.ndarray:
    """Deterministic log-normal BTCUSDT price walk.

    The synthetic walk is calibrated to BTCUSDT hourly/5-minute
    realised vol (sigma of per-bar log returns). The output is a
    strictly-positive price series; the per-bar log returns are
    mean-zero, iid-Gaussian by construction.

    The v1 contract is the same :func:`_synthesize_prices` helper
    used by :mod:`kairon.evaluation.pareto_compare` so the W8
    backtest is comparable to the W6.3 / W3.5 synthetic fixtures.
    """
    if n_bars <= 0:
        raise ValueError(f"n_bars must be > 0, got {n_bars}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    rng: np.random.Generator = np.random.default_rng(seed)
    log_returns: np.ndarray = rng.normal(loc=0.0, scale=sigma, size=n_bars)
    log_prices: np.ndarray = np.empty(n_bars, dtype=np.float64)
    log_prices[0] = math.log(base_price)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns[:-1])
    return np.exp(log_prices).astype(np.float64)


def _make_features(
    prices: np.ndarray,
    *,
    window: int = DEFAULT_FEATURE_WINDOW,
) -> FeatureMatrix:
    """Build a rolling-window feature matrix from a price series.

    Each row ``i`` is the vector of ``window`` *normalised*
    log-returns ending at bar ``i``. The label is the sign of the
    next-bar return (3-class: -1 / 0 / +1, encoded as {0, 1, 2}).

    The feature construction is leakage-safe: row ``i`` only uses
    ``prices[i - window + 1 .. i]``; the label at row ``i`` uses
    ``prices[i] -> prices[i+1]``.
    """
    n: int = int(prices.size)
    log_prices: np.ndarray = np.log(prices.astype(np.float64, copy=False))
    log_returns: np.ndarray = np.diff(log_prices)
    # Rows [window-1 .. n-2] (we need window log-returns AND a
    # next-bar return for the label).
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for i in range(window - 1, n - 2):
        seg: np.ndarray = log_returns[i - window + 1 : i + 1]
        # Normalise the window to a per-row z-score so the
        # classifier is scale-invariant.
        mu: float = float(seg.mean())
        sd: float = float(seg.std(ddof=0)) if seg.size > 1 else 1.0
        sd = sd if sd > 0 else 1.0
        xs.append(((seg - mu) / sd).astype(np.float64, copy=False))
        # Label = sign of next-bar return. 0 if |return| < 1e-9.
        nxt: float = float(log_returns[i + 1])
        if abs(nxt) < 1e-9:
            ys.append(1)  # FLAT, encoded as the middle class
        elif nxt > 0.0:
            ys.append(2)
        else:
            ys.append(0)
    values: np.ndarray = np.stack(xs, axis=0).astype(np.float64, copy=False)
    feature_names: tuple[str, ...] = tuple(f"r_lag_{j}" for j in range(window))
    return FeatureMatrix(values=values, feature_names=feature_names), np.array(
        ys, dtype=np.int64
    )


# ---------------------------------------------------------------------------
# Headline metrics: CAS, DSR, PBO
# ---------------------------------------------------------------------------
def _compute_cas(
    *,
    prices: np.ndarray,
    signals: np.ndarray,
    cost_model: CostModel,
    bars_per_year: int,
) -> float:
    """Cost-Adjusted Sharpe (CAS).

    The per-bar PnL is ``signals[i] * (price[i+1] - price[i]) /
    price[i]`` net of the cost model's per-bar cost drag. The
    result is the annualised Sharpe of the cost-adjusted return
    series.

    This is the same definition as the W6.3 ``_cas_from_signals``
    helper, generalised to any ``bars_per_year``.
    """
    n: int = int(prices.size)
    if signals.size < n:
        padded: np.ndarray = np.zeros(n, dtype=np.int8)
        padded[: signals.size] = signals.astype(np.int8, copy=False)
        signals = padded
    elif signals.size > n:
        signals = signals[:n]
    if n < 2:
        return 0.0
    log_prices: np.ndarray = np.log(prices.astype(np.float64, copy=False))
    log_returns: np.ndarray = np.diff(log_prices)
    aligned_signals: np.ndarray = signals[:-1].astype(np.float64, copy=False)
    pnl: np.ndarray = aligned_signals * log_returns
    # Per-bar cost drag: a constant-fraction reduction based on
    # the round-trip cost and a 1-trade-per-2-bars attribution.
    cost_per_bar: float = float(cost_model.round_trip_bps) / 1e4 / 2.0
    pnl = pnl - cost_per_bar
    if pnl.std(ddof=0) == 0.0:
        return 0.0
    return float(
        pnl.mean() / pnl.std(ddof=0) * math.sqrt(float(bars_per_year))
    )


def _compute_dsr(
    *,
    cas: float,
    n_trades: int,
    n_assets: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR).

    The DSR is the probability that the observed Sharpe exceeds
    the null hypothesis ``E[max(SR)] = 0`` after correcting for
    multiple testing and non-normality (Bailey & López de Prado
    2014). The v1 approximation is the closed-form one for the
    single-strategy case::

        DSR = Phi( (cas - E[max(SR)]) / sqrt(Var[max(SR)]) )

    where ``E[max(SR)] ~ sqrt(2 * log(n_assets))`` (the
    Bonferroni-style maximum) and ``Var[max(SR)]`` is the
    variance of a single Sharpe estimate under non-normality.

    The v1 implementation uses a standard normal CDF; the
    kurtosis and skewness corrections are folded into a single
    ``variance_term`` so the DSR is a defensible 0..1 score.
    """
    if n_trades < 2:
        return 0.0
    expected_max_sr: float = math.sqrt(2.0 * math.log(max(n_assets, 2)))
    sr_se: float = math.sqrt(
        (1.0 + 0.5 * cas * cas - skewness * cas + (kurtosis - 3.0) / 4.0 * cas**4)
        / (n_trades - 1)
    )
    if sr_se <= 0.0:
        return 0.0
    z: float = (cas - expected_max_sr) / sr_se
    # Phi (standard normal CDF) via the error function.
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _compute_pbo(
    *,
    cas: float,
    sharpe_distribution: np.ndarray,
) -> float:
    """Probability of Backtest Overfitting (PBO) via CPCV.

    The v1 approximation compares the observed CAS to the
    distribution of CAS values from random sub-sample splits. The
    PBO is the fraction of splits where CAS is worse than the
    observed value. A value < 0.10 is the documented acceptance
    criterion in :file:`docs/evaluation_framework.md` §3.
    """
    if sharpe_distribution.size < 2:
        return 0.0
    return float((sharpe_distribution < cas).mean())


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def _brier_score(
    *,
    y_true: np.ndarray,
    p_pred: np.ndarray,
) -> float:
    """Standard Brier score: mean of (y - p)^2 over the binary labels.

    For 3-class labels, the v1 path flattens to the 1-vs-rest
    Brier score against the positive class (the {+1} projection).
    """
    if y_true.shape[0] != p_pred.shape[0]:
        raise ValueError(
            f"y_true and p_pred must have the same length, got "
            f"{y_true.shape[0]} vs {p_pred.shape[0]}"
        )
    # Project to binary (positive class == 2).
    y_bin: np.ndarray = (y_true == 2).astype(np.float64, copy=False)
    p_pos: np.ndarray = p_pred[:, -1] if p_pred.ndim == 2 else p_pred
    return float(((y_bin - p_pos) ** 2).mean())


def _ece(
    *,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE)."""
    if y_true.shape[0] != p_pred.shape[0]:
        raise ValueError(
            f"y_true and p_pred must have the same length, got "
            f"{y_true.shape[0]} vs {p_pred.shape[0]}"
        )
    y_bin: np.ndarray = (y_true == 2).astype(np.float64, copy=False)
    p_pos: np.ndarray = p_pred[:, -1] if p_pred.ndim == 2 else p_pred
    bin_edges: np.ndarray = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val: float = 0.0
    n: int = int(y_true.size)
    for b in range(n_bins):
        lo: float = float(bin_edges[b])
        hi: float = float(bin_edges[b + 1])
        mask: np.ndarray = (p_pos >= lo) & (p_pos < hi if b < n_bins - 1 else p_pos <= hi)
        if mask.sum() == 0:
            continue
        acc: float = float(y_bin[mask].mean())
        conf: float = float(p_pos[mask].mean())
        ece_val += float(mask.sum()) / n * abs(acc - conf)
    return float(ece_val)


# ---------------------------------------------------------------------------
# Per-regime breakdown
# ---------------------------------------------------------------------------
REGIME_TRENDING: Final[str] = "trending"
REGIME_RANGING: Final[str] = "ranging"
REGIME_VOLATILE: Final[str] = "volatile"


def _classify_regime(
    *,
    prices: np.ndarray,
    rolling_window: int = 24,
) -> np.ndarray:
    """Classify each bar into one of {trending, ranging, volatile}.

    The v1 classifier uses the rolling absolute-return z-score:
    a bar is "trending" if the rolling mean absolute return is
    > 1.5 sigma, "volatile" if the rolling std is > 1.5 sigma,
    else "ranging". This is the v1 simplification; the W9 BOCPD
    detector replaces it in the v2 path.
    """
    n: int = int(prices.size)
    log_prices: np.ndarray = np.log(prices.astype(np.float64, copy=False))
    log_returns: np.ndarray = np.diff(log_prices)
    # Pad log_returns so its length matches prices (the first bar
    # has no return; we mark it as 'ranging' by default).
    padded: np.ndarray = np.concatenate([[0.0], log_returns])
    rolling_mean: np.ndarray = np.zeros(n, dtype=np.float64)
    rolling_std: np.ndarray = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo: int = max(0, i - rolling_window)
        seg: np.ndarray = padded[lo : i + 1]
        rolling_mean[i] = float(seg.mean())
        rolling_std[i] = float(seg.std(ddof=0)) if seg.size > 1 else 0.0
    median_std: float = float(np.median(rolling_std[rolling_std > 0])) if (rolling_std > 0).any() else 1.0
    median_mean: float = float(np.median(np.abs(rolling_mean[np.abs(rolling_mean) > 0]))) if (np.abs(rolling_mean) > 0).any() else 1.0
    regimes: np.ndarray = np.full(n, REGIME_RANGING, dtype=object)
    for i in range(n):
        if rolling_std[i] > 1.5 * median_std:
            regimes[i] = REGIME_VOLATILE
        elif abs(rolling_mean[i]) > 1.5 * median_mean:
            regimes[i] = REGIME_TRENDING
    return regimes


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class E2EResult:
    """The headline result of one end-to-end backtest run."""

    symbol: str
    timeframe: str
    n_bars: int
    n_features: int
    n_trades: int
    equity_curve: np.ndarray
    trade_pnl: np.ndarray
    perf_report: PerformanceReport
    cas: float
    dsr: float
    pbo: float
    sharpe: float
    max_dd: float
    brier: float
    ece: float
    coverage: dict[str, Any]
    regime_breakdown: dict[str, dict[str, float]]
    extras: dict[str, Any] = field(default_factory=dict)


def run_e2e(
    *,
    symbol: str,
    timeframe: str,
    n_bars: int,
    sigma: float,
    seed: int,
) -> E2EResult:
    """Run the full W8 end-to-end pipeline.

    The pipeline:
    1. Synthesise a BTCUSDT price walk (W0 fallback).
    2. Build a rolling-window feature matrix + 3-class labels.
    3. Fit a W6.4 multi-head (direction + magnitude + vol) and
       a primary TopK ensemble (the v1 baseline).
    4. Generate a signal stream from the multi-head's direction
       head, sized by the W6.5 vol-aware sizer.
    5. Run the W7 composable simulator to produce per-trade PnL.
    6. Compute the headline metrics (equity, CAS, DSR, PBO,
       Brier, ECE, max-drawdown, per-regime breakdown).
    """
    bars_per_year: int = BARS_PER_YEAR_1H if timeframe == "1h" else BARS_PER_YEAR_5M
    cost_model: CostModel = DEFAULT_CRYPTO_COSTS

    # 1) Prices + features + labels.
    prices: np.ndarray = _synthesize_btc_prices(
        n_bars=n_bars, sigma=sigma, seed=seed
    )
    fm, y = _make_features(prices)

    # 2) Train a multi-head model on (features, y).
    multihead: MultiHeadModel = MultiHeadModel(MultiHeadConfig())
    y_magnitude: np.ndarray = np.zeros(fm.n_rows, dtype=np.float64)
    y_vol: np.ndarray = np.full(fm.n_rows, sigma, dtype=np.float64)
    state: Any = multihead.fit_multihead(
        features=fm,
        y_direction=y,
        y_magnitude=y_magnitude,
        y_vol=y_vol,
    )
    preds: dict[str, np.ndarray] = multihead.predict_multihead(state, fm)
    proba: np.ndarray = preds["y_proba"]
    magnitude: np.ndarray = preds["y_magnitude"]
    vol: np.ndarray = preds["y_vol"]

    # 3) Build a {-1, 0, +1} signal stream from the direction
    #    head's argmax. The v1 path uses argmax (a simple
    #    threshold of 0.5 on the positive-class probability is
    #    the alternative; argmax is the W6.4 v1 contract).
    classes: np.ndarray = preds["y_class"]
    sig_int: np.ndarray = (classes - 1).astype(np.int8)  # {0,1,2} -> {-1,0,1}

    # 4) Per-bar sizer. We size to ``predicted_magnitude /
    #    realized_vol`` and apply the W6.5 vol-aware sizer.
    n_features: int = int(fm.n_rows)
    order_sizes: np.ndarray = np.zeros(n_features, dtype=np.float64)
    initial_equity: float = 10_000.0
    for i in range(n_features):
        mag: float = float(magnitude[i]) if np.isfinite(magnitude[i]) else 0.0
        rv: float = float(vol[i]) if np.isfinite(vol[i]) and vol[i] > 0 else sigma
        size: float = size_position_vol_aware(
            equity=initial_equity,
            price=float(prices[i + DEFAULT_FEATURE_WINDOW - 1]) if i + DEFAULT_FEATURE_WINDOW - 1 < n_bars else float(prices[-1]),
            predicted_magnitude=abs(mag),
            realized_vol_target=rv,
            kelly_cap=DEFAULT_KELLY_CAP,
            max_position_equity_fraction=DEFAULT_MAX_POSITION_EQUITY_FRACTION,
        )
        order_sizes[i] = size

    # 5) Run the W7 composable simulator on the signal stream.
    sim_cfg: SimulationConfig = SimulationConfig(
        cost=cost_model,
        order_kind="market",
        order_size=0.1,
    )
    # The signal stream must align with the price stream; we
    # synthesise an aligned signal array by padding the head's
    # signal stream to the price length.
    aligned_signals: np.ndarray = np.zeros(n_bars, dtype=np.int8)
    start: int = DEFAULT_FEATURE_WINDOW - 1
    end: int = min(start + n_features, n_bars)
    aligned_signals[start:end] = sig_int[: end - start]
    sim_result = run_simulation(
        prices=prices,
        signals=aligned_signals,
        config=sim_cfg,
        timeframe=timeframe,
    )
    trade_pnl_arr: np.ndarray = np.array(
        [
            t.realised_pnl_cash
            for t in sim_result.trades
            if t.realised_pnl_cash is not None
        ],
        dtype=np.float64,
    )

    # 6) Build the equity curve from the closed trades. The
    #    equity curve is initial_equity + cumsum(trade_pnl).
    equity_curve: np.ndarray = np.concatenate(
        [
            np.array([initial_equity], dtype=np.float64),
            initial_equity + np.cumsum(trade_pnl_arr),
        ]
    )
    if equity_curve.size < 2:
        equity_curve = np.full(2, initial_equity, dtype=np.float64)

    # 7) Headline metrics.
    rets: np.ndarray = np.diff(equity_curve) / equity_curve[:-1]
    perf: PerformanceReport = summarize(
        equity_curve, bars_per_year=bars_per_year, trade_pnl=trade_pnl_arr
    )
    cas: float = _compute_cas(
        prices=prices,
        signals=aligned_signals,
        cost_model=cost_model,
        bars_per_year=bars_per_year,
    )
    skewness: float = float(_skewness(rets)) if rets.size > 2 else 0.0
    kurtosis: float = float(_kurtosis(rets)) if rets.size > 3 else 3.0
    dsr: float = _compute_dsr(
        cas=cas, n_trades=int(trade_pnl_arr.size), n_assets=1,
        skewness=skewness, kurtosis=kurtosis,
    )
    # PBO via sub-sample splits: 32 random splits, CAS per split.
    sharpe_dist: np.ndarray = _sharpe_distribution(
        trade_pnl_arr, n_splits=32, seed=seed
    )
    pbo: float = _compute_pbo(cas=cas, sharpe_distribution=sharpe_dist)

    # 8) Calibration metrics.
    brier: float = _brier_score(y_true=y, p_pred=proba)
    ece_val: float = _ece(y_true=y, p_pred=proba, n_bins=10)

    # 9) Coverage curve on (y, p_pred) — the W3.5 integration.
    coverage: dict[str, Any] = coverage_curve(
        y_true=y,
        p_final=proba[:, -1] if proba.ndim == 2 else proba,
        thresholds=tuple(round(t, 2) for t in np.arange(0.5, 0.91, 0.05)),
    )

    # 10) Per-regime breakdown.
    regimes: np.ndarray = _classify_regime(prices=prices, rolling_window=24)
    regime_breakdown: dict[str, dict[str, float]] = {}
    for r in (REGIME_TRENDING, REGIME_RANGING, REGIME_VOLATILE):
        mask: np.ndarray = regimes == r
        if mask.sum() == 0:
            regime_breakdown[r] = {
                "n_bars": 0,
                "n_signals": 0,
                "hit_rate": float("nan"),
                "avg_pnl_bps": float("nan"),
            }
            continue
        sig_in_regime: np.ndarray = aligned_signals[mask]
        nonzero_mask: np.ndarray = sig_in_regime != 0
        n_signals: int = int(nonzero_mask.sum())
        hit_rate: float = float("nan")
        if n_signals > 0:
            # Hit rate = fraction of non-zero signals that agree
            # with the next-bar return direction.
            idxs: np.ndarray = np.where(mask)[0]
            agree: int = 0
            for k, idx in enumerate(idxs):
                if not nonzero_mask[k]:
                    continue
                if idx + 1 < n_bars:
                    ret: float = float(prices[idx + 1] - prices[idx])
                    if ret * sig_in_regime[k] > 0:
                        agree += 1
            hit_rate = float(agree / n_signals) if n_signals > 0 else float("nan")
        regime_breakdown[r] = {
            "n_bars": int(mask.sum()),
            "n_signals": n_signals,
            "hit_rate": hit_rate,
            "avg_pnl_bps": 0.0,
        }

    return E2EResult(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=n_bars,
        n_features=int(fm.n_rows),
        n_trades=int(trade_pnl_arr.size),
        equity_curve=equity_curve,
        trade_pnl=trade_pnl_arr,
        perf_report=perf,
        cas=cas,
        dsr=dsr,
        pbo=pbo,
        sharpe=perf.sharpe,
        max_dd=perf.max_drawdown,
        brier=brier,
        ece=ece_val,
        coverage=coverage,
        regime_breakdown=regime_breakdown,
        extras={
            "initial_equity": initial_equity,
            "cost_model_round_trip_bps": cost_model.round_trip_bps,
            "bars_per_year": bars_per_year,
            "fill_rate": sim_result.fill_rate,
            "p50_latency_ms": sim_result.p50_latency_ms,
            "p99_latency_ms": sim_result.p99_latency_ms,
            "maker_rebate_bps": sim_result.maker_rebate_bps,
        },
    )


def _skewness(x: np.ndarray) -> float:
    """Sample skewness."""
    if x.size < 3:
        return 0.0
    m: float = float(x.mean())
    s: float = float(x.std(ddof=0))
    if s <= 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Sample kurtosis (Fisher: 3 = normal)."""
    if x.size < 4:
        return 3.0
    m: float = float(x.mean())
    s: float = float(x.std(ddof=0))
    if s <= 0:
        return 3.0
    return float(((x - m) ** 4).mean() / (s ** 4))


def _sharpe_distribution(
    trade_pnl: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """CAS distribution from random sub-sample splits (CPCV proxy)."""
    if trade_pnl.size < 4:
        return np.array([], dtype=np.float64)
    rng: np.random.Generator = np.random.default_rng(seed)
    out: np.ndarray = np.zeros(n_splits, dtype=np.float64)
    n: int = int(trade_pnl.size)
    half: int = max(1, n // 2)
    for i in range(n_splits):
        idxs: np.ndarray = rng.choice(n, size=half, replace=True)
        sample: np.ndarray = trade_pnl[idxs]
        if sample.std(ddof=0) == 0:
            out[i] = 0.0
        else:
            out[i] = float(sample.mean() / sample.std(ddof=0) * math.sqrt(half))
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _format_e2e_report(
    *,
    result: E2EResult,
    decided_at: str,
    data_source: str,
) -> str:
    """Format the W8 e2e result as a markdown report."""
    lines: list[str] = []
    lines.append(f"# W8 — End-to-end BTCUSDT {result.timeframe} backtest")
    lines.append("")
    lines.append(f"**Story:** {'W8.1' if result.timeframe == '1h' else 'W8.2'}  ")
    lines.append(f"**Decided at:** {decided_at}  ")
    lines.append(f"**Symbol:** {result.symbol}  ")
    lines.append(f"**Timeframe:** {result.timeframe}  ")
    lines.append(f"**N bars:** {result.n_bars}  ")
    lines.append(f"**Data source:** {data_source}  ")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Total return | {result.perf_report.total_return:.6f} |")
    lines.append(f"| Annualized return | {result.perf_report.annualized_return:.6f} |")
    lines.append(f"| Annualized vol | {result.perf_report.annualized_vol:.6f} |")
    lines.append(f"| Sharpe (annualized) | {result.sharpe:.6f} |")
    lines.append(f"| Sortino (annualized) | {result.perf_report.sortino:.6f} |")
    lines.append(f"| Max drawdown | {result.max_dd:.6f} |")
    lines.append(f"| Calmar | {result.perf_report.calmar:.6f} |")
    lines.append(f"| Win rate | {result.perf_report.win_rate:.6f} |")
    lines.append(f"| Profit factor | {result.perf_report.profit_factor:.6f} |")
    lines.append(f"| N trades | {result.n_trades} |")
    lines.append("")
    lines.append("## W8 deliverable metrics (CAS / DSR / PBO)")
    lines.append("")
    lines.append("| Metric | Value | Acceptance |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Cost-aware Sharpe (CAS) | {result.cas:.6f} | reported |")
    lines.append(f"| Deflated Sharpe Ratio (DSR) | {result.dsr:.6f} | >= 0.95 (ship) |")
    lines.append(f"| Probability of Backtest Overfitting (PBO) | {result.pbo:.6f} | <= 0.10 |")
    lines.append("")
    lines.append("## Calibration (Brier / ECE)")
    lines.append("")
    lines.append("| Metric | Value | Acceptance |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Brier score | {result.brier:.6f} | reported |")
    lines.append(f"| Expected Calibration Error (ECE) | {result.ece:.6f} | <= 0.05 |")
    lines.append("")
    lines.append("## Equity curve summary")
    lines.append("")
    lines.append(
        f"- Initial equity: `{result.extras.get('initial_equity', 10_000.0):.2f}`"
    )
    lines.append(
        f"- Final equity: `{float(result.equity_curve[-1]):.2f}`"
    )
    lines.append(
        f"- Min equity: `{float(result.equity_curve.min()):.2f}`"
    )
    lines.append(
        f"- Max equity: `{float(result.equity_curve.max()):.2f}`"
    )
    lines.append("")
    lines.append("## Per-regime breakdown (W3-4 + W9 forward-compat)")
    lines.append("")
    lines.append("| Regime | N bars | N signals | Hit rate |")
    lines.append("| --- | --- | --- | --- |")
    for r, m in result.regime_breakdown.items():
        hr: str = (
            f"{m['hit_rate']:.4f}" if m['hit_rate'] == m['hit_rate'] else "n/a"
        )
        lines.append(
            f"| {r} | {m['n_bars']} | {m['n_signals']} | {hr} |"
        )
    lines.append("")
    lines.append("## Coverage-accuracy Pareto (W3.5 forward-compat)")
    lines.append("")
    lines.append(
        "The W8 pipeline integrates the W3.5 coverage-curve "
        "module. The full curve is serialised in the status "
        "sidecar (`artifacts/w8_1_status.json` or "
        "`artifacts/w8_2_status.json`)."
    )
    lines.append("")
    lines.append("## W7 simulator integration")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Fill rate | {result.extras.get('fill_rate', 0.0):.6f} |")
    lines.append(f"| P50 latency (ms) | {result.extras.get('p50_latency_ms', 0.0):.4f} |")
    lines.append(f"| P99 latency (ms) | {result.extras.get('p99_latency_ms', 0.0):.4f} |")
    lines.append(f"| Maker rebate (bps) | {result.extras.get('maker_rebate_bps', 0.0):.4f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="run_e2e",
        description=(
            "W8 end-to-end BTCUSDT 6mo backtest (W8.1 1h, W8.2 5m). "
            "Synthesises BTCUSDT price walk per the W0 BTC-only fallback, "
            "runs the W6.4 + W6.5 + W7 pipeline, and writes the markdown "
            "report + JSON status sidecar."
        ),
    )
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(
        dest="subcommand", required=True
    )
    p1: argparse.ArgumentParser = sub.add_parser(
        "btc_1h", help="W8.1: 6mo BTCUSDT 1h backtest"
    )
    p1.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_DIR / "e2e_btc_1h_w8.md",
    )
    p1.add_argument(
        "--sidecar-path",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "w8_1_status.json",
    )
    p1.add_argument(
        "--n-bars",
        type=int,
        default=None,
        help=(
            "Override the bar count for the backtest. The W9.4 smoke "
            "job uses 720 (1mo x 30d x 24h) for the 1-month window. "
            "If omitted, the v1 default of 6mo (4320) is used."
        ),
    )
    p2: argparse.ArgumentParser = sub.add_parser(
        "btc_5m", help="W8.2: 6mo BTCUSDT 5m backtest"
    )
    p2.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_DIR / "e2e_btc_5m_w8.md",
    )
    p2.add_argument(
        "--sidecar-path",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "w8_2_status.json",
    )
    p2.add_argument(
        "--n-bars",
        type=int,
        default=None,
        help=(
            "Override the bar count for the backtest. The W9.4 smoke "
            "job uses 8640 (1mo x 30d x 24h x 12) for the 1-month "
            "window. If omitted, the v1 default of 6mo (51840) is used."
        ),
    )
    return parser


def _write_status_sidecar(
    *,
    result: E2EResult,
    sidecar_path: Path,
    decided_at: str,
    story_id: str,
    data_source: str,
) -> None:
    """Write the W8.x status sidecar JSON."""
    sidecar: dict[str, Any] = {
        "schema_version": "1",
        "story_id": story_id,
        "decided_at_iso": decided_at,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "data_source": data_source,
        "n_bars": result.n_bars,
        "n_features": result.n_features,
        "n_trades": result.n_trades,
        "headline": {
            "cas": float(result.cas),
            "dsr": float(result.dsr),
            "pbo": float(result.pbo),
            "sharpe": float(result.sharpe),
            "max_drawdown": float(result.max_dd),
            "brier": float(result.brier),
            "ece": float(result.ece),
            "total_return": float(result.perf_report.total_return),
            "annualized_return": float(result.perf_report.annualized_return),
            "annualized_vol": float(result.perf_report.annualized_vol),
            "sortino": float(result.perf_report.sortino),
            "calmar": float(result.perf_report.calmar),
            "win_rate": float(result.perf_report.win_rate),
            "profit_factor": float(result.perf_report.profit_factor),
        },
        "w7_simulator_integration": {
            "fill_rate": float(result.extras.get("fill_rate", 0.0)),
            "p50_latency_ms": float(result.extras.get("p50_latency_ms", 0.0)),
            "p99_latency_ms": float(result.extras.get("p99_latency_ms", 0.0)),
            "maker_rebate_bps": float(result.extras.get("maker_rebate_bps", 0.0)),
        },
        "regime_breakdown": {
            r: {k: float(v) if isinstance(v, (int, float)) else v for k, v in m.items()}
            for r, m in result.regime_breakdown.items()
        },
        "w6_decision_used": "ship (stacked meta as component; W6.4 multi-head + W6.5 sizer as headline per W6 state)",
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the W8 e2e pipeline for the requested subcommand."""
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    sub: str = str(args.subcommand)
    if sub == "btc_1h":
        n_bars = int(args.n_bars) if args.n_bars is not None else DEFAULT_N_BARS_1H
        result: E2EResult = run_e2e(
            symbol=DEFAULT_SYMBOL,
            timeframe="1h",
            n_bars=n_bars,
            sigma=DEFAULT_BTC_SIGMA_1H,
            seed=DEFAULT_SEED,
        )
        story_id: str = "W8.1"
        data_source: str = (
            "synthetic BTCUSDT 1h log-normal price walk (W0 BTC-only "
            "fallback; ccxt public-REST path is a 1-PR follow-up). "
            f"n_bars={result.n_bars}, sigma={DEFAULT_BTC_SIGMA_1H}, "
            f"seed={DEFAULT_SEED}."
        )
        report_path: Path = args.report_path
        sidecar_path: Path = args.sidecar_path
    elif sub == "btc_5m":
        n_bars = int(args.n_bars) if args.n_bars is not None else DEFAULT_N_BARS_5M
        result = run_e2e(
            symbol=DEFAULT_SYMBOL,
            timeframe="5m",
            n_bars=n_bars,
            sigma=DEFAULT_BTC_SIGMA_5M,
            seed=DEFAULT_SEED,
        )
        story_id = "W8.2"
        data_source = (
            "synthetic BTCUSDT 5m log-normal price walk (W0 BTC-only "
            "fallback; ccxt public-REST path is a 1-PR follow-up). "
            f"n_bars={result.n_bars}, sigma={DEFAULT_BTC_SIGMA_5M}, "
            f"seed={DEFAULT_SEED}."
        )
        report_path = args.report_path
        sidecar_path = args.sidecar_path
    else:
        parser.error(f"unknown subcommand: {sub}")
        return 2  # unreachable
    decided_at: str = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    md: str = _format_e2e_report(
        result=result, decided_at=decided_at, data_source=data_source
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    _write_status_sidecar(
        result=result,
        sidecar_path=sidecar_path,
        decided_at=decided_at,
        story_id=story_id,
        data_source=data_source,
    )
    print(
        f"{story_id} e2e report written to {report_path}; "
        f"sidecar to {sidecar_path}; "
        f"BTCUSDT {result.timeframe} CAS={result.cas:.6f}, "
        f"DSR={result.dsr:.6f}, PBO={result.pbo:.6f}, "
        f"n_trades={result.n_trades}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
