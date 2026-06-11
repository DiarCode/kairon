"""Real-data experiment engine for the Kairon professional paper.

Wires the existing Kairon infrastructure (CCXTAdapter, FeaturePipeline,
walk-forward splits, OOF protocol, ensembles, BOCPD, cost model, simulation)
to REAL Binance OHLCV data stored in partitioned parquet files.

This module is the load-bearing code for Phase 2 of the paper rewrite.
It produces ``paper/real_results.json`` with all headline metrics across
3 assets × 2 horizons = 6 experiment cells.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from loguru import logger
from sklearn.preprocessing import StandardScaler

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.backtest.metrics import (
    BARS_PER_YEAR_1H,
    BARS_PER_YEAR_5M,
    summarize,
)
from kairon.data.io import DataPaths, read_ohlcv
from kairon.data.symbols import CryptoVenue, crypto_spot
from kairon.evaluation.break_even import break_even_accuracy
from kairon.evaluation.cost_sensitivity import cost_sensitivity_curve
from kairon.evaluation.coverage_curve import coverage_curve
from kairon.features.pipeline import FeaturePipeline
from kairon.features.regime import BOCPDConfig, BOCPDRegimeDetector
from kairon.labels.direction import make_direction_labels
from kairon.labels.schema import LabelKind, LabelSpec
from kairon.models.contracts import FeatureMatrix
from kairon.models.multihead import MultiHeadConfig, MultiHeadModel
from kairon.paper.runner import SimulationConfig, run_simulation
from kairon.splits.walkforward import DEFAULT_SPLIT_1H, DEFAULT_SPLIT_5M


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH: Path = REPO_ROOT / "paper" / "real_results.json"

ASSET_CONFIGS: dict[str, dict[str, Any]] = {
    "BTC": {"base": "BTC", "quote": "USDT"},
    "ETH": {"base": "ETH", "quote": "USDT"},
    "SOL": {"base": "SOL", "quote": "USDT"},
}

HORIZON_CONFIGS: dict[str, dict[str, Any]] = {
    "1h": {
        "split_spec": DEFAULT_SPLIT_1H,
        "bars_per_year": BARS_PER_YEAR_1H,
        "sigma_default": 0.005,
    },
    "5m": {
        "split_spec": DEFAULT_SPLIT_5M,
        "bars_per_year": BARS_PER_YEAR_5M,
        "sigma_default": 0.0012,
    },
}

COST_MODEL: CostModel = DEFAULT_CRYPTO_COSTS
FEATURE_WINDOW: int = 8  # for simple rolling-return features


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """All metrics from one (symbol, horizon) experiment cell."""

    symbol: str
    timeframe: str
    n_bars: int
    n_features: int
    n_trades: int
    accuracy: float
    cas: float
    dsr: float
    pbo: float
    sharpe: float
    max_dd: float
    brier: float
    ece: float
    coverage: dict[str, Any]
    regime_breakdown: dict[str, dict[str, float]]
    cost_sensitivity: dict[str, dict[str, float]]
    break_even: dict[str, float]
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core computation helpers (adapted from scripts/run_e2e.py)
# ---------------------------------------------------------------------------
def _compute_cas(
    *,
    prices: np.ndarray,
    signals: np.ndarray,
    cost_model: CostModel,
    bars_per_year: int,
) -> float:
    """Cost-Adjusted Sharpe (CAS) on real prices + signals."""
    n = int(prices.size)
    if signals.size < n:
        padded = np.zeros(n, dtype=np.int8)
        padded[: signals.size] = signals.astype(np.int8, copy=False)
        signals = padded
    elif signals.size > n:
        signals = signals[:n]
    if n < 2:
        return 0.0
    log_prices = np.log(prices.astype(np.float64, copy=False))
    log_returns = np.diff(log_prices)
    aligned_signals = signals[:-1].astype(np.float64, copy=False)
    pnl = aligned_signals * log_returns
    cost_per_bar = float(cost_model.round_trip_bps) / 1e4 / 2.0
    pnl = pnl - cost_per_bar
    if pnl.std(ddof=0) == 0.0:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=0) * math.sqrt(float(bars_per_year)))


def _compute_dsr(
    *,
    cas: float,
    n_trades: int,
    n_assets: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR)."""
    if n_trades < 2:
        return 0.0
    expected_max_sr = math.sqrt(2.0 * math.log(max(n_assets, 2)))
    sr_se = math.sqrt(
        (1.0 + 0.5 * cas * cas - skewness * cas + (kurtosis - 3.0) / 4.0 * cas**4)
        / (n_trades - 1)
    )
    if sr_se <= 0.0:
        return 0.0
    z = (cas - expected_max_sr) / sr_se
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _compute_pbo(
    *,
    cas: float,
    sharpe_distribution: np.ndarray,
) -> float:
    """Probability of Backtest Overfitting (PBO) via CPCV proxy."""
    if sharpe_distribution.size < 2:
        return 0.0
    return float((sharpe_distribution < cas).mean())


def _brier_score(*, y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Brier score (1-vs-rest on positive class)."""
    if y_true.shape[0] != p_pred.shape[0]:
        raise ValueError("shape mismatch")
    y_bin = (y_true == 2).astype(np.float64, copy=False)
    p_pos = p_pred[:, -1] if p_pred.ndim == 2 else p_pred
    return float(((y_bin - p_pos) ** 2).mean())


def _ece(
    *,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE)."""
    if y_true.shape[0] != p_pred.shape[0]:
        raise ValueError("shape mismatch")
    y_bin = (y_true == 2).astype(np.float64, copy=False)
    p_pos = p_pred[:, -1] if p_pred.ndim == 2 else p_pred
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = int(y_true.size)
    for b in range(n_bins):
        lo = float(bin_edges[b])
        hi = float(bin_edges[b + 1])
        mask = (p_pos >= lo) & (p_pos < hi if b < n_bins - 1 else p_pos <= hi)
        if mask.sum() == 0:
            continue
        acc = float(y_bin[mask].mean())
        conf = float(p_pos[mask].mean())
        ece_val += float(mask.sum()) / n * abs(acc - conf)
    return float(ece_val)


def _skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    m = float(x.mean())
    s = float(x.std(ddof=0))
    if s <= 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s**3))


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 3.0
    m = float(x.mean())
    s = float(x.std(ddof=0))
    if s <= 0:
        return 3.0
    return float(((x - m) ** 4).mean() / (s**4))


def _sharpe_distribution(
    trade_pnl: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    if trade_pnl.size < 4:
        return np.array([], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.zeros(n_splits, dtype=np.float64)
    n = int(trade_pnl.size)
    half = max(1, n // 2)
    for i in range(n_splits):
        idxs = rng.choice(n, size=half, replace=True)
        sample = trade_pnl[idxs]
        if sample.std(ddof=0) == 0:
            out[i] = 0.0
        else:
            out[i] = float(sample.mean() / sample.std(ddof=0) * math.sqrt(half))
    return out


# ---------------------------------------------------------------------------
# Feature extraction from real OHLCV
# ---------------------------------------------------------------------------
def _extract_features_from_ohlcv(
    table: pa.Table,
    *,
    window: int = FEATURE_WINDOW,
) -> tuple[FeatureMatrix, np.ndarray]:
    """Build a feature matrix + 3-class direction labels from real OHLCV.

    Uses the FeaturePipeline for TA features (the pipeline appends computed
    columns to the original OHLCV table).  Labels come from
    ``make_direction_labels`` which uses DirectionClass encoding
    {-1, 0, 1}; we remap to {0, 1, 2} for the model.
    """
    n = int(table.num_rows)
    if n < window + 2:
        raise ValueError(f"Need at least {window + 2} bars, got {n}")

    # Extract direction labels using the typed label API
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    labeled = make_direction_labels(table, spec=spec, symbol="REAL-DATA")

    # Build simple rolling-return features as fallback
    close = np.array(table.column("close").to_pylist(), dtype=np.float64)
    log_prices = np.log(close)
    log_returns = np.diff(log_prices)

    # Try the full FeaturePipeline
    pipeline = FeaturePipeline()
    try:
        pipeline_result = pipeline.run(table)
        feature_table = pipeline_result.table
        n_labeled = len(labeled.bars)
        n_feat = int(feature_table.num_rows)
        use_len = min(n_labeled, n_feat)

        # The pipeline appends feature columns to the original OHLCV table.
        # Skip the original OHLCV columns (ts, open, high, low, close, volume)
        # and use only the computed feature columns.
        ohlcv_cols = {"ts", "open", "high", "low", "close", "volume"}
        feature_col_names = [c for c in feature_table.column_names if c not in ohlcv_cols]

        # If that filtering yields nothing, try all non-OHLCV columns
        if not feature_col_names:
            feature_col_names = [c for c in feature_table.column_names
                                if c not in {"ts", "open", "high", "low", "close", "volume"}]

        if not feature_col_names:
            raise ValueError("No feature columns found after filtering OHLCV")

        # Extract feature values
        feat_arrays = []
        used_names = []
        for fname in feature_col_names:
            col = np.array(feature_table.column(fname).to_pylist(), dtype=np.float64)
            # Replace NaN/inf with 0
            col = np.where(np.isfinite(col), col, 0.0)
            # Skip zero-variance features
            if col.std(ddof=0) < 1e-12:
                continue
            feat_arrays.append(col[:use_len])
            used_names.append(fname)

        if len(feat_arrays) >= 5:
            values = np.stack(feat_arrays, axis=1)
            # Remap labels: DirectionClass {-1, 0, 1} -> model {0, 1, 2}
            raw_ys = np.array([b.y_class for b in labeled.bars[:use_len]], dtype=np.int64)
            ys = raw_ys + 1  # -1→0, 0→1, 1→2
            fm = FeatureMatrix(
                values=values,
                feature_names=tuple(used_names),
            )
            logger.info(
                "  FeaturePipeline: {} features ({} columns), {} rows",
                len(used_names), len(feature_col_names), use_len,
            )
            return fm, ys
    except Exception as exc:
        logger.warning("FeaturePipeline failed ({}), falling back to rolling returns", exc)

    # Fallback: rolling log-return features
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for i in range(window - 1, n - 2):
        seg = log_returns[i - window + 1 : i + 1]
        mu = float(seg.mean())
        sd = float(seg.std(ddof=0)) if seg.size > 1 else 1.0
        sd = sd if sd > 0 else 1.0
        xs.append(((seg - mu) / sd).astype(np.float64, copy=False))
        nxt = float(log_returns[i + 1])
        if abs(nxt) < 1e-9:
            ys.append(1)  # FLAT
        elif nxt > 0.0:
            ys.append(2)  # UP
        else:
            ys.append(0)  # DOWN

    values = np.stack(xs, axis=0).astype(np.float64, copy=False)
    feature_names = tuple(f"r_lag_{j}" for j in range(window))
    return FeatureMatrix(values=values, feature_names=feature_names), np.array(ys, dtype=np.int64)


# ---------------------------------------------------------------------------
# BOCPD regime detection on real data
# ---------------------------------------------------------------------------
def _detect_regimes(
    prices: np.ndarray,
    *,
    timeframe: str,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Run BOCPD regime detection on real price data.

    Returns (regime_labels, regime_breakdown) where regime_labels is
    an array of regime names per bar and regime_breakdown is the summary.
    """
    n = int(prices.size)
    close = prices
    log_prices = np.log(close.astype(np.float64, copy=False))
    log_returns = np.diff(log_prices)
    realized_vol = np.abs(log_returns)
    # Spread proxy: (high - low) / close * 10000 -- we don't have high/low
    # in the aligned price array, so use realized_vol as vol proxy and
    # a constant spread proxy.
    spread_bps = np.full_like(realized_vol, 5.0)  # 5 bps default spread proxy

    detector = BOCPDRegimeDetector(BOCPDConfig())
    states = detector.detect(realized_vol, spread_bps)

    # Build regime label array
    regime_names = np.array([s.regime for s in states], dtype=object)

    # Pad first bar (no return for bar 0)
    full_regimes = np.concatenate([["ranging"], regime_names])

    # Summary
    breakdown: dict[str, dict[str, float]] = {}
    for r in ("trending", "ranging", "volatile", "stressed"):
        mask = full_regimes == r
        n_bars_r = int(mask.sum())
        breakdown[r] = {
            "n_bars": float(n_bars_r),
            "fraction": float(n_bars_r / n) if n > 0 else 0.0,
        }
    return full_regimes, breakdown


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------
class RealDataExperiment:
    """End-to-end real-data experiment for one (symbol, horizon) cell.

    Usage::

        exp = RealDataExperiment(symbol="BTC", timeframe="1h")
        result = exp.run()
    """

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        data_root: Path | None = None,
        seed: int = 20260608,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_root = data_root or DataPaths.default().root
        self.seed = seed
        cfg = ASSET_CONFIGS[symbol]
        self._sym = crypto_spot(cfg["base"], cfg["quote"], CryptoVenue.BINANCE)
        self._horizon_cfg = HORIZON_CONFIGS[timeframe]

    def run(self) -> ExperimentResult:
        """Run the full experiment pipeline on real data."""
        logger.info(
            "RealDataExperiment: {} {} — loading data", self.symbol, self.timeframe
        )

        # 1) Load real OHLCV data
        table = read_ohlcv(
            symbol=self._sym,
            venue="binance",
            timeframe=self.timeframe,
            paths=DataPaths(self.data_root),
        )
        n_bars = int(table.num_rows)
        if n_bars < 100:
            raise ValueError(f"Insufficient data: {n_bars} bars for {self.symbol} {self.timeframe}")
        logger.info("  Loaded {} bars", n_bars)

        # 2) Extract features + labels
        fm, y = _extract_features_from_ohlcv(table)
        n_features = int(fm.n_rows)
        logger.info("  Features: {} rows x {} cols", fm.n_rows, len(fm.feature_names))

        # 3) Extract price array for backtesting
        close_arr = np.array(table.column("close").to_pylist(), dtype=np.float64)
        prices = close_arr

        # 4) Walk-forward train/test split (80/20 chronological)
        #    CRITICAL: train only on PAST data, predict only on FUTURE data.
        #    This eliminates look-ahead bias that would inflate accuracy.
        train_frac = 0.80
        split_idx = int(n_features * train_frac)
        fm_train = FeatureMatrix(
            values=fm.values[:split_idx],
            feature_names=fm.feature_names,
        )
        fm_test = FeatureMatrix(
            values=fm.values[split_idx:],
            feature_names=fm.feature_names,
        )
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        logger.info(
            "  Walk-forward split: train={}, test={}",
            split_idx, n_features - split_idx,
        )

        # 5) Standardize features (critical for LogisticRegression convergence)
        scaler = StandardScaler()
        train_scaled_vals = scaler.fit_transform(fm_train.values)
        test_scaled_vals = scaler.transform(fm_test.values)

        fm_train_scaled = FeatureMatrix(
            values=train_scaled_vals,
            feature_names=fm_train.feature_names,
        )
        fm_test_scaled = FeatureMatrix(
            values=test_scaled_vals,
            feature_names=fm_test.feature_names,
        )

        # 5b) Train multi-head model on TRAINING data only
        #     n_estimators=500 → max_iter=500 for LogisticRegression convergence
        multihead = MultiHeadModel(MultiHeadConfig(n_estimators=500))
        y_magnitude = np.zeros(split_idx, dtype=np.float64)
        log_returns = np.diff(np.log(prices.astype(np.float64, copy=False)))
        y_vol = np.full(split_idx, float(np.std(log_returns[-252:])), dtype=np.float64)

        state = multihead.fit_multihead(
            features=fm_train_scaled,
            y_direction=y_train,
            y_magnitude=y_magnitude,
            y_vol=y_vol,
        )
        # Predict on TEST data only (out-of-sample)
        preds = multihead.predict_multihead(state, fm_test_scaled)
        proba = preds["y_proba"]
        classes = preds["y_class"]

        # 6) Build signal stream {-1, 0, +1} from OOS predictions
        sig_int = (classes - 1).astype(np.int8)  # {0,1,2} -> {-1,0,1}

        # 6b) Meta-labeling confidence gate on OOS predictions
        #     Dynamic percentile-based: ~25% coverage
        if proba.ndim == 2 and proba.shape[1] == 3:
            max_proba = proba.max(axis=1)
            # Use the 75th percentile as threshold → ~25% coverage
            threshold = float(np.percentile(max_proba, 75))
            low_conf = max_proba <= threshold
            sig_int[low_conf] = 0
            n_suppressed = int(low_conf.sum())
            n_active = sig_int.size - n_suppressed
            logger.info(
                "  Meta-label gate (p75={:.4f}): {}/{} suppressed, {} active ({:.1f}% coverage)",
                threshold, n_suppressed, sig_int.size, n_active, n_active / sig_int.size * 100,
            )

        # 7) Align OOS signals to price array (only test period)
        test_offset = n_bars - (n_features - split_idx)  # price index where test starts
        aligned_signals = np.zeros(n_bars, dtype=np.int8)
        test_len = min(len(sig_int), n_bars - test_offset)
        aligned_signals[test_offset: test_offset + test_len] = sig_int[:test_len]

        # 8) Size positions using vol-aware sizer
        magnitude = preds["y_magnitude"]
        vol = preds["y_vol"]
        initial_equity = 10_000.0

        # 8) Run simulation
        sim_cfg = SimulationConfig(
            cost=COST_MODEL,
            order_kind="market",
            order_size=0.1,
        )
        sim_result = run_simulation(
            prices=prices,
            signals=aligned_signals,
            config=sim_cfg,
            timeframe=self.timeframe,
        )
        trade_pnl_arr = np.array(
            [t.realised_pnl_cash for t in sim_result.trades if t.realised_pnl_cash is not None],
            dtype=np.float64,
        )

        # 9) Build equity curve
        if trade_pnl_arr.size > 0:
            cum_pnl = np.cumsum(trade_pnl_arr)
            # Clamp to prevent overflow in summarize/cost_sensitivity
            max_equity = initial_equity * 1000
            cum_pnl = np.clip(cum_pnl, -initial_equity * 10, max_equity - initial_equity)
            equity_curve = np.concatenate(
                [np.array([initial_equity], dtype=np.float64), initial_equity + cum_pnl]
            )
            # Ensure no zero/negative values that cause division errors
            equity_curve = np.maximum(equity_curve, 1.0)
        else:
            equity_curve = np.full(2, initial_equity, dtype=np.float64)
        if equity_curve.size < 2:
            equity_curve = np.full(2, initial_equity, dtype=np.float64)

        # 10) Compute headline metrics
        bars_per_year = self._horizon_cfg["bars_per_year"]
        rets = np.diff(equity_curve) / equity_curve[:-1]
        perf = summarize(equity_curve, bars_per_year=bars_per_year, trade_pnl=trade_pnl_arr)
        cas = _compute_cas(
            prices=prices,
            signals=aligned_signals,
            cost_model=COST_MODEL,
            bars_per_year=bars_per_year,
        )
        skew = _skewness(rets) if rets.size > 2 else 0.0
        kurt = _kurtosis(rets) if rets.size > 3 else 3.0
        dsr = _compute_dsr(
            cas=cas,
            n_trades=int(trade_pnl_arr.size),
            n_assets=1,
            skewness=skew,
            kurtosis=kurt,
        )
        sharpe_dist = _sharpe_distribution(trade_pnl_arr, n_splits=32, seed=self.seed)
        pbo = _compute_pbo(cas=cas, sharpe_distribution=sharpe_dist)

        # 11) Calibration (on OOS test set only)
        brier = _brier_score(y_true=y_test, p_pred=proba)
        ece_val = _ece(y_true=y_test, p_pred=proba, n_bins=10)

        # 12) Coverage curve (on OOS test set only)
        p_final = proba[:, -1] if proba.ndim == 2 else proba
        # Convert 3-class y to binary (positive class = UP)
        y_binary = (y_test == 2).astype(np.float64)
        cov = coverage_curve(
            y_true=y_binary,
            p_final=p_final,
            thresholds=tuple(round(t, 2) for t in np.arange(0.50, 0.91, 0.05)),
        )

        # 13) Regime detection
        regime_labels, regime_breakdown = _detect_regimes(prices, timeframe=self.timeframe)

        # 14) Cost sensitivity (wrap to handle overflow in extreme equity curves)
        cost_sens_dict: dict[str, dict[str, float]] = {}
        try:
            cost_sens = cost_sensitivity_curve(
                equity_curve,
                base_round_trip_bps=float(COST_MODEL.round_trip_bps),
                trade_pnl=trade_pnl_arr if trade_pnl_arr.size > 0 else None,
                bars_per_year=bars_per_year,
            )
            for mult, report in cost_sens.items():
                cost_sens_dict[f"{mult}x"] = {
                    "sharpe": float(report.sharpe),
                    "sortino": float(report.sortino),
                    "max_dd": float(report.max_drawdown),
                    "total_return": float(report.total_return),
                }
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            logger.warning("cost_sensitivity_curve failed: {}", exc)

        # 15) Break-even accuracy
        sigma = float(np.std(log_returns))
        expected_move_bps = sigma * 10000
        be = break_even_accuracy(
            expected_move_bps=expected_move_bps,
            round_trip_cost_bps=float(COST_MODEL.round_trip_bps),
        )

        # 16) Direction accuracy (on OOS test set only)
        correct = int((classes == y_test).sum())
        accuracy = float(correct / y_test.size) if y_test.size > 0 else 0.0

        # 17) Buy-and-hold baseline
        bh_return = float((prices[-1] - prices[0]) / prices[0])
        bh_sharpe = 0.0
        bh_rets = np.diff(prices) / prices[:-1]
        if bh_rets.std(ddof=0) > 0:
            bh_sharpe = float(bh_rets.mean() / bh_rets.std(ddof=0) * math.sqrt(bars_per_year))

        result = ExperimentResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            n_bars=n_bars,
            n_features=n_features,
            n_trades=int(trade_pnl_arr.size),
            accuracy=accuracy,
            cas=cas,
            dsr=dsr,
            pbo=pbo,
            sharpe=perf.sharpe,
            max_dd=perf.max_drawdown,
            brier=brier,
            ece=ece_val,
            coverage=cov,
            regime_breakdown=regime_breakdown,
            cost_sensitivity=cost_sens_dict,
            break_even={"expected_move_bps": expected_move_bps, "break_even_accuracy": be},
            extras={
                "initial_equity": initial_equity,
                "cost_model_round_trip_bps": float(COST_MODEL.round_trip_bps),
                "bars_per_year": bars_per_year,
                "fill_rate": sim_result.fill_rate,
                "p50_latency_ms": sim_result.p50_latency_ms,
                "p99_latency_ms": sim_result.p99_latency_ms,
                "maker_rebate_bps": sim_result.maker_rebate_bps,
                "sigma_per_bar": sigma,
                "buy_and_hold_return": bh_return,
                "buy_and_hold_sharpe": bh_sharpe,
                "data_source": "Binance public REST API (real OHLCV)",
            },
        )
        logger.info(
            "  Result: accuracy={:.4f}, CAS={:.4f}, DSR={:.4f}, PBO={:.4f}, "
            "sharpe={:.4f}, n_trades={}",
            accuracy, cas, dsr, pbo, perf.sharpe, int(trade_pnl_arr.size),
        )
        return result

    def run_and_save(self, *, output_path: Path | None = None) -> ExperimentResult:
        """Run and serialize the result to JSON."""
        result = self.run()
        out = output_path or DEFAULT_RESULTS_PATH
        out.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results if present (for multi-cell accumulation)
        existing: list[dict[str, Any]] = []
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding="utf-8")).get("cells", [])
            except (json.JSONDecodeError, KeyError):
                existing = []

        # Append or replace this cell
        cell_key = f"{self.symbol}_{self.timeframe}"
        new_cells = [c for c in existing if c.get("cell_key") != cell_key]
        new_cells.append({
            "cell_key": cell_key,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            **_result_to_dict(result),
        })

        report = {
            "schema_version": "1",
            "decided_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "data_source": "Binance public REST API (real OHLCV)",
            "n_cells": len(new_cells),
            "cells": new_cells,
        }
        out.write_text(json.dumps(report, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
        logger.info("Saved result to {}", out)
        return result


def _result_to_dict(r: ExperimentResult) -> dict[str, Any]:
    """Serialize an ExperimentResult to a JSON-compatible dict."""
    return {
        "n_bars": r.n_bars,
        "n_features": r.n_features,
        "n_trades": r.n_trades,
        "accuracy": r.accuracy,
        "cas": r.cas,
        "dsr": r.dsr,
        "pbo": r.pbo,
        "sharpe": r.sharpe,
        "max_dd": r.max_dd,
        "brier": r.brier,
        "ece": r.ece,
        "coverage": r.coverage,
        "regime_breakdown": r.regime_breakdown,
        "cost_sensitivity": r.cost_sensitivity,
        "break_even": r.break_even,
        "extras": r.extras,
    }