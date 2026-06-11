"""Weekly prediction experiment for BTC and ETH.

Loads 1W CSV data, runs the full feature pipeline (ALL_FEATURES),
constructs direction/magnitude/vol labels, and evaluates three models
(LogisticRegression baseline, TreeMultiHead, StackedMultiHead) via
walk-forward cross-validation.

Usage:
    python scripts/predict_weekly.py
    python scripts/predict_weekly.py --asset BTC
    python scripts/predict_weekly.py --asset ETH --feature-set default
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from sklearn.preprocessing import StandardScaler

from kairon.data.io import OHLCV_SCHEMA
from kairon.evaluation.walkforward_cv import WalkForwardCV, cost_adjusted_sharpe
from kairon.features.pipeline import FeaturePipeline
from kairon.features.registry import ALL_FEATURES, DEFAULT_FEATURES
from kairon.labels.direction import make_direction_labels
from kairon.labels.magnitude import make_magnitude_labels
from kairon.labels.schema import LabelKind, LabelSpec
# Volatility labels are computed manually for weekly data (see build_labels)
from kairon.models.contracts import FeatureMatrix
from kairon.models.multihead import MultiHeadConfig, MultiHeadModel
from kairon.models.stacked_multihead import StackedMultiHeadConfig, StackedMultiHeadModel
from kairon.models.tree_multihead import TreeMultiHeadConfig, TreeMultiHeadModel

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1A. CSV Loader
# ---------------------------------------------------------------------------

def load_weekly_csv(path: Path, symbol: str = "UNKNOWN") -> pa.Table:
    """Load a semicolon-delimited weekly CSV into an OHLCV pyarrow Table.

    Handles:
    - Semicolon delimiter
    - 'time' → 'ts' column rename
    - 'Volume' → 'volume' column rename (case-insensitive)
    - Missing volume column (fills with 0.0)
    - Timestamp parsing with UTC timezone
    """
    logger.info("Loading {} weekly data from {}", symbol, path)
    df = pd.read_csv(path, sep=";")

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower == "time":
            col_map[col] = "ts"
        elif lower == "volume":
            col_map[col] = "volume"
        elif lower == "open":
            col_map[col] = "open"
        elif lower == "high":
            col_map[col] = "high"
        elif lower == "low":
            col_map[col] = "low"
        elif lower == "close":
            col_map[col] = "close"
    df = df.rename(columns=col_map)

    # Add volume if missing
    if "volume" not in df.columns:
        logger.info("  No volume column found for {}; filling with 0.0", symbol)
        df["volume"] = 0.0

    # Parse timestamps
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Sort ascending
    df = df.sort_values("ts").reset_index(drop=True)

    # Ensure required columns exist and are float
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(np.float64)

    # Select and order columns
    df = df[["ts", "open", "high", "low", "close", "volume"]]

    # Convert to pyarrow with OHLCV_SCHEMA
    table = pa.Table.from_pandas(df, schema=OHLCV_SCHEMA)
    logger.info("  Loaded {} bars for {} ({} to {})",
                table.num_rows, symbol,
                df["ts"].iloc[0].date(),
                df["ts"].iloc[-1].date())
    return table


# ---------------------------------------------------------------------------
# 1B. Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(
    table: pa.Table,
    *,
    feature_set: str = "all",
) -> tuple[FeatureMatrix, pa.Table]:
    """Build a feature matrix from OHLCV data using the feature pipeline.

    Parameters
    ----------
    table : pa.Table
        OHLCV data with columns: ts, open, high, low, close, volume.
    feature_set : str
        "all" for ALL_FEATURES, "default" for DEFAULT_FEATURES.

    Returns
    -------
    (FeatureMatrix, augmented_table)
        FeatureMatrix with NaN/inf replaced and zero-variance features dropped.
        augmented_table is the full pipeline output for label alignment.
    """
    features = ALL_FEATURES if feature_set == "all" else DEFAULT_FEATURES
    logger.info("Building features with {} feature builders...", len(features))

    pipeline = FeaturePipeline(features=features)

    # Some features may fail on weekly data (e.g. volume features when volume=0).
    # Run each builder individually so we can skip failures gracefully.
    cur = table
    feature_col_names = []

    from kairon.features.registry import get as get_feature

    for fname in features:
        try:
            spec = get_feature(fname)
            result_table = spec.builder(cur)
            # Find new columns added by this builder
            new_cols = [c for c in result_table.column_names if c not in cur.column_names]
            if new_cols:
                cur = result_table
                feature_col_names.extend(new_cols)
                logger.debug("  {}: +{} columns", fname, len(new_cols))
            else:
                logger.debug("  {}: no new columns (skipped)", fname)
        except Exception as exc:
            logger.warning("  {}: FAILED — {} (skipping)", fname, exc)

    n_rows = cur.num_rows
    ohlcv_cols = {"ts", "open", "high", "low", "close", "volume"}

    # Extract feature values, replacing NaN/inf with 0
    feat_arrays = []
    used_names = []
    for fname in feature_col_names:
        if fname in ohlcv_cols:
            continue
        col = np.array(cur.column(fname).to_pylist(), dtype=np.float64)
        # Replace NaN/inf with 0
        col = np.where(np.isfinite(col), col, 0.0)
        # Skip zero-variance features
        if col.std(ddof=0) < 1e-12:
            logger.debug("  Skipping zero-variance feature: {}", fname)
            continue
        feat_arrays.append(col)
        used_names.append(fname)

    if len(feat_arrays) < 3:
        raise ValueError(f"Only {len(feat_arrays)} features survived filtering; need at least 3")

    values = np.stack(feat_arrays, axis=1)
    fm = FeatureMatrix(values=values, feature_names=tuple(used_names))
    logger.info("  Feature matrix: {} rows x {} features", fm.n_rows, len(fm.feature_names))
    return fm, cur


# ---------------------------------------------------------------------------
# 1C. Label Construction
# ---------------------------------------------------------------------------

def build_labels(
    table: pa.Table,
    *,
    horizon: str = "1w",
    flat_threshold_pct: float = 0.005,
    vol_window: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build direction, magnitude, and volatility labels.

    For weekly data, volatility labels are computed as rolling realized
    volatility over ``vol_window`` bars (default 4 weeks), because the
    1-week horizon provides only 2 bars — insufficient for the standard
    ``make_volatility_labels`` which requires ≥3 bars.

    Direction and magnitude labels use the Kairon label API with a 1-week
    horizon, which works correctly (1-bar lookahead on weekly data).

    Parameters
    ----------
    table : pa.Table
        OHLCV table with ``ts``, ``close`` columns.
    horizon : str
        Prediction horizon (default "1w").
    flat_threshold_pct : float
        Half-width of the flat band for direction labels (default 0.5%).
    vol_window : int
        Rolling window in bars for realized volatility (default 4).

    Returns
    -------
    (y_direction, y_magnitude, y_vol)
        y_direction: {0, 1, 2} for {DOWN, FLAT, UP}, shape (N,)
        y_magnitude: log-return at horizon, shape (N,)
        y_vol: realized volatility, shape (N,)
    """
    # Use the label API for direction and magnitude (1-bar lookahead works fine)
    spec_dir = LabelSpec(kind=LabelKind.DIRECTION, horizon=horizon)
    spec_mag = LabelSpec(kind=LabelKind.MAGNITUDE, horizon=horizon)

    dir_frame = make_direction_labels(table, spec=spec_dir, symbol="WEEKLY")
    mag_frame = make_magnitude_labels(table, spec=spec_mag, symbol="WEEKLY")

    # Align direction and magnitude labels by timestamp
    dir_by_ts = {b.ts: b for b in dir_frame.bars}
    mag_by_ts = {b.ts: b for b in mag_frame.bars}
    common_ts = sorted(set(dir_by_ts.keys()) & set(mag_by_ts.keys()))

    logger.info("  Labels: {} direction, {} magnitude, {} aligned",
                len(dir_frame.bars), len(mag_frame.bars), len(common_ts))

    if len(common_ts) < 50:
        raise ValueError(f"Only {len(common_ts)} aligned labels; need at least 50")

    y_direction = np.array([dir_by_ts[t].y_class + 1 for t in common_ts], dtype=np.int64)
    y_magnitude = np.array([mag_by_ts[t].y for t in common_ts], dtype=np.float64)

    # Compute volatility labels manually: rolling std of log-returns over vol_window bars
    # This avoids the make_volatility_labels issue (needs ≥3 bars per window, but
    # 1-week horizon on weekly data only gives 2 bars).
    close = np.array(table.column("close").to_pylist(), dtype=np.float64)
    ts_list = table.column("ts").to_pylist()
    log_returns = np.diff(np.log(close))

    # Map common_ts timestamps to bar indices
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    y_vol = np.zeros(len(common_ts), dtype=np.float64)
    for i, t in enumerate(common_ts):
        bar_idx = ts_to_idx.get(t)
        if bar_idx is not None and bar_idx >= vol_window:
            window = log_returns[bar_idx - vol_window:bar_idx]
            y_vol[i] = float(np.std(window, ddof=1))
        elif bar_idx is not None:
            # Not enough history for full window; use available bars
            window = log_returns[:bar_idx] if bar_idx > 1 else np.array([0.0001])
            y_vol[i] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0001
        else:
            y_vol[i] = 0.0001  # tiny fallback

    # Ensure vol is non-zero (needed for model)
    y_vol = np.maximum(y_vol, 1e-8)

    logger.info("  Volatility: computed manually with {}-bar rolling window, mean={:.6f}",
                vol_window, float(y_vol.mean()))

    return y_direction, y_magnitude, y_vol


# ---------------------------------------------------------------------------
# 1D. Walk-Forward CV Evaluation
# ---------------------------------------------------------------------------

def evaluate_walk_forward(
    fm: FeatureMatrix,
    y_direction: np.ndarray,
    y_magnitude: np.ndarray,
    y_vol: np.ndarray,
    prices: np.ndarray,
    *,
    n_folds: int = 5,
    min_train: int = 100,
    model_types: tuple[str, ...] = ("lr", "tree", "stacked"),
) -> dict[str, dict[str, Any]]:
    """Run walk-forward CV with multiple model types.

    Returns a dict of model_type -> results dict with per-fold and aggregate metrics.
    """
    n_samples = fm.n_rows
    logger.info("Walk-forward CV: {} samples, {} folds, min_train={}",
                n_samples, n_folds, min_train)

    cv = WalkForwardCV(n_folds=n_folds, min_train=min_train, gap=0, val_ratio=0.2)
    folds = cv.split(fm.values)

    # Compute log-returns aligned to feature matrix.
    # prices may be longer than features (original data) or the same length.
    # log_returns[i] = return from bar i to bar i+1, same index as features[i].
    log_returns_full = np.diff(np.log(prices.astype(np.float64, copy=False)))
    # Slice to match feature matrix length
    log_returns = log_returns_full[:n_samples]

    results: dict[str, dict[str, Any]] = {}

    for model_type in model_types:
        logger.info("  Evaluating model: {}", model_type)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Split data
            X_train = fm.values[train_idx]
            X_val = fm.values[val_idx]

            y_dir_train = y_direction[train_idx]
            y_dir_val = y_direction[val_idx]

            # Align magnitude and vol to available training data
            y_mag_train = y_magnitude[train_idx] if len(y_magnitude) >= len(train_idx) else y_magnitude[:len(train_idx)]
            y_mag_val = y_magnitude[val_idx] if len(y_magnitude) >= len(val_idx) else y_magnitude[:len(val_idx)]

            y_vol_train = y_vol[train_idx] if len(y_vol) >= len(train_idx) else y_vol[:len(train_idx)]
            y_vol_val = y_vol[val_idx] if len(y_vol) >= len(val_idx) else y_vol[:len(val_idx)]

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            fm_train = FeatureMatrix(values=X_train_scaled, feature_names=fm.feature_names)
            fm_val = FeatureMatrix(values=X_val_scaled, feature_names=fm.feature_names)

            # Fit model
            try:
                if model_type == "lr":
                    model = MultiHeadModel(MultiHeadConfig(n_estimators=500))
                    state = model.fit_multihead(fm_train, y_dir_train, y_mag_train, y_vol_train)
                    preds = model.predict_multihead(state, fm_val)
                elif model_type == "tree":
                    model = TreeMultiHeadModel(TreeMultiHeadConfig(n_estimators=300))
                    state = model.fit_multihead(fm_train, y_dir_train, y_mag_train, y_vol_train)
                    preds = model.predict_multihead(state, fm_val)
                elif model_type == "stacked":
                    model = StackedMultiHeadModel(StackedMultiHeadConfig(
                        n_estimators=300,
                        confidence_threshold=0.55,
                    ))
                    state = model.fit_multihead(fm_train, y_dir_train, y_mag_train, y_vol_train)
                    preds = model.predict_multihead(state, fm_val)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            except Exception as exc:
                logger.warning("    Fold {} FAILED for {}: {}", fold_idx, model_type, exc)
                continue

            y_pred = preds["y_class"]
            y_proba = preds["y_proba"]

            # Compute metrics
            metrics = WalkForwardCV.compute_metrics(
                y_true=y_dir_val,
                y_pred=y_pred,
                y_proba=y_proba,
                confidence_threshold=0.0,
            )

            # CAS — log_returns is aligned to feature matrix indices
            if len(log_returns) >= max(val_idx) + 1:
                fold_returns = log_returns[val_idx]
                cas = cost_adjusted_sharpe(
                    y_true=y_dir_val,
                    y_pred=y_pred,
                    returns=fold_returns,
                    y_proba=y_proba,
                    cost_bps=28.0,
                )
            else:
                cas = 0.0

            # Direction accuracy (excluding flat class)
            non_flat = y_dir_val != 1
            dir_acc = float(np.mean(y_dir_val[non_flat] == y_pred[non_flat])) if non_flat.sum() > 0 else metrics["accuracy"]

            fold_result = {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "accuracy": metrics["accuracy"],
                "direction_accuracy": dir_acc,
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
                "coverage": metrics["coverage"],
                "cas": cas,
            }
            fold_metrics.append(fold_result)
            logger.info(
                "    Fold {}/{}: acc={:.3f} dir_acc={:.3f} CAS={:.3f} Brier={:.4f} ECE={:.4f}",
                fold_idx + 1, len(folds),
                metrics["accuracy"], dir_acc, cas,
                metrics["brier_score"] if not math.isnan(metrics["brier_score"]) else float("nan"),
                metrics["ece"] if not math.isnan(metrics["ece"]) else float("nan"),
            )

        # Aggregate
        if fold_metrics:
            avg_acc = np.mean([f["accuracy"] for f in fold_metrics])
            avg_dir_acc = np.mean([f["direction_accuracy"] for f in fold_metrics])
            avg_cas = np.mean([f["cas"] for f in fold_metrics])
            avg_brier = np.mean([f["brier_score"] for f in fold_metrics if not math.isnan(f["brier_score"])])
            avg_ece = np.mean([f["ece"] for f in fold_metrics if not math.isnan(f["ece"])])
            std_acc = np.std([f["accuracy"] for f in fold_metrics])

            results[model_type] = {
                "folds": fold_metrics,
                "mean_accuracy": float(avg_acc),
                "std_accuracy": float(std_acc),
                "mean_direction_accuracy": float(avg_dir_acc),
                "mean_cas": float(avg_cas),
                "mean_brier": float(avg_brier) if not math.isnan(avg_brier) else float("nan"),
                "mean_ece": float(avg_ece) if not math.isnan(avg_ece) else float("nan"),
            }
        else:
            results[model_type] = {"error": "all folds failed"}

    return results


# ---------------------------------------------------------------------------
# 1E. Full Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(
    csv_path: Path,
    symbol: str,
    *,
    feature_set: str = "all",
    model_types: tuple[str, ...] = ("lr", "tree", "stacked"),
    n_folds: int = 5,
    min_train: int = 100,
    horizon: str = "1w",
) -> dict[str, Any]:
    """Run the full prediction experiment for one asset.

    Parameters
    ----------
    csv_path : Path
        Path to semicolon-delimited weekly CSV.
    symbol : str
        "BTC" or "ETH".
    feature_set : str
        "all" for ALL_FEATURES, "default" for DEFAULT_FEATURES.
    model_types : tuple
        Which models to evaluate: "lr", "tree", "stacked".
    n_folds : int
        Number of walk-forward CV folds.
    min_train : int
        Minimum training rows for first CV fold.
    horizon : str
        Prediction horizon (default "1w").

    Returns
    -------
    dict with all experiment results.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT: {} {} — feature_set={}", symbol, "1W", feature_set)
    logger.info("=" * 70)

    # 1) Load data
    table = load_weekly_csv(csv_path, symbol)
    n_bars = table.num_rows
    prices = np.array(table.column("close").to_pylist(), dtype=np.float64)

    # 2) Extract features
    fm, aug_table = extract_features(table, feature_set=feature_set)
    n_features = fm.n_rows

    # 3) Build labels
    y_dir, y_mag, y_vol = build_labels(table, horizon=horizon)

    # 4) Align features and labels
    # Features may have fewer rows than labels due to warmup (e.g., EMA-200).
    # Labels may have fewer rows than features due to horizon look-ahead.
    # Use the minimum length.
    use_len = min(n_features, len(y_dir), len(y_mag), len(y_vol))
    if use_len < 50:
        raise ValueError(
            f"Only {use_len} aligned samples after warmup/look-ahead; "
            f"need at least 50 (features={n_features}, dir_labels={len(y_dir)})"
        )

    # Trim all arrays to the same length
    fm_aligned = FeatureMatrix(
        values=fm.values[:use_len],
        feature_names=fm.feature_names,
    )
    y_dir_aligned = y_dir[:use_len]
    y_mag_aligned = y_mag[:use_len]
    y_vol_aligned = y_vol[:use_len]

    logger.info("  Aligned: {} samples (features={} x {}, dir={}, mag={}, vol={})",
                use_len, fm_aligned.n_rows, len(fm_aligned.feature_names),
                len(y_dir_aligned), len(y_mag_aligned), len(y_vol_aligned))

    # 5) Walk-forward CV
    # Pass full prices array — evaluate_walk_forward computes log_returns and slices to match
    results = evaluate_walk_forward(
        fm_aligned, y_dir_aligned, y_mag_aligned, y_vol_aligned, prices,
        n_folds=n_folds,
        min_train=min_train,
        model_types=model_types,
    )

    # 6) Simple train/test split for final metrics
    logger.info("-" * 50)
    logger.info("FINAL TRAIN/TEST SPLIT (80/20)")
    logger.info("-" * 50)

    split_idx = int(use_len * 0.8)
    fm_train = FeatureMatrix(
        values=fm_aligned.values[:split_idx],
        feature_names=fm_aligned.feature_names,
    )
    fm_test = FeatureMatrix(
        values=fm_aligned.values[split_idx:],
        feature_names=fm_aligned.feature_names,
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(fm_train.values)
    test_scaled = scaler.transform(fm_test.values)

    fm_train_s = FeatureMatrix(values=train_scaled, feature_names=fm_train.feature_names)
    fm_test_s = FeatureMatrix(values=test_scaled, feature_names=fm_test.feature_names)

    y_dir_train = y_dir_aligned[:split_idx]
    y_dir_test = y_dir_aligned[split_idx:]
    y_mag_train = y_mag_aligned[:split_idx]
    y_mag_test = y_mag_aligned[split_idx:]
    y_vol_train = y_vol_aligned[:split_idx]
    y_vol_test = y_vol_aligned[split_idx:]

    final_results: dict[str, dict[str, Any]] = {}

    for model_type in model_types:
        try:
            if model_type == "lr":
                model = MultiHeadModel(MultiHeadConfig(n_estimators=500))
            elif model_type == "tree":
                model = TreeMultiHeadModel(TreeMultiHeadConfig(n_estimators=300))
            elif model_type == "stacked":
                model = StackedMultiHeadModel(StackedMultiHeadConfig(
                    n_estimators=300,
                    confidence_threshold=0.55,
                ))
            else:
                continue

            state = model.fit_multihead(fm_train_s, y_dir_train, y_mag_train, y_vol_train)
            preds = model.predict_multihead(state, fm_test_s)

            y_pred = preds["y_class"]
            y_proba = preds["y_proba"]

            acc = float(np.mean(y_pred == y_dir_test))
            non_flat = y_dir_test != 1
            dir_acc = float(np.mean(y_dir_test[non_flat] == y_pred[non_flat])) if non_flat.sum() > 0 else acc

            # Confidence-gated accuracy at p75
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                max_proba = np.max(y_proba, axis=1)
                p75 = float(np.percentile(max_proba, 75))
                confident = max_proba >= p75
                if confident.sum() > 0:
                    conf_acc = float(np.mean(y_dir_test[confident] == y_pred[confident]))
                    conf_coverage = float(confident.mean())
                else:
                    conf_acc = acc
                    conf_coverage = 0.0
            else:
                conf_acc = acc
                conf_coverage = 1.0

            final_results[model_type] = {
                "accuracy": acc,
                "direction_accuracy": dir_acc,
                "confident_accuracy": conf_acc,
                "confident_coverage": conf_coverage,
                "n_train": split_idx,
                "n_test": len(y_dir_test),
                "class_distribution": {
                    "down": int((y_dir_test == 0).sum()),
                    "flat": int((y_dir_test == 1).sum()),
                    "up": int((y_dir_test == 2).sum()),
                },
            }

            logger.info("  {} — acc={:.3f} dir_acc={:.3f} conf_acc(p75)={:.3f} cov={:.1%}",
                        model_type.upper(), acc, dir_acc, conf_acc, conf_coverage)

            # Feature importance for tree model
            if model_type in ("tree", "stacked"):
                try:
                    inner = state.get("tree_state", state)
                    head = inner.get("direction_head", None)
                    if head is not None and hasattr(head, "feature_importances_"):
                        importances = head.feature_importances_
                        top_idx = np.argsort(importances)[::-1][:15]
                        logger.info("  Top 15 features ({}):", model_type)
                        for rank, idx in enumerate(top_idx):
                            fname = fm_aligned.feature_names[idx]
                            logger.info("    {:2d}. {} = {:.4f}", rank + 1, fname, importances[idx])
                except Exception:
                    pass  # Feature importance is optional

        except Exception as exc:
            logger.error("  {} FAILED: {}", model_type, exc)
            final_results[model_type] = {"error": str(exc)}

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY: {} {} ({} features, {} samples)",
                symbol, "1W", len(fm_aligned.feature_names), use_len)
    logger.info("=" * 70)
    logger.info("{:<10} {:>10} {:>10} {:>10} {:>10}",
                "Model", "Accuracy", "Dir Acc", "Conf Acc", "Coverage")
    logger.info("-" * 50)
    for model_type, res in final_results.items():
        if "error" not in res:
            logger.info("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.1%}",
                        model_type.upper(),
                        res["accuracy"],
                        res["direction_accuracy"],
                        res["confident_accuracy"],
                        res["confident_coverage"])
        else:
            logger.info("{:<10} ERROR: {}", model_type, res["error"])

    # Walk-forward results summary
    if results:
        logger.info("")
        logger.info("WALK-FORWARD CV RESULTS ({} folds, min_train={}):", n_folds, min_train)
        logger.info("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                    "Model", "Avg Acc", "Avg Dir", "Avg CAS", "Avg Brier", "Std Acc")
        logger.info("-" * 60)
        for model_type, res in results.items():
            if "error" not in res:
                logger.info("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.3f}",
                            model_type.upper(),
                            res["mean_accuracy"],
                            res["mean_direction_accuracy"],
                            res["mean_cas"],
                            res["mean_brier"] if not math.isnan(res["mean_brier"]) else float("nan"),
                            res["std_accuracy"])
            else:
                logger.info("{:<10} ERROR: {}", model_type, res["error"])

    return {
        "symbol": symbol,
        "timeframe": "1W",
        "n_bars": n_bars,
        "n_features": len(fm_aligned.feature_names),
        "n_samples": use_len,
        "feature_set": feature_set,
        "final_results": final_results,
        "walk_forward_results": results,
        "feature_names": list(fm_aligned.feature_names),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly prediction experiment for BTC/ETH")
    parser.add_argument("--asset", choices=["BTC", "ETH", "both"], default="both",
                        help="Which asset to run (default: both)")
    parser.add_argument("--feature-set", choices=["all", "default"], default="all",
                        help="Feature set: 'all' (ALL_FEATURES) or 'default' (baseline 12)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of walk-forward CV folds (default: 5)")
    parser.add_argument("--min-train", type=int, default=100,
                        help="Minimum training rows for first CV fold (default: 100)")
    args = parser.parse_args()

    model_types = ("lr", "tree", "stacked")

    if args.asset == "both":
        assets = ["BTC", "ETH"]
    else:
        assets = [args.asset]

    all_results: dict[str, Any] = {}

    for asset in assets:
        csv_path = REPO_ROOT / f"{asset}_1W_tradingview_coinmarketcap.csv"
        if not csv_path.exists():
            logger.error("CSV not found: {}", csv_path)
            continue

        result = run_experiment(
            csv_path=csv_path,
            symbol=asset,
            feature_set=args.feature_set,
            model_types=model_types,
            n_folds=args.n_folds,
            min_train=args.min_train,
        )
        all_results[asset] = result

    # Print comparison
    if len(all_results) == 2:
        logger.info("")
        logger.info("=" * 70)
        logger.info("CROSS-ASSET COMPARISON")
        logger.info("=" * 70)
        for asset in ("BTC", "ETH"):
            if asset in all_results:
                res = all_results[asset]
                for model_type in model_types:
                    if model_type in res.get("final_results", {}):
                        fr = res["final_results"][model_type]
                        if "error" not in fr:
                            logger.info("{} {} {}: acc={:.3f} dir_acc={:.3f}",
                                        asset, "1W", model_type.upper(),
                                        fr["accuracy"], fr["direction_accuracy"])

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()