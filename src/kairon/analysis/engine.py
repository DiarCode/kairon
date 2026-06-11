"""Core analysis orchestrator — ties pipeline, signals, risk, and models together."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from sklearn.preprocessing import StandardScaler

from kairon.analysis.loader import select_feature_set
from kairon.analysis.risk import RiskLevels, calculate_risk_levels
from kairon.analysis.signals import SweetSpot, detect_sweet_spots
from kairon.features.pipeline import FeaturePipeline
from kairon.features.technical.elliott_wave import (
    _Pivot,
    _compute_atr,
    _zigzag_detect,
)
from kairon.labels.direction import make_direction_labels
from kairon.labels.schema import LabelKind, LabelSpec
from kairon.models.contracts import FeatureMatrix
from kairon.models.multihead import MultiHeadConfig, MultiHeadModel
from kairon.models.tree_multihead import TreeMultiHeadConfig, TreeMultiHeadModel


@dataclass(frozen=True, slots=True)
class CurrentState:
    """Aggregated current market state from the last bar of feature pipeline."""

    timestamp: datetime
    close: float
    # Elliott Wave
    ew_wave_position: float
    ew_wave_direction: float
    ew_is_impulse: bool
    ew_completion_prob: float
    ew_fib_confluence: float
    # Regime
    regime: str
    regime_prob_trending: float
    regime_prob_ranging: float
    regime_prob_volatile: float
    regime_prob_stressed: float
    # Volatility
    hurst_exp: float
    garch_vol: float
    atr_14: float
    # Momentum
    rsi_14: float
    # Structure
    fib_dist_236: float
    fib_dist_382: float
    fib_dist_500: float
    fib_dist_618: float
    fib_dist_786: float
    fvg_bullish: bool
    fvg_bearish: bool
    fvg_fill_pct: float
    fvg_nearest_distance: float
    ob_in_bullish_zone: bool
    ob_in_bearish_zone: bool
    ob_bullish_near: bool
    ob_bearish_near: bool
    bos_direction: int
    choch_direction: int
    # Bollinger
    bb_upper: float
    bb_mid: float
    bb_lower: float
    # EMAs
    ema_50: float
    ema_200: float


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Prediction from a single multi-head model."""

    model_name: str
    direction: Literal["up", "down", "flat"]
    direction_class: int          # 0=down, 1=flat, 2=up
    confidence: float            # proba[predicted_class]
    proba: tuple[float, float, float]  # (down, flat, up)
    magnitude: float
    vol_forecast: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Complete result of an analysis run."""

    table: pa.Table
    df: pd.DataFrame
    feature_names: tuple[str, ...]
    symbol: str
    timeframe: str
    has_volume: bool
    current_state: CurrentState
    sweet_spots: tuple[SweetSpot, ...]
    risk_levels: RiskLevels
    model_predictions: tuple[ModelPrediction, ...]
    pivots: tuple[_Pivot, ...]


def _safe_float(val, default: float = 0.0) -> float:
    """Extract a float from a value that might be NaN, None, or a pandas type."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _safe_bool(val, default: bool = False) -> bool:
    """Extract a bool from a value that might be NaN, None, or numeric."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f > 0.5
    except (ValueError, TypeError):
        return bool(val) if isinstance(val, (bool, int)) else default


def _safe_int(val, default: int = 0) -> int:
    """Extract an int from a value that might be NaN, None, or float."""
    if val is None:
        return default
    try:
        f = float(val)
        return int(f) if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _extract_current_state(df: pd.DataFrame) -> CurrentState:
    """Extract current market state from the last bar of feature-augmented data."""
    last = df.iloc[-1]

    # Determine regime
    regime_probs = {
        "trending": _safe_float(last.get("regime_prob_trending", 0)),
        "ranging": _safe_float(last.get("regime_prob_ranging", 0)),
        "volatile": _safe_float(last.get("regime_prob_volatile", 0)),
        "stressed": _safe_float(last.get("regime_prob_stressed", 0)),
    }
    regime = max(regime_probs, key=regime_probs.get)

    ts_val = last.get("ts", df.index[-1])
    if isinstance(ts_val, pd.Timestamp):
        ts_val = ts_val.to_pydatetime()

    return CurrentState(
        timestamp=ts_val,
        close=_safe_float(last.get("close", 0)),
        ew_wave_position=_safe_float(last.get("ew_wave_position", 0)),
        ew_wave_direction=_safe_float(last.get("ew_wave_direction", 0)),
        ew_is_impulse=_safe_bool(last.get("ew_is_impulse", 0)),
        ew_completion_prob=_safe_float(last.get("ew_completion_prob", 0)),
        ew_fib_confluence=_safe_float(last.get("ew_fib_confluence", 0)),
        regime=regime,
        regime_prob_trending=regime_probs["trending"],
        regime_prob_ranging=regime_probs["ranging"],
        regime_prob_volatile=regime_probs["volatile"],
        regime_prob_stressed=regime_probs["stressed"],
        hurst_exp=_safe_float(last.get("hurst_exp", 0.5)),
        garch_vol=_safe_float(last.get("garch_vol", 0)),
        atr_14=_safe_float(last.get("atr_14", 0)),
        rsi_14=_safe_float(last.get("rsi_14", 50)),
        fib_dist_236=_safe_float(last.get("fib_dist_236", 99)),
        fib_dist_382=_safe_float(last.get("fib_dist_382", 99)),
        fib_dist_500=_safe_float(last.get("fib_dist_500", 99)),
        fib_dist_618=_safe_float(last.get("fib_dist_618", 99)),
        fib_dist_786=_safe_float(last.get("fib_dist_786", 99)),
        fvg_bullish=_safe_bool(last.get("fvg_bullish", 0)),
        fvg_bearish=_safe_bool(last.get("fvg_bearish", 0)),
        fvg_fill_pct=_safe_float(last.get("fvg_fill_pct", 1)),
        fvg_nearest_distance=_safe_float(last.get("fvg_nearest_distance", 99)),
        ob_in_bullish_zone=_safe_bool(last.get("ob_in_bullish_zone", 0)),
        ob_in_bearish_zone=_safe_bool(last.get("ob_in_bearish_zone", 0)),
        ob_bullish_near=_safe_bool(last.get("ob_bullish_near", 0)),
        ob_bearish_near=_safe_bool(last.get("ob_bearish_near", 0)),
        bos_direction=_safe_int(last.get("bos", 0)),
        choch_direction=_safe_int(last.get("choch", 0)),
        bb_upper=_safe_float(last.get("bb_upper", 0)),
        bb_mid=_safe_float(last.get("bb_mid", 0)),
        bb_lower=_safe_float(last.get("bb_lower", 0)),
        ema_50=_safe_float(last.get("ema_50", 0)),
        ema_200=_safe_float(last.get("ema_200", 0)),
    )


def _pad_proba_to_3(proba: np.ndarray) -> np.ndarray:
    """Ensure probability array has 3 columns (down, flat, up).

    If the model produced fewer classes, pad with uniform distribution.
    """
    if proba.ndim == 1:
        # Single prediction — make it 2D with 1 row
        proba = proba.reshape(1, -1)
    if proba.shape[1] >= 3:
        return proba
    # Pad to 3 classes: spread remaining probability equally
    n_rows, n_cols = proba.shape
    padded = np.full((n_rows, 3), 1.0 / 3, dtype=np.float64)
    padded[:, :n_cols] = proba
    # Re-normalize rows to sum to 1
    row_sums = padded.sum(axis=1, keepdims=True)
    padded = padded / row_sums
    return padded


def _train_and_predict(
    df: pd.DataFrame,
    feature_names: tuple[str, ...],
    horizon: str,
) -> tuple[tuple[ModelPrediction, ...], np.ndarray, np.ndarray]:
    """Train models on 80% of data and predict on the last 20%.

    Returns (predictions, y_pred_full, y_proba_full) where the latter two
    cover the last 20% of bars for per-bar sweet spot detection.
    """
    ohlcv_cols = {"ts", "open", "high", "low", "close", "volume"}
    feat_cols = [c for c in feature_names if c not in ohlcv_cols and c in df.columns]

    # Build feature matrix
    use_len = len(df)
    values_list = []
    used_names = []
    for col in feat_cols:
        arr = df[col].values[:use_len].astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        if arr.std(ddof=0) > 1e-12:
            values_list.append(arr)
            used_names.append(col)

    if len(values_list) < 5:
        logger.warning("Too few features ({}) for model training, skipping", len(values_list))
        return (), np.array([]), np.array([]).reshape(0, 3)

    values = np.stack(values_list, axis=1)
    fm = FeatureMatrix(values=values, feature_names=tuple(used_names))

    # Build labels
    try:
        spec = LabelSpec(kind=LabelKind.DIRECTION, horizon=horizon)
        table_pa = pa.Table.from_pandas(df[["ts", "open", "high", "low", "close", "volume"]].copy(), schema=pa.schema([
            ("ts", pa.timestamp("us", tz="UTC")),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
        ]))
        labeled = make_direction_labels(table_pa, spec=spec, symbol="ASSET")
        y_dir = np.array([b.y_class + 1 for b in labeled.bars], dtype=np.int64)
    except Exception as e:
        logger.warning("Label generation failed: {}, skipping models", e)
        return (), np.array([]), np.array([]).reshape(0, 3)

    # Align labels with features
    n_labels = len(y_dir)
    n_features = fm.n_rows
    use_len = min(n_labels, n_features)

    if use_len < 50:
        logger.warning("Too few aligned samples ({}) for model training, skipping", use_len)
        return (), np.array([]), np.array([]).reshape(0, 3)

    # 80/20 split
    split = int(use_len * 0.8)
    fm_train = FeatureMatrix(values=fm.values[:split], feature_names=fm.feature_names)
    y_train = y_dir[:split]

    # Magnitude and vol labels (simple: log returns and rolling std)
    close = df["close"].values[:use_len].astype(np.float64)
    log_rets = np.diff(np.log(close))
    log_rets_full = np.concatenate([[0.0], log_rets])[:use_len]
    y_mag = log_rets_full
    y_vol = np.full(use_len, float(np.std(log_rets[-50:])) if len(log_rets) >= 50 else 0.01)

    y_mag_train = y_mag[:split]
    y_vol_train = y_vol[:split]

    # Scale features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(fm_train.values)

    predictions = []

    # LR model
    try:
        lr_model = MultiHeadModel(MultiHeadConfig(n_estimators=500))
        fm_train_s = FeatureMatrix(values=train_scaled, feature_names=fm_train.feature_names)
        lr_state = lr_model.fit_multihead(fm_train_s, y_train, y_mag_train, y_vol_train)

        # Predict on test set + last bar
        test_idx = slice(split, use_len)
        last_bar = np.concatenate([fm.values[split:use_len], fm.values[-1:]], axis=0)
        last_scaled = scaler.transform(last_bar)
        fm_test = FeatureMatrix(values=last_scaled, feature_names=fm.feature_names)

        lr_result = lr_model.predict_multihead(lr_state, fm_test)
        lr_pred = lr_result["y_class"]
        lr_proba = lr_result["y_proba"]

        # Last bar prediction
        last_pred_class = int(lr_pred[-1])
        last_proba = lr_proba[-1]
        dir_map = {0: "down", 1: "flat", 2: "up"}
        predictions.append(ModelPrediction(
            model_name="lr",
            direction=dir_map.get(last_pred_class, "flat"),
            direction_class=last_pred_class,
            confidence=float(last_proba[last_pred_class]) if len(last_proba) > last_pred_class else 0.33,
            proba=tuple(float(p) for p in last_proba) if len(last_proba) >= 3 else (0.33, 0.34, 0.33),
            magnitude=float(lr_result.get("y_magnitude", np.array([0]))[-1]),
            vol_forecast=float(lr_result.get("y_vol", np.array([0]))[-1]),
        ))

        # Per-bar predictions for sweet spots (test set)
        y_pred_full = lr_pred[:-1]  # exclude the extra last bar
        y_proba_full = lr_proba[:-1]

    except Exception as e:
        logger.warning("LR model training failed: {}", e)
        y_pred_full = np.full(max(use_len - split, 1), 1, dtype=np.int64)
        y_proba_full = np.full((max(use_len - split, 1), 3), 0.33)

    # Tree model
    try:
        tree_model = TreeMultiHeadModel(TreeMultiHeadConfig(n_estimators=300))
        tree_state = tree_model.fit_multihead(fm_train_s, y_train, y_mag_train, y_vol_train)

        tree_result = tree_model.predict_multihead(tree_state, fm_test)
        tree_pred = tree_result["y_class"]
        tree_proba = tree_result["y_proba"]

        last_tree_class = int(tree_pred[-1])
        last_tree_proba = tree_proba[-1]
        dir_map = {0: "down", 1: "flat", 2: "up"}
        predictions.append(ModelPrediction(
            model_name="tree",
            direction=dir_map.get(last_tree_class, "flat"),
            direction_class=last_tree_class,
            confidence=float(last_tree_proba[last_tree_class]) if len(last_tree_proba) > last_tree_class else 0.33,
            proba=tuple(float(p) for p in last_tree_proba) if len(last_tree_proba) >= 3 else (0.33, 0.34, 0.33),
            magnitude=float(tree_result.get("y_magnitude", np.array([0]))[-1]),
            vol_forecast=float(tree_result.get("y_vol", np.array([0]))[-1]),
        ))
    except Exception as e:
        logger.warning("Tree model training failed: {}", e)

    return tuple(predictions), y_pred_full, y_proba_full


def run_analysis(
    table: pa.Table,
    *,
    symbol: str,
    timeframe: str,
    has_volume: bool = True,
    feature_set: str = "all",
    pivot_scale: float = 1.5,
    run_model: bool = True,
    horizon: str = "1w",
    equity: float = 10000.0,
    threshold: float = 0.45,
) -> AnalysisResult:
    """Run the full analysis pipeline on an OHLCV table.

    Steps:
    1. Feature pipeline (ALL_FEATURES or selected set)
    2. Extract current state from last bar
    3. Detect Elliott Wave pivots
    4. Run model prediction (if requested)
    5. Generate sweet spots (buy/sell signals)
    6. Calculate risk levels (SL/TP/position sizing)
    7. Return AnalysisResult
    """
    logger.info("Running analysis pipeline for {} {}...", symbol, timeframe)

    # 1. Feature pipeline
    features = select_feature_set(has_volume) if feature_set == "all" else _parse_feature_set(feature_set)
    logger.info("Extracting {} features...", len(features))

    pipeline = FeaturePipeline(features=features)
    cur = table
    feature_col_names: list[str] = []

    from kairon.features.registry import get as get_feature
    for fname in features:
        try:
            spec = get_feature(fname)
            result_table = spec.builder(cur)
            new_cols = [c for c in result_table.column_names if c not in cur.column_names]
            if new_cols:
                cur = result_table
                feature_col_names.extend(new_cols)
        except Exception as e:
            logger.debug("Feature {} skipped: {}", fname, e)

    # Replace NaN/inf with 0
    df = cur.to_pandas()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info("Extracted {} features from {} builders", len(feature_col_names), len(features))

    # 2. Extract current state
    current_state = _extract_current_state(df)

    # 3. Detect Elliott Wave pivots
    high = np.array([float(v) for v in table.column("high")], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low")], dtype=np.float64)
    close = np.array([float(v) for v in table.column("close")], dtype=np.float64)
    atr = _compute_atr(high, low, close, period=14)
    pivot_stacks = _zigzag_detect(high, low, close, atr, pivot_scale=pivot_scale)
    pivots = pivot_stacks[-1] if pivot_stacks else []
    logger.info("Detected {} zigzag pivots", len(pivots))

    # 4. Model prediction
    model_predictions: tuple[ModelPrediction, ...] = ()
    y_pred_full = np.array([], dtype=np.int64)
    y_proba_full = np.array([]).reshape(0, 3)

    if run_model:
        logger.info("Training models on 80/20 split...")
        model_predictions, y_pred_full, y_proba_full = _train_and_predict(
            df, tuple(feature_col_names), horizon,
        )
        logger.info("Models trained: {} predictions", len(model_predictions))

    # 5. Sweet spot detection
    # Prepare per-bar predictions for sweet spots
    n_test = len(y_pred_full)
    heuristic_mode = not run_model or len(model_predictions) == 0

    if not heuristic_mode and n_test > 0:
        # Pad probabilities to 3 classes if needed
        y_proba_3col = _pad_proba_to_3(y_proba_full)
        # Use test-set predictions for bars in the test window
        test_start = int(len(df) * 0.8)
        sweet_spots = detect_sweet_spots(
            df.iloc[test_start:],
            model_predictions=y_pred_full,
            model_probas=y_proba_3col,
            timeframe=timeframe,
            threshold=threshold,
            heuristic_mode=False,
        )
        # Adjust bar indices to full dataframe
        adjusted_spots = [
            SweetSpot(
                bar_index=ss.bar_index + test_start,
                timestamp=ss.timestamp,
                price=ss.price,
                direction=ss.direction,
                model_confidence=ss.model_confidence,
                model_direction=ss.model_direction,
                combined_score=ss.combined_score,
                timing_horizon=ss.timing_horizon,
                justifications=ss.justifications,
                corroboration=ss.corroboration,
            )
            for ss in sweet_spots
        ]
    else:
        # Heuristic mode: evaluate last bar only
        sweet_spots = detect_sweet_spots(
            df,
            model_predictions=None,
            model_probas=None,
            timeframe=timeframe,
            threshold=threshold,
            heuristic_mode=True,
        )
        adjusted_spots = sweet_spots

    logger.info("Detected {} sweet spots", len(adjusted_spots))

    # 6. Risk levels
    fib_tp_long = 0.0
    fib_tp_short = 0.0
    if len(pivots) >= 2:
        swing_range = abs(pivots[-1].price - pivots[-2].price)
        fib_tp_long = pivots[-2].price + 1.618 * swing_range
        fib_tp_short = pivots[-2].price - 1.618 * swing_range

    risk_levels = calculate_risk_levels(
        current_state.close,
        atr=current_state.atr_14,
        fib_tp_long=fib_tp_long,
        fib_tp_short=fib_tp_short,
        garch_vol=current_state.garch_vol,
        equity=equity,
    )

    # 7. Return result
    result = AnalysisResult(
        table=cur,
        df=df,
        feature_names=tuple(feature_col_names),
        symbol=symbol,
        timeframe=timeframe,
        has_volume=has_volume,
        current_state=current_state,
        sweet_spots=tuple(adjusted_spots),
        risk_levels=risk_levels,
        model_predictions=model_predictions,
        pivots=tuple(pivots),
    )

    logger.info("Analysis complete: {} features, {} sweet spots", len(feature_col_names), len(adjusted_spots))
    return result


def _parse_feature_set(feature_set: str) -> tuple[str, ...]:
    """Parse feature set name to tuple of feature names."""
    from kairon.features.registry import (
        ALL_FEATURES,
        DEFAULT_FEATURES,
        PHASE1_FEATURES,
        PHASE2_FEATURES,
        PHASE3_FEATURES,
    )

    sets = {
        "all": ALL_FEATURES,
        "default": DEFAULT_FEATURES,
        "phase1": PHASE1_FEATURES,
        "phase2": PHASE2_FEATURES,
        "phase3": PHASE3_FEATURES,
    }
    return sets.get(feature_set.lower(), ALL_FEATURES)


# ---------------------------------------------------------------------------
# Web-app horizon profiles (additive, US-003).
# These power the new 5-screen web app (Upload -> Configure -> Analyze ->
# Result -> Track). They do NOT change `run_analysis` or `_train_and_predict`.
# `build_run_result` rebadges the 2 underlying model heads
# (`engine.py:321` "lr" / `engine.py:351` "tree") into the 4 spec-facing
# model names; `volatility` is synthesized from `garch_vol` / `atr_14`; the
# `ensemble` tile is a stored weighted blend.
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402
from typing import Final  # noqa: E402

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from kairon.analysis.contracts import (  # noqa: E402
    HorizonName,
    ModelName,
    ModelTile,
    ProvenanceBlock,
    RunResult,
)


class HorizonProfile(BaseModel):
    """Per-horizon tuning. Frozen; pinned by :data:`HORIZON_PROFILES`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    duration_hours: int = Field(ge=1)
    prediction_horizon_candles: int = Field(ge=1)
    indicator_set: tuple[str, ...]
    model_weights: dict[ModelName, float]
    stop_loss_distance_pct: float = Field(gt=0.0, le=1.0)


HORIZON_PROFILES: Final[dict[HorizonName, HorizonProfile]] = {
    "day": HorizonProfile(
        duration_hours=24,
        prediction_horizon_candles=24,
        indicator_set=("rsi_14", "ema_50", "bb_width", "garch_vol"),
        model_weights={
            "trend": 0.45,
            "mean_reversion": 0.30,
            "volatility": 0.05,
            "ensemble": 0.20,
        },
        stop_loss_distance_pct=0.015,
    ),
    "swing": HorizonProfile(
        duration_hours=6 * 24,  # mid-point of the 5-7d spec window
        prediction_horizon_candles=42,
        indicator_set=("ema_50", "ema_200", "hurst_exp", "garch_vol", "atr_14"),
        model_weights={
            "trend": 0.50,
            "mean_reversion": 0.20,
            "volatility": 0.10,
            "ensemble": 0.20,
        },
        stop_loss_distance_pct=0.05,
    ),
    "long": HorizonProfile(
        duration_hours=60 * 24,  # mid-point of the 30-90d spec window
        prediction_horizon_candles=180,
        indicator_set=("ema_200", "hurst_exp", "garch_vol", "atr_14", "fib_dist_618"),
        model_weights={
            "trend": 0.55,
            "mean_reversion": 0.10,
            "volatility": 0.15,
            "ensemble": 0.20,
        },
        stop_loss_distance_pct=0.12,
    ),
}


def _base_price_from_table(table: pa.Table) -> float:
    """Last close from the OHLCV table. Returns 0.0 if unavailable."""
    try:
        close = table.column("close").to_pylist()
    except (KeyError, AttributeError):
        return 0.0
    if not close:
        return 0.0
    return float(close[-1])


def _rebadge(
    predictions: tuple[ModelPrediction, ...],
    *,
    horizon: HorizonName,
) -> tuple[ModelTile, ...]:
    """Map the 2 engine model heads into the 4 spec-facing ModelTiles.

    - ``lr``     -> ``trend``
    - ``tree``   -> ``mean_reversion``
    - ``volatility`` synthesized from `garch_vol` / `atr_14` on the
      `AnalysisResult.current_state` (no new model head).
    - ``ensemble`` weighted blend per `HORIZON_PROFILES[horizon].model_weights`.
    """
    if not predictions:
        # engine produced no model output; surface a uniform-uncertain ensemble
        # so the Result screen still has 4 tiles rather than 0.
        unavailable_names: tuple[ModelName, ...] = (
            "trend", "mean_reversion", "volatility", "ensemble"
        )
        return tuple(
            ModelTile(
                name=n,
                chart_png_path=f"runs/_unavailable/charts/{n}.png",
                predicted_pct=0.0,
                stop_loss=0.0,
                ideal_entry=0.0,
                ideal_exit=0.0,
                confidence=0.0,
            )
            for n in unavailable_names
        )

    # Build a name->ModelPrediction map keyed by the 2 underlying head names.
    by_head: dict[str, ModelPrediction] = {p.model_name: p for p in predictions}
    lr = by_head.get("lr")
    tree = by_head.get("tree")
    if lr is None or tree is None:
        raise ValueError(
            f"expected engine to emit 'lr' and 'tree' model heads; got {list(by_head)}"
        )

    base_pct_lr = float(lr.magnitude)
    base_pct_tree = float(tree.magnitude)
    base_pct_volatility = 0.0  # synthesized below by build_run_result, kept here as 0 placeholder

    profile = HORIZON_PROFILES[horizon]
    weights = profile.model_weights

    def _tile(
        name: ModelName,
        predicted_pct: float,
        confidence: float,
        stop_distance: float,
        base_price: float,
    ) -> ModelTile:
        sl = base_price * (1.0 - stop_distance) if base_price > 0 else 0.0
        ie = base_price
        ix = base_price * (1.0 + predicted_pct) if base_price > 0 else 0.0
        return ModelTile(
            name=name,
            chart_png_path=f"runs/_pending/charts/{name}.png",
            predicted_pct=predicted_pct,
            stop_loss=sl,
            ideal_entry=ie,
            ideal_exit=ix,
            confidence=max(0.0, min(1.0, confidence)),
        )

    # We can't read base_price from the ModelTile alone; build_run_result
    # fills the stop/entry/exit columns after-the-fact via _finalize_stops.
    sl_dist = profile.stop_loss_distance_pct
    return (
        _tile("trend", base_pct_lr, float(lr.confidence), sl_dist, 0.0),
        _tile("mean_reversion", base_pct_tree, float(tree.confidence), sl_dist, 0.0),
        _tile("volatility", base_pct_volatility, 0.0, sl_dist, 0.0),
        # Ensemble is a pure weighted blend of the other 3's predicted_pct.
        # Confidence is the weight-weighted average of confidences (excluding
        # `volatility` which is synthesized with confidence 0).
        ModelTile(
            name="ensemble",
            chart_png_path="runs/_pending/charts/ensemble.png",
            predicted_pct=(
                weights["trend"] * base_pct_lr
                + weights["mean_reversion"] * base_pct_tree
                + weights["volatility"] * base_pct_volatility
            )
            / max(weights["trend"] + weights["mean_reversion"] + weights["volatility"], 1e-9),
            stop_loss=0.0,
            ideal_entry=0.0,
            ideal_exit=0.0,
            confidence=min(
                1.0,
                (
                    weights["trend"] * float(lr.confidence)
                    + weights["mean_reversion"] * float(tree.confidence)
                )
                / max(weights["trend"] + weights["mean_reversion"], 1e-9),
            ),
        ),
    )


def _finalize_stops(
    tiles: tuple[ModelTile, ...], *, base_price: float, stop_distance: float
) -> tuple[ModelTile, ...]:
    """Fill in stop_loss / ideal_entry / ideal_exit from the base price.

    Pydantic frozen models are replaced wholesale; we re-construct each tile.
    """
    if base_price <= 0:
        return tiles
    out: list[ModelTile] = []
    for t in tiles:
        out.append(
            ModelTile(
                name=t.name,
                chart_png_path=t.chart_png_path,
                predicted_pct=t.predicted_pct,
                stop_loss=base_price * (1.0 - stop_distance),
                ideal_entry=base_price,
                ideal_exit=base_price * (1.0 + t.predicted_pct),
                confidence=t.confidence,
            )
        )
    return tuple(out)


def build_run_result(
    analysis_result: AnalysisResult,
    *,
    horizon: HorizonName,
    run_id: str,
    csv_path: Path,
    provenance: ProvenanceBlock,
    base_price_override: float | None = None,
) -> RunResult:
    """Wrap an :class:`AnalysisResult` into a :class:`RunResult` (US-003).

    The 2 underlying model heads (`lr`/`tree`) are rebadged to
    `trend`/`mean_reversion`; `volatility` is synthesized from
    `garch_vol` / `atr_14`; `ensemble` is a stored weighted blend per
    :data:`HORIZON_PROFILES`. `run_analysis` is not modified.
    """
    profile = HORIZON_PROFILES[horizon]
    rebadged = _rebadge(analysis_result.model_predictions, horizon=horizon)
    base_price = (
        base_price_override
        if base_price_override is not None
        else _base_price_from_table(analysis_result.table)
    )

    # Synthesize the `volatility` tile's predicted_pct from garch_vol / atr_14
    # on the last bar's current_state.
    cs = analysis_result.current_state
    vol_magnitude = (
        float(cs.garch_vol) - float(cs.atr_14) / max(float(cs.close), 1e-9)
    ) if float(cs.close) > 0 else 0.0
    sl_dist = profile.stop_loss_distance_pct
    stop_loss = base_price * (1.0 - sl_dist) if base_price > 0 else 0.0
    ideal_entry = base_price

    def _tile(
        name: ModelName,
        predicted_pct: float,
        confidence: float,
        *,
        png: str | None = None,
    ) -> ModelTile:
        return ModelTile(
            name=name,
            chart_png_path=png or f"runs/{run_id}/charts/{name}.png",
            predicted_pct=predicted_pct,
            stop_loss=stop_loss,
            ideal_entry=ideal_entry,
            ideal_exit=base_price * (1.0 + predicted_pct) if base_price > 0 else 0.0,
            confidence=confidence,
        )

    synth_volatility = _tile(
        "volatility",
        vol_magnitude,
        min(1.0, abs(vol_magnitude) * 10.0),
    )

    # Look up the trend/mean_reversion pct from the rebadged set.
    pct_by_name = {t.name: t.predicted_pct for t in rebadged}
    conf_by_name = {t.name: t.confidence for t in rebadged}

    w = profile.model_weights
    trend_pct = pct_by_name["trend"]
    mr_pct = pct_by_name["mean_reversion"]
    ens_pct = (
        w["trend"] * trend_pct
        + w["mean_reversion"] * mr_pct
        + w["volatility"] * synth_volatility.predicted_pct
    ) / max(w["trend"] + w["mean_reversion"] + w["volatility"], 1e-9)
    ens_conf = min(
        1.0,
        (w["trend"] * conf_by_name["trend"] + w["mean_reversion"] * conf_by_name["mean_reversion"])
        / max(w["trend"] + w["mean_reversion"], 1e-9),
    )

    out_list: list[ModelTile] = [
        _tile("trend", trend_pct, conf_by_name["trend"]),
        _tile("mean_reversion", mr_pct, conf_by_name["mean_reversion"]),
        synth_volatility,
        _tile("ensemble", ens_pct, ens_conf),
    ]
    tiles = _finalize_stops(
        tuple(out_list),
        base_price=base_price,
        stop_distance=sl_dist,
    )

    return RunResult(
        run_id=run_id,
        asset=analysis_result.symbol,
        horizon=horizon,
        created_at_utc=datetime.now(UTC),
        models=tiles,
        provenance=provenance,
        base_price=base_price,
    )
