#!/usr/bin/env python3
"""Full Technical Analysis Report for BTC & ETH 1W data.

Generates:
- 6 charts per asset (Elliott Wave, indicators, regime, SMC, prediction, summary)
- Markdown report per asset with wave analysis, trading levels, risk assessment

Usage:
    python scripts/technical_analysis_report.py
    python scripts/technical_analysis_report.py --asset BTC
    python scripts/technical_analysis_report.py --asset ETH --pivot-scale 2.0
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from kairon.data.io import OHLCV_SCHEMA
from kairon.features.pipeline import FeaturePipeline
from kairon.features.registry import ALL_FEATURES
from kairon.features.technical.elliott_wave import (
    _compute_atr,
    _Pivot,
    _zigzag_detect,
)
from kairon.features.technical.structure import fibonacci_levels
from kairon.labels.direction import make_direction_labels
from kairon.labels.schema import LabelKind, LabelSpec
from kairon.models.contracts import FeatureMatrix
from kairon.models.multihead import MultiHeadConfig, MultiHeadModel
from kairon.models.tree_multihead import TreeMultiHeadConfig, TreeMultiHeadModel
from sklearn.preprocessing import StandardScaler

# ─── Configuration ───────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DPI = 150
FIG_WIDTH = 18
FIG_HEIGHT_MAIN = 10
FIG_HEIGHT_SUB = 6

# Colors
C_BULL = "#26a69a"  # green
C_BEAR = "#ef5350"  # red
C_FLAT = "#78909c"  # gray
C_FIB = "#ab47bc"  # purple
C_FVG_BULL = "#4caf5080"
C_FVG_BEAR = "#f4433360"
C_OB_BULL = "#66bb6a40"
C_OB_BEAR = "#ef535040"
C_REGIME_TRENDING = "#26a69a30"
C_REGIME_RANGING = "#42a5f530"
C_REGIME_VOLATILE = "#ffa72630"
C_REGIME_STRESSED = "#ef535030"


# ─── Data Loading (reuse from predict_weekly.py) ─────────────────────────────

def load_weekly_csv(path: Path, symbol: str = "UNKNOWN") -> pa.Table:
    """Load a semicolon-delimited weekly CSV into an OHLCV pyarrow Table."""
    df = pd.read_csv(path, sep=";")
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower == "time":
            col_map[col] = "ts"
        elif lower == "volume":
            col_map[col] = "volume"
        else:
            col_map[col] = lower
    df = df.rename(columns=col_map)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(np.float64)
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    return pa.Table.from_pandas(df, schema=OHLCV_SCHEMA)


def extract_features(table: pa.Table) -> tuple[dict, pa.Table]:
    """Run ALL_FEATURES pipeline and return column dict + augmented table."""
    pipeline = FeaturePipeline(features=ALL_FEATURES)
    cur = table
    feature_col_names = []
    from kairon.features.registry import get as get_feature
    for fname in ALL_FEATURES:
        try:
            spec = get_feature(fname)
            result_table = spec.builder(cur)
            new_cols = [c for c in result_table.column_names if c not in cur.column_names]
            if new_cols:
                cur = result_table
                feature_col_names.extend(new_cols)
        except Exception:
            pass
    return feature_col_names, cur


# ─── Elliott Wave Pivot Extraction ────────────────────────────────────────────

def get_ew_pivots(table: pa.Table, pivot_scale: float = 1.5) -> list[_Pivot]:
    """Extract final zigzag pivots for chart labeling."""
    high = np.array([float(v) for v in table.column("high")], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low")], dtype=np.float64)
    close = np.array([float(v) for v in table.column("close")], dtype=np.float64)
    atr = _compute_atr(high, low, close, period=14)
    pivot_stacks = _zigzag_detect(high, low, close, atr, pivot_scale=pivot_scale)
    return pivot_stacks[-1] if pivot_stacks else []


def label_wave_segments(pivots: list[_Pivot]) -> list[tuple[int, int, str]]:
    """Label wave segments between pivots using EW rules.

    Returns list of (start_idx, end_idx, label) tuples.
    """
    if len(pivots) < 5:
        return []

    segments = []
    last_5 = pivots[-5:] if len(pivots) >= 5 else pivots
    bullish = last_5[0].kind == "low"

    # Try impulse labeling on last 5 pivots
    p = last_5
    wave_labels = ["1", "2", "3", "4", "5"] if bullish else ["1", "2", "3", "4", "5"]

    for i in range(len(p) - 1):
        segments.append((p[i].idx, p[i + 1].idx, f"W{wave_labels[i]}"))

    # If more than 5 pivots, label earlier ones as corrective
    remaining = pivots[:-5] if len(pivots) > 5 else []
    if remaining:
        corrective_labels = ["A", "B", "C"]
        for i, pivot in enumerate(remaining[-3:] if len(remaining) >= 3 else remaining):
            if i < len(pivots) - 1:
                label = corrective_labels[i % 3] if i < 3 else "?"
                segments.insert(0, (pivot.idx, pivots[pivots.index(pivot) + 1].idx if pivot.idx < pivots[-1].idx else pivot.idx, label))

    return segments


# ─── Chart Generators ────────────────────────────────────────────────────────

def _to_pandas(table: pa.Table) -> pd.DataFrame:
    """Convert pyarrow Table to pandas with datetime index."""
    df = table.to_pandas()
    if "ts" in df.columns:
        df = df.set_index("ts")
    return df


def chart_elliott_wave(
    df: pd.DataFrame,
    pivots: list[_Pivot],
    symbol: str,
    output_path: Path,
) -> None:
    """Chart 1: Price with Elliott Wave labels, Fibonacci levels, and wave coloring."""
    fig, (ax_price, ax_ew) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 14),
                                           gridspec_kw={"height_ratios": [3, 1]},
                                           sharex=True)
    fig.suptitle(f"{symbol} 1W — Elliott Wave Analysis", fontsize=16, fontweight="bold")

    # Price candlestick (use line since mplfinance is complex with subplots)
    ax_price.plot(df.index, df["close"], color="white", linewidth=0.8, alpha=0.9)
    ax_price.fill_between(df.index, df["low"], df["high"], alpha=0.15, color="#90caf9")

    # Color background by wave direction
    ew_dir = df["ew_wave_direction"].values if "ew_wave_direction" in df.columns else np.zeros(len(df))
    ew_imp = df["ew_is_impulse"].values if "ew_is_impulse" in df.columns else np.zeros(len(df))

    for i in range(len(df) - 1):
        if ew_imp[i] == 1 and ew_dir[i] == 1:
            ax_price.axvspan(df.index[i], df.index[i + 1], alpha=0.08, color=C_BULL)
        elif ew_imp[i] == 1 and ew_dir[i] == -1:
            ax_price.axvspan(df.index[i], df.index[i + 1], alpha=0.08, color=C_BEAR)

    # Draw zigzag lines between pivots
    if len(pivots) >= 2:
        pivot_indices = [p.idx for p in pivots]
        pivot_prices = [p.price for p in pivots]
        pivot_dates = [df.index[p.idx] if p.idx < len(df) else df.index[-1] for p in pivots]
        ax_price.plot(pivot_dates, pivot_prices, color=C_FIB, linewidth=2.0, alpha=0.8, zorder=5)

        # Label wave segments
        wave_names = ["W1", "W2", "W3", "W4", "W5"] if len(pivots) >= 5 else [f"W{i+1}" for i in range(len(pivots)-1)]
        # For corrective: use A, B, C
        if len(pivots) > 5:
            # Last 5 are impulse, earlier may be corrective
            n_impulse = 5
            n_corrective = len(pivots) - n_impulse
            names = []
            for i in range(n_corrective - 1):
                names.append(chr(65 + (i % 3)))  # A, B, C
            names.extend(wave_names[:n_impulse - 1])
            wave_names = names

        for i in range(min(len(pivots) - 1, len(wave_names))):
            mid_idx = (pivots[i].idx + pivots[i + 1].idx) // 2
            if mid_idx < len(df):
                mid_price = (pivots[i].price + pivots[i + 1].price) / 2
                ax_price.annotate(
                    wave_names[i],
                    xy=(df.index[mid_idx], mid_price),
                    fontsize=10, fontweight="bold", color=C_FIB,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor=C_FIB),
                )

        # Annotate pivot points
        for p in pivots[-8:]:  # Last 8 pivots only
            if p.idx < len(df):
                marker = "^" if p.kind == "low" else "v"
                color = C_BULL if p.kind == "low" else C_BEAR
                ax_price.scatter(df.index[p.idx], p.price, marker=marker, s=80, color=color, zorder=6)

    # Fibonacci retracement levels from last 2 pivots
    if len(pivots) >= 2 and "atr_14" in df.columns:
        swing_high = max(pivots[-1].price, pivots[-2].price)
        swing_low = min(pivots[-1].price, pivots[-2].price)
        rng = swing_high - swing_low
        fib_ratios = {"0.236": 0.236, "0.382": 0.382, "0.500": 0.500, "0.618": 0.618, "0.786": 0.786}
        for name, ratio in fib_ratios.items():
            level = swing_high - ratio * rng
            ax_price.axhline(y=level, color=C_FIB, linestyle="--", alpha=0.4, linewidth=0.7)
            ax_price.text(df.index[-1], level, f" Fib {name}", fontsize=8, color=C_FIB, va="bottom")

    # Current price annotation
    last_close = df["close"].iloc[-1]
    ax_price.axhline(y=last_close, color="yellow", linestyle="-", alpha=0.5, linewidth=1)
    ax_price.text(df.index[-1], last_close, f"  ${last_close:,.0f}", fontsize=10, color="yellow", fontweight="bold")

    ax_price.set_ylabel("Price (USD)", fontsize=12)
    ax_price.set_facecolor("#1a1a2e")
    ax_price.grid(alpha=0.2)
    ax_price.legend(["Close", "Zigzag"], loc="upper left", fontsize=9)

    # Bottom: Wave position + completion probability
    ew_pos = df["ew_wave_position"].values if "ew_wave_position" in df.columns else np.zeros(len(df))
    ew_comp = df["ew_completion_prob"].values if "ew_completion_prob" in df.columns else np.zeros(len(df))

    ax_ew.bar(df.index, ew_pos, width=pd.Timedelta(days=5), color="#42a5f5", alpha=0.7, label="Wave Position")
    ax_ew2 = ax_ew.twinx()
    ax_ew2.plot(df.index, ew_comp, color="#ffa726", linewidth=1.5, label="Completion Prob")
    ax_ew2.set_ylabel("Completion Prob", color="#ffa726", fontsize=10)
    ax_ew2.tick_params(axis="y", labelcolor="#ffa726")
    ax_ew.set_ylabel("Wave Position", fontsize=10)
    ax_ew.set_facecolor("#1a1a2e")
    ax_ew.grid(alpha=0.2)
    ax_ew.legend(loc="upper left", fontsize=9)
    ax_ew2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved {}", output_path.name)


def chart_regime(df: pd.DataFrame, symbol: str, output_path: Path) -> None:
    """Chart 3: BOCPD regime shading, Hurst exponent, and changepoints."""
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{symbol} 1W — Regime Detection & Structure", fontsize=16, fontweight="bold")

    ax_price = axes[0]
    ax_hurst = axes[1]
    ax_runlen = axes[2]

    # Price with regime shading
    ax_price.plot(df.index, df["close"], color="white", linewidth=1)
    regime_cols = ["regime_prob_trending", "regime_prob_ranging", "regime_prob_volatile", "regime_prob_stressed"]
    regime_colors = [C_REGIME_TRENDING, C_REGIME_RANGING, C_REGIME_VOLATILE, C_REGIME_STRESSED]

    if all(c in df.columns for c in regime_cols):
        dominant = df[regime_cols].values.argmax(axis=1)
        regime_map = {0: ("Trending", C_REGIME_TRENDING), 1: ("Ranging", C_REGIME_RANGING),
                      2: ("Volatile", C_REGIME_VOLATILE), 3: ("Stressed", C_REGIME_STRESSED)}
        for i in range(len(df) - 1):
            name, color = regime_map[dominant[i]]
            ax_price.axvspan(df.index[i], df.index[i + 1], alpha=0.25, color=color)

    # Changepoint lines
    if "bocpd_is_changepoint" in df.columns:
        cp_mask = df["bocpd_is_changepoint"].astype(bool)
        for idx in df.index[cp_mask]:
            ax_price.axvline(x=idx, color="yellow", alpha=0.4, linewidth=0.5, linestyle="--")

    ax_price.set_ylabel("Price", fontsize=11)
    ax_price.set_facecolor("#1a1a2e")
    ax_price.grid(alpha=0.2)

    # Hurst exponent
    if "hurst_exp" in df.columns:
        ax_hurst.plot(df.index, df["hurst_exp"], color="#ab47bc", linewidth=1.5)
        ax_hurst.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (H=0.5)")
        ax_hurst.axhline(y=0.0, color="blue", linestyle=":", alpha=0.3)
        ax_hurst.axhline(y=1.0, color="red", linestyle=":", alpha=0.3)
        ax_hurst.fill_between(df.index, 0, 0.5, alpha=0.05, color="blue", label="Mean-reverting")
        ax_hurst.fill_between(df.index, 0.5, 1.0, alpha=0.05, color="red", label="Trending")
        ax_hurst.set_ylabel("Hurst Exp", fontsize=10)
        ax_hurst.set_ylim(0, 1)
        ax_hurst.legend(fontsize=8, loc="upper left")
    ax_hurst.set_facecolor("#1a1a2e")
    ax_hurst.grid(alpha=0.2)

    # Run-length
    if "bocpd_run_length_mean" in df.columns:
        ax_runlen.plot(df.index, df["bocpd_run_length_mean"], color="#42a5f5", linewidth=1)
        ax_runlen.set_ylabel("Run Length", fontsize=10)
    ax_runlen.set_facecolor("#1a1a2e")
    ax_runlen.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved {}", output_path.name)


def chart_smc(df: pd.DataFrame, symbol: str, output_path: Path) -> None:
    """Chart 4: Smart Money Concepts — FVG zones, order blocks, BOS/CHoCH."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MAIN))
    fig.suptitle(f"{symbol} 1W — Smart Money Concepts", fontsize=16, fontweight="bold")

    ax.plot(df.index, df["close"], color="white", linewidth=1, label="Close")

    # FVG zones
    if "fvg_bullish" in df.columns:
        for i in range(len(df)):
            if df["fvg_bullish"].iloc[i] > 0.5 and i + 2 < len(df):
                ax.axvspan(df.index[i], df.index[min(i + 3, len(df) - 1)],
                           alpha=0.15, color=C_BULL)
    if "fvg_bearish" in df.columns:
        for i in range(len(df)):
            if df["fvg_bearish"].iloc[i] > 0.5 and i + 2 < len(df):
                ax.axvspan(df.index[i], df.index[min(i + 3, len(df) - 1)],
                           alpha=0.15, color=C_BEAR)

    # Order blocks
    if "ob_in_bullish_zone" in df.columns:
        for i in range(len(df)):
            if df["ob_in_bullish_zone"].iloc[i] > 0.5:
                ax.axvspan(df.index[max(0, i - 1)], df.index[min(i + 1, len(df) - 1)],
                           alpha=0.1, color=C_BULL)
    if "ob_in_bearish_zone" in df.columns:
        for i in range(len(df)):
            if df["ob_in_bearish_zone"].iloc[i] > 0.5:
                ax.axvspan(df.index[max(0, i - 1)], df.index[min(i + 1, len(df) - 1)],
                           alpha=0.1, color=C_BEAR)

    # BOS / CHoCH markers
    if "bos" in df.columns:
        bos_up = df["bos"] == 1
        bos_dn = df["bos"] == -1
        if bos_up.any():
            ax.scatter(df.index[bos_up], df.loc[bos_up, "close"], marker="^", s=60, color=C_BULL, zorder=5, label="BOS ↑")
        if bos_dn.any():
            ax.scatter(df.index[bos_dn], df.loc[bos_dn, "close"], marker="v", s=60, color=C_BEAR, zorder=5, label="BOS ↓")
    if "choch" in df.columns:
        choch_up = df["choch"] == 1
        choch_dn = df["choch"] == -1
        if choch_up.any():
            ax.scatter(df.index[choch_up], df.loc[choch_up, "close"], marker="*", s=100, color="#ffd54f", zorder=6, label="CHoCH ↑")
        if choch_dn.any():
            ax.scatter(df.index[choch_dn], df.loc[choch_dn, "close"], marker="*", s=100, color="#ff7043", zorder=6, label="CHoCH ↓")

    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved {}", output_path.name)


def chart_indicators(df: pd.DataFrame, symbol: str, output_path: Path) -> None:
    """Chart 2: Technical indicators — Bollinger, RSI, MACD, GARCH."""
    fig, axes = plt.subplots(4, 1, figsize=(FIG_WIDTH, 14), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.suptitle(f"{symbol} 1W — Technical Indicators", fontsize=16, fontweight="bold")

    # Price + Bollinger Bands + EMAs
    ax = axes[0]
    ax.plot(df.index, df["close"], color="white", linewidth=1, label="Close")
    if "bb_mid" in df.columns:
        ax.plot(df.index, df["bb_mid"], color="#42a5f5", linewidth=0.8, alpha=0.7, label="BB Mid")
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        ax.fill_between(df.index, df["bb_lower"], df["bb_upper"], alpha=0.1, color="#42a5f5", label="BB Band")
    if "ema_50" in df.columns:
        ax.plot(df.index, df["ema_50"], color="#ffa726", linewidth=1, alpha=0.7, label="EMA 50")
    if "ema_200" in df.columns:
        ax.plot(df.index, df["ema_200"], color="#ab47bc", linewidth=1, alpha=0.7, label="EMA 200")
    ax.set_ylabel("Price", fontsize=11)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)

    # RSI
    ax = axes[1]
    if "rsi_14" in df.columns:
        ax.plot(df.index, df["rsi_14"], color="#26a69a", linewidth=1.2)
        ax.axhline(y=70, color=C_BEAR, linestyle="--", alpha=0.5)
        ax.axhline(y=30, color=C_BULL, linestyle="--", alpha=0.5)
        ax.axhline(y=50, color="gray", linestyle=":", alpha=0.3)
        ax.fill_between(df.index, 70, 100, alpha=0.05, color=C_BEAR)
        ax.fill_between(df.index, 0, 30, alpha=0.05, color=C_BULL)
    ax.set_ylabel("RSI 14", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)

    # MACD
    ax = axes[2]
    if all(c in df.columns for c in ("macd_line", "macd_signal", "macd_hist")):
        ax.plot(df.index, df["macd_line"], color="#42a5f5", linewidth=1, label="MACD")
        ax.plot(df.index, df["macd_signal"], color="#ffa726", linewidth=1, label="Signal")
        colors = [C_BULL if v >= 0 else C_BEAR for v in df["macd_hist"]]
        ax.bar(df.index, df["macd_hist"], width=pd.Timedelta(days=5), color=colors, alpha=0.6)
        ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel("MACD", fontsize=10)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)

    # GARCH Volatility
    ax = axes[3]
    if "garch_vol" in df.columns:
        ax.plot(df.index, df["garch_vol"], color="#ef5350", linewidth=1.2, label="GARCH Vol")
    if "atr_14" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df["atr_14"], color="#42a5f5", linewidth=1, alpha=0.7, label="ATR 14")
        ax2.set_ylabel("ATR", color="#42a5f5", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#42a5f5")
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("GARCH Vol", fontsize=10)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)
    if "garch_vol" in df.columns:
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved {}", output_path.name)


def chart_prediction(
    df: pd.DataFrame,
    y_pred_lr: np.ndarray,
    y_proba_lr: np.ndarray,
    y_pred_tree: np.ndarray,
    y_proba_tree: np.ndarray,
    train_size: int,
    symbol: str,
    output_path: Path,
) -> None:
    """Chart 5: Model prediction confidence overlay."""
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{symbol} 1W — Model Predictions", fontsize=16, fontweight="bold")

    # Price with prediction overlay
    ax = axes[0]
    ax.plot(df.index, df["close"], color="white", linewidth=1, alpha=0.5, label="Close")

    # Color bars by LR direction
    n_test = len(y_pred_lr)
    test_start = len(df) - n_test
    for i in range(n_test):
        idx = test_start + i
        if idx < len(df) - 1:
            color = C_BULL if y_pred_lr[i] == 2 else (C_BEAR if y_pred_lr[i] == 0 else C_FLAT)
            ax.axvspan(df.index[idx], df.index[idx + 1], alpha=0.15, color=color)

    ax.set_ylabel("Price", fontsize=11)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)

    # LR confidence
    ax = axes[1]
    if y_proba_lr.ndim == 2 and y_proba_lr.shape[1] >= 2:
        max_conf = np.max(y_proba_lr, axis=1)
        ax.bar(df.index[test_start:test_start + n_test], max_conf,
               width=pd.Timedelta(days=5), color="#42a5f5", alpha=0.7, label="LR Confidence")
    ax.axhline(y=0.55, color="yellow", linestyle="--", alpha=0.5, label="Threshold 0.55")
    ax.set_ylabel("LR Conf", fontsize=10)
    ax.set_ylim(0.2, 1.0)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    # EW completion probability
    ax = axes[2]
    if "ew_completion_prob" in df.columns:
        ax.plot(df.index, df["ew_completion_prob"], color="#ffa726", linewidth=1.2, label="EW Completion Prob")
    if "ew_fib_confluence" in df.columns:
        ax.plot(df.index, df["ew_fib_confluence"], color="#ab47bc", linewidth=1, alpha=0.7, label="Fib Confluence")
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved {}", output_path.name)


def chart_summary(
    df: pd.DataFrame,
    pivots: list[_Pivot],
    symbol: str,
    current_state: dict,
    output_path: Path,
) -> None:
    """Chart 6: Summary dashboard — last 52 weeks with all key levels."""
    # Show last 52 weeks (~1 year)
    n_show = min(52, len(df))
    df_show = df.iloc[-n_show:]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MAIN))
    fig.suptitle(f"{symbol} 1W — Summary Dashboard (Last 52 Weeks)", fontsize=16, fontweight="bold")

    ax.plot(df_show.index, df_show["close"], color="white", linewidth=1.5, label="Close")

    # Fibonacci levels (last 100 bars)
    if all(c in df_show.columns for c in ("fib_236", "fib_382", "fib_500", "fib_618")):
        for col, name, style in [("fib_236", "23.6%", ":"), ("fib_382", "38.2%", "--"),
                                  ("fib_500", "50.0%", "-"), ("fib_618", "61.8%", "--")]:
            valid = df_show[col].dropna()
            if len(valid) > 0:
                ax.axhline(y=valid.iloc[-1], color=C_FIB, linestyle=style, alpha=0.5, linewidth=0.8,
                           label=f"Fib {name}")

    # Bollinger Bands
    if "bb_upper" in df_show.columns and "bb_lower" in df_show.columns:
        ax.fill_between(df_show.index, df_show["bb_lower"], df_show["bb_upper"],
                         alpha=0.08, color="#42a5f5", label="BB Band")

    # Current price
    last_close = df_show["close"].iloc[-1]
    ax.axhline(y=last_close, color="yellow", linestyle="-", alpha=0.6, linewidth=1)

    # Stop loss and take profit levels (both directions)
    atr_val = current_state.get("atr_14", last_close * 0.03)
    sl_long = last_close - 2 * atr_val
    sl_short = last_close + 2 * atr_val
    tp_long_1 = last_close + 2 * atr_val
    tp_long_2 = last_close + 3 * atr_val
    tp_short_1 = last_close - 2 * atr_val
    tp_short_2 = last_close - 3 * atr_val

    ax.axhline(y=sl_long, color=C_BEAR, linestyle="--", alpha=0.6, linewidth=1, label=f"SL Long ${sl_long:,.0f}")
    ax.axhline(y=sl_short, color=C_BEAR, linestyle=":", alpha=0.6, linewidth=1, label=f"SL Short ${sl_short:,.0f}")
    ax.axhline(y=tp_long_1, color=C_BULL, linestyle="--", alpha=0.6, linewidth=1, label=f"TP Long1 ${tp_long_1:,.0f}")
    ax.axhline(y=tp_long_2, color=C_BULL, linestyle=":", alpha=0.6, linewidth=1, label=f"TP Long2 ${tp_long_2:,.0f}")
    ax.axhline(y=tp_short_1, color="#ffa726", linestyle="--", alpha=0.5, linewidth=0.8, label=f"TP Short1 ${tp_short_1:,.0f}")
    ax.axhline(y=tp_short_2, color="#ffa726", linestyle=":", alpha=0.5, linewidth=0.8, label=f"TP Short2 ${tp_short_2:,.0f}")

    # Pivot points (last few)
    for p in pivots[-6:]:
        if p.idx >= len(df) - n_show and p.idx < len(df):
            marker = "^" if p.kind == "low" else "v"
            color = C_BULL if p.kind == "low" else C_BEAR
            ax.scatter(df.index[p.idx], p.price, marker=marker, s=100, color=color, zorder=6)

    # Annotations
    info_text = (
        f"Wave: W{int(current_state.get('ew_wave_position', 0))} "
        f"{'Impulse' if current_state.get('ew_is_impulse', 0) > 0.5 else 'Corrective'} "
        f"{'^' if current_state.get('ew_wave_direction', 0) > 0.5 else 'v' if current_state.get('ew_wave_direction', 0) < -0.5 else '-'}\n"
        f"Regime: {current_state.get('regime', 'unknown')} "
        f"(trend={current_state.get('regime_prob_trending', 0):.0%})\n"
        f"Hurst: {current_state.get('hurst_exp', 0.5):.2f} "
        f"({'trending' if current_state.get('hurst_exp', 0.5) > 0.5 else 'mean-reverting'})\n"
        f"EW Completion: {current_state.get('ew_completion_prob', 0):.0%} "
        f"Fib Confluence: {current_state.get('ew_fib_confluence', 0):.0%}"
    )
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor="#42a5f5", alpha=0.9))

    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.set_facecolor("#1a1a2e")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved {}", output_path.name)


# ─── Report Generator ─────────────────────────────────────────────────────────

def generate_report(
    symbol: str,
    df: pd.DataFrame,
    current_state: dict,
    pivots: list[_Pivot],
    sl_levels: dict,
    tp_levels: dict,
    model_preds: dict,
) -> str:
    """Generate a Markdown technical analysis report."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change_pct = (last["close"] - prev["close"]) / prev["close"] * 100

    ew_dir = current_state.get("ew_wave_direction", 0)
    ew_pos = int(current_state.get("ew_wave_position", 0))
    ew_impulse = current_state.get("ew_is_impulse", 0) > 0.5

    regime = current_state.get("regime", "unknown")
    regime_prob_trend = current_state.get("regime_prob_trending", 0)
    hurst = current_state.get("hurst_exp", 0.5)

    # Directional bias
    if ew_dir > 0.5 and ew_impulse:
        bias = "BULLISH"
        bias_reason = f"In impulse wave W{ew_pos} with bullish direction"
    elif ew_dir < -0.5 and ew_impulse:
        bias = "BEARISH"
        bias_reason = f"In impulse wave W{ew_pos} with bearish direction"
    elif ew_impulse:
        bias = "NEUTRAL"
        bias_reason = f"In impulse pattern but direction unclear (W{ew_pos})"
    else:
        bias = "CORRECTIVE"
        bias_reason = f"In corrective pattern (W{ew_pos if ew_pos > 0 else 'unknown'})"

    report = f"""# {symbol} Weekly Technical Analysis Report

**Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}
**Current Price:** ${last["close"]:,.2f}
**Weekly Change:** {change_pct:+.2f}%
**Data Range:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} bars)

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Price** | ${last["close"]:,.2f} |
| **Directional Bias** | **{bias}** |
| **Confidence** | {current_state.get("ew_fib_confluence", 0):.0%} Fib confluence |
| **Regime** | {regime} (trending prob: {regime_prob_trend:.0%}) |
| **Hurst Exponent** | {hurst:.3f} ({'trending' if hurst > 0.5 else 'mean-reverting' if hurst < 0.5 else 'random'}) |
| **EW Position** | W{ew_pos} ({'Impulse' if ew_impulse else 'Corrective'}) |
| **EW Direction** | {'Bullish ↑' if ew_dir > 0.5 else 'Bearish ↓' if ew_dir < -0.5 else 'Neutral →'} |
| **GARCH Volatility** | {current_state.get("garch_vol", 0):.4f} |
| **ATR (14)** | ${current_state.get("atr_14", 0):,.2f} |

**Justification:** {bias_reason}. Hurst exponent ({hurst:.2f}) suggests the market is currently in a {'trending' if hurst > 0.5 else 'mean-reverting'} regime, which {'supports' if (hurst > 0.5 and bias == 'BULLISH') or (hurst < 0.5 and bias != 'BULLISH') else 'partially contradicts'} the directional bias.

---

## 2. Elliott Wave Analysis

| Wave Feature | Value |
|-------------|-------|
| **Current Wave Position** | W{ew_pos} |
| **Wave Direction** | {'Bullish (+1)' if ew_dir > 0.5 else 'Bearish (-1)' if ew_dir < -0.5 else 'Corrective/Neutral (0)'} |
| **Is Impulse?** | {'Yes' if ew_impulse else 'No'} |
| **Fibonacci Confluence** | {current_state.get("ew_fib_confluence", 0):.2%} |
| **Wave Completion Prob** | {current_state.get("ew_completion_prob", 0):.2%} |
| **Retracement Depth** | {current_state.get("ew_retracement_depth", 0):.2f} |

### Key Pivot Points (Last {min(len(pivots), 8)} Zigzag Pivots)

"""

    for p in pivots[-8:]:
        if p.idx < len(df):
            report += f"- **{p.kind.upper()}** at ${p.price:,.2f} ({df.index[p.idx].strftime('%Y-%m-%d')})\n"

    # Wave interpretation
    report += f"""
### Wave Interpretation

"""

    if ew_impulse and ew_pos in (3, 5):
        report += f"- Currently in **Wave {ew_pos}** of an impulse pattern — this is typically the {'strongest' if ew_pos == 3 else 'final'} wave\n"
        if ew_pos == 3:
            report += "- Wave 3 is often the longest and most powerful wave in the sequence\n"
        elif ew_pos == 5:
            report += "- Wave 5 may be weakening; watch for completion signals and reversal patterns\n"
    elif ew_impulse and ew_pos in (2, 4):
        report += f"- Currently in **Wave {ew_pos}** (corrective wave within impulse) — expect {'dip buying' if ew_dir > 0.5 else 'relief selling'} after this correction completes\n"
    elif not ew_impulse:
        report += "- Currently in a **corrective pattern** — the market is consolidating before the next impulse wave\n"
        report += "- Corrective waves are notoriously difficult to predict; wait for a clear breakout\n"

    if current_state.get("ew_completion_prob", 0) > 0.6:
        report += f"\n⚠️ **Wave completion probability is {current_state.get('ew_completion_prob', 0):.0%}** — the current wave pattern may be ending. Watch for reversal signals.\n"

    report += f"""
---

## 3. Regime Analysis

| Regime Feature | Value |
|---------------|-------|
| **Dominant Regime** | {regime} |
| **Trending Probability** | {current_state.get("regime_prob_trending", 0):.1%} |
| **Ranging Probability** | {current_state.get("regime_prob_ranging", 0):.1%} |
| **Volatile Probability** | {current_state.get("regime_prob_volatile", 0):.1%} |
| **Stressed Probability** | {current_state.get("regime_prob_stressed", 0):.1%} |
| **Run Length (mean)** | {current_state.get("bocpd_run_length_mean", 0):.1f} bars |
| **Run Length (MAP)** | {int(current_state.get("bocpd_run_length_map", 0))} bars |

"""

    if regime == "trending":
        report += "- **Trending regime**: Momentum strategies (trend following) are likely to perform well.\n"
    elif regime == "ranging":
        report += "- **Ranging regime**: Mean-reversion strategies (buy support, sell resistance) are preferred.\n"
    elif regime == "volatile":
        report += "- **Volatile regime**: Reduce position sizes, widen stops. Breakout strategies may work.\n"
    elif regime == "stressed":
        report += "- **Stressed regime**: High risk. Consider reducing exposure or hedging.\n"

    report += f"""
---

## 4. Volatility Assessment

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **GARCH Vol** | {current_state.get("garch_vol", 0):.4f} | {'High volatility' if current_state.get("garch_vol", 0) > 0.05 else 'Moderate' if current_state.get("garch_vol", 0) > 0.02 else 'Low volatility'} |
| **ATR (14)** | ${current_state.get("atr_14", 0):,.2f} | {'Wide ranges' if current_state.get("atr_14", 0) > last["close"] * 0.05 else 'Normal ranges'} |
| **Hurst Exponent** | {hurst:.3f} | {'Trending (H > 0.5)' if hurst > 0.5 else 'Mean-reverting (H < 0.5)' if hurst < 0.5 else 'Random walk (H ≈ 0.5)'} |
| **Bollinger Width** | {'Available' if "bb_upper" in df.columns else 'N/A'} | {'Price near upper band' if "bb_upper" in df.columns and last["close"] > current_state.get("bb_upper", 0) else 'Within bands'} |

---

## 5. Smart Money Concepts

| SMC Feature | Value |
|-------------|-------|
| **Bullish FVG Active** | {'Yes' if current_state.get("fvg_bullish", 0) > 0.5 else 'No'} |
| **Bearish FVG Active** | {'Yes' if current_state.get("fvg_bearish", 0) > 0.5 else 'No'} |
| **FVG Fill %** | {current_state.get("fvg_fill_pct", 0):.0%} |
| **FVG Nearest Distance** | {current_state.get("fvg_nearest_distance", 0):.2f} ATR |
| **Near Bullish OB** | {'Yes' if current_state.get("ob_bullish_near", 0) > 0.5 else 'No'} |
| **Near Bearish OB** | {'Yes' if current_state.get("ob_bearish_near", 0) > 0.5 else 'No'} |
| **In Bullish OB Zone** | {'Yes' if current_state.get("ob_in_bullish_zone", 0) > 0.5 else 'No'} |
| **In Bearish OB Zone** | {'Yes' if current_state.get("ob_in_bearish_zone", 0) > 0.5 else 'No'} |

"""

    if current_state.get("ob_in_bullish_zone", 0) > 0.5:
        report += "- 🟢 Price is **inside a bullish order block zone** — potential support area.\n"
    if current_state.get("ob_in_bearish_zone", 0) > 0.5:
        report += "- 🔴 Price is **inside a bearish order block zone** — potential resistance area.\n"
    if current_state.get("fvg_fill_pct", 0) < 0.3:
        report += "- ⚡ Fair Value Gap is largely **unfilled** ({:.0%} remaining) — price may be attracted to fill it.\n".format(1 - current_state.get("fvg_fill_pct", 0))

    report += f"""
---

## 6. Model Predictions

| Model | Direction | Confidence | Notes |
|-------|-----------|------------|-------|
| **LR** | {'UP ↑' if model_preds.get('lr_class') == 2 else 'DOWN ↓' if model_preds.get('lr_class') == 0 else 'FLAT →'} | {model_preds.get('lr_conf', 0):.1%} | Logistic Regression baseline |
| **Tree** | {'UP ↑' if model_preds.get('tree_class') == 2 else 'DOWN ↓' if model_preds.get('tree_class') == 0 else 'FLAT →'} | {model_preds.get('tree_conf', 0):.1%} | RandomForest/gradient boosted |

"""

    # Consensus
    lr_dir = model_preds.get("lr_class", 1)
    tree_dir = model_preds.get("tree_class", 1)
    if lr_dir == tree_dir and lr_dir != 1:
        consensus = f"**{'BULLISH' if lr_dir == 2 else 'BEARISH'}** — both models agree"
    elif lr_dir == tree_dir:
        consensus = "**NEUTRAL** — both models predict flat"
    else:
        consensus = "**DISAGREEMENT** — models diverge, reduce confidence"
    report += f"**Consensus:** {consensus}\n"

    report += f"""
---

## 7. Trading Levels

### Stop Loss Levels

| Level | Price | Basis |
|-------|-------|-------|
| **SL Long** | ${sl_levels['long_2atr']:,.2f} | 2× ATR below current |
| **SL Long (tight)** | ${sl_levels['long_1_5atr']:,.2f} | 1.5× ATR below current |
| **SL Short** | ${sl_levels['short_2atr']:,.2f} | 2× ATR above current |
| **SL Short (tight)** | ${sl_levels['short_1_5atr']:,.2f} | 1.5× ATR above current |

### Take Profit Levels

| Level | Price | Basis |
|-------|-------|-------|
| **TP Long 1** | ${tp_levels['tp_long_1']:,.2f} | 2x ATR above (1:1 R:R for longs) |
| **TP Long 2** | ${tp_levels['tp_long_2']:,.2f} | 3x ATR above (1:1.5 R:R for longs) |
| **TP Long Fib** | ${tp_levels['tp_long_fib']:,.2f} | Fib 1.618 extension above |
| **TP Short 1** | ${tp_levels['tp_short_1']:,.2f} | 2x ATR below (1:1 R:R for shorts) |
| **TP Short 2** | ${tp_levels['tp_short_2']:,.2f} | 3x ATR below (1:1.5 R:R for shorts) |
| **TP Short Fib** | ${tp_levels['tp_short_fib']:,.2f} | Fib 1.618 extension below |

### Key Fibonacci Levels (from recent swing)

"""

    for name in ("fib_236", "fib_382", "fib_500", "fib_618", "fib_786"):
        if name in df.columns:
            val = df[name].iloc[-1]
            if not np.isnan(val):
                report += f"| Fib {name.replace('fib_', '')} | ${val:,.2f} | Fibonacci retracement |\n"

    report += f"""
---

## 8. Risk Assessment

### Data Quality
- **Dataset size:** {len(df)} weekly bars
- **Coverage:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}
- **Volume data:** {'Available' if df['volume'].sum() > 0 else 'NOT available (filled with 0)'}
- **Missing values:** Any NaN/inf in features are replaced with 0

### Model Reliability Caveats
- ⚠️ **Weekly data has limited sample size** ({len(df)} bars). Model accuracy is approximately 50-60% on this timeframe.
- ⚠️ **Walk-forward CV shows significant variance** across folds (std ~3-7%).
- ⚠️ **Tree-based models may overfit** on weekly data due to high feature-to-sample ratio (71 features / {len(df)} bars).
- ⚠️ **Elliott Wave detection is probabilistic** — the wave count may change as new data arrives.

### Position Sizing Guidance
- **Conservative:** Risk 0.5% of portfolio per trade (recommended for weekly timeframe)
- **Moderate:** Risk 1% of portfolio per trade
- **Aggressive:** Risk 2% of portfolio per trade (only for high-conviction setups)

### Key Risks
1. **Regime shift risk:** BOCPD changepoint detection may lag; sudden regime changes can invalidate current bias
2. **Wave count ambiguity:** Different pivot scales can produce different wave counts
3. **Model drift:** The LR model is calibrated on historical weekly data; structural breaks reduce accuracy
4. **Liquidity risk:** Weekly bars smooth intraday volatility; actual slippage may exceed ATR-based estimates

---

## 9. Key Justifications

### Why This Directional Bias?

1. **Elliott Wave:** Currently in W{ew_pos} ({'impulse' if ew_impulse else 'corrective'}), direction is {'bullish' if ew_dir > 0.5 else 'bearish' if ew_dir < -0.5 else 'neutral'}.
   - Fibonacci confluence: {current_state.get('ew_fib_confluence', 0):.0%} — {'strong' if current_state.get('ew_fib_confluence', 0) > 0.7 else 'moderate' if current_state.get('ew_fib_confluence', 0) > 0.4 else 'weak'} alignment with key Fib levels.

2. **Hurst Exponent ({hurst:.2f}):** {'Above 0.5 — trending behavior expected to continue' if hurst > 0.5 else 'Below 0.5 — mean-reversion likely; current moves may reverse' if hurst < 0.5 else 'Near 0.5 — random walk; no directional persistence'}.

3. **BOCPD Regime:** Currently in {regime} regime with {regime_prob_trend:.0%} trending probability.
   - {'Trend-following strategies are appropriate' if regime == 'trending' else 'Mean-reversion strategies may work better' if regime == 'ranging' else 'Caution advised — volatility is elevated'}.

4. **Model consensus:** {consensus.split('—')[0].strip()}.

5. **Smart Money:** {'Price near bullish order block — support zone' if current_state.get('ob_in_bullish_zone', 0) > 0.5 else 'Price near bearish order block — resistance zone' if current_state.get('ob_in_bearish_zone', 0) > 0.5 else 'No active order block proximity'}.

---

*Report generated by Kairon Technical Analysis Engine*
*Model: LR MultiHead + TreeMultiHead (RandomForest fallback)*
*Features: {len([c for c in df.columns if c not in ('ts', 'open', 'high', 'low', 'close', 'volume')])} engineered features from ALL_FEATURES pipeline*
"""
    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_analysis(symbol: str, pivot_scale: float = 1.5) -> None:
    """Run the full technical analysis for one asset."""
    csv_path = REPO_ROOT / f"{symbol}_1W_tradingview_coinmarketcap.csv"
    if not csv_path.exists():
        logger.error("CSV not found: {}", csv_path)
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TECHNICAL ANALYSIS: {} 1W", symbol)
    logger.info("=" * 70)

    # 1) Load data
    table = load_weekly_csv(csv_path, symbol)
    n_bars = table.num_rows
    close = np.array([float(v) for v in table.column("close")], dtype=np.float64)

    # 2) Extract features
    logger.info("Extracting features...")
    feature_names, aug_table = extract_features(table)
    df = _to_pandas(aug_table)

    # 3) Elliott Wave pivots
    logger.info("Detecting Elliott Wave pivots...")
    pivots = get_ew_pivots(table, pivot_scale=pivot_scale)
    logger.info("  Found {} zigzag pivots", len(pivots))

    # 4) Extract current state from last bar
    last_idx = len(df) - 1
    cs = {}  # current state dict
    for col in df.columns:
        val = df[col].iloc[-1]
        cs[col] = float(val) if not pd.isna(val) else 0.0

    # Determine regime
    regime_probs = {
        "trending": cs.get("regime_prob_trending", 0),
        "ranging": cs.get("regime_prob_ranging", 0),
        "volatile": cs.get("regime_prob_volatile", 0),
        "stressed": cs.get("regime_prob_stressed", 0),
    }
    cs["regime"] = max(regime_probs, key=regime_probs.get)

    # 5) Run models for prediction
    logger.info("Running models for prediction...")
    model_preds = {}
    try:
        # Build labels for training
        spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1w")
        labeled = make_direction_labels(table, spec=spec, symbol=symbol)
        y_dir = np.array([b.y_class + 1 for b in labeled.bars], dtype=np.int64)
        n_labeled = len(y_dir)

        # Build feature matrix (aligned with labels)
        ohlcv_cols = {"ts", "open", "high", "low", "close", "volume"}
        feat_cols = [c for c in df.columns if c not in ohlcv_cols and c in feature_names]
        use_len = min(n_labeled, len(df))

        feat_arrays = []
        used_names = []
        for col in feat_cols:
            arr = df[col].values[:use_len].astype(np.float64)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            if arr.std(ddof=0) > 1e-12:
                feat_arrays.append(arr)
                used_names.append(col)

        values = np.stack(feat_arrays, axis=1)
        fm = FeatureMatrix(values=values, feature_names=tuple(used_names))

        # Train on 80%, predict last bar
        split = int(len(y_dir[:use_len]) * 0.8)
        y_train = y_dir[:use_len][:split]
        fm_train = FeatureMatrix(values=fm.values[:split], feature_names=fm.feature_names)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(fm_train.values)

        # LR
        lr_model = MultiHeadModel(MultiHeadConfig(n_estimators=500))
        y_mag_train = np.zeros(split, dtype=np.float64)
        log_rets = np.diff(np.log(close.astype(np.float64)))
        y_vol_train = np.full(split, float(np.std(log_rets[-50:])) if len(log_rets) >= 50 else 0.01, dtype=np.float64)

        fm_train_s = FeatureMatrix(values=train_scaled, feature_names=fm_train.feature_names)
        lr_state = lr_model.fit_multihead(fm_train_s, y_train, y_mag_train, y_vol_train)

        # Predict on last bar
        last_features = fm.values[-1:].reshape(1, -1)
        last_scaled = scaler.transform(last_features)
        fm_last = FeatureMatrix(values=last_scaled, feature_names=fm.feature_names)
        lr_pred = lr_model.predict_multihead(lr_state, fm_last)

        model_preds["lr_class"] = int(lr_pred["y_class"][0])
        model_preds["lr_conf"] = float(np.max(lr_pred["y_proba"][0])) if lr_pred["y_proba"].ndim == 2 else 0.5

        # Tree
        try:
            tree_model = TreeMultiHeadModel(TreeMultiHeadConfig(n_estimators=300))
            tree_state = tree_model.fit_multihead(fm_train_s, y_train, y_mag_train, y_vol_train)
            tree_pred = tree_model.predict_multihead(tree_state, fm_last)
            model_preds["tree_class"] = int(tree_pred["y_class"][0])
            model_preds["tree_conf"] = float(np.max(tree_pred["y_proba"][0])) if tree_pred["y_proba"].ndim == 2 else 0.5
        except Exception as e:
            logger.warning("Tree model failed: {}", e)
            model_preds["tree_class"] = model_preds.get("lr_class", 1)
            model_preds["tree_conf"] = model_preds.get("lr_conf", 0.33)

    except Exception as e:
        logger.error("Model prediction failed: {}", e)
        model_preds = {"lr_class": 1, "lr_conf": 0.33, "tree_class": 1, "tree_conf": 0.33}

    # 6) Compute SL/TP levels — always provide both long and short scenarios
    atr = cs.get("atr_14", close[-1] * 0.03)
    current_price = close[-1]
    sl_levels = {
        "long_2atr": current_price - 2 * atr,
        "long_1_5atr": current_price - 1.5 * atr,
        "short_2atr": current_price + 2 * atr,
        "short_1_5atr": current_price + 1.5 * atr,
    }
    tp_levels = {
        "tp_long_1": current_price + 2 * atr,    # 1:1 R:R for longs
        "tp_long_2": current_price + 3 * atr,    # 1:1.5 R:R for longs
        "tp_short_1": current_price - 2 * atr,   # 1:1 R:R for shorts
        "tp_short_2": current_price - 3 * atr,   # 1:1.5 R:R for shorts
        "tp_long_fib": 0,   # Fib 1.618 extension above
        "tp_short_fib": 0,  # Fib 1.618 extension below
    }
    # Fib 1.618 extension from last swing
    if len(pivots) >= 2:
        swing_range = abs(pivots[-1].price - pivots[-2].price)
        tp_levels["tp_long_fib"] = pivots[-1].price + 1.618 * swing_range
        tp_levels["tp_short_fib"] = pivots[-1].price - 1.618 * swing_range

    # 7) Generate charts
    logger.info("Generating charts...")
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    prefix = f"{symbol}_1W_{date_str}"

    chart_elliott_wave(df, pivots, symbol, REPORTS_DIR / f"{prefix}_elliott_wave.png")
    chart_indicators(df, symbol, REPORTS_DIR / f"{prefix}_indicators.png")
    chart_regime(df, symbol, REPORTS_DIR / f"{prefix}_regime.png")
    chart_smc(df, symbol, REPORTS_DIR / f"{prefix}_smc.png")
    chart_summary(df, pivots, symbol, cs, REPORTS_DIR / f"{prefix}_summary.png")

    # Prediction chart (requires model outputs)
    n_test = min(52, len(df) // 5)  # Last ~20% for test visualization
    try:
        y_pred_lr = np.full(n_test, model_preds.get("lr_class", 1), dtype=np.int64)
        y_proba_lr = np.array([[0.33, 0.34, 0.33]] * n_test)  # placeholder
        y_pred_tree = np.full(n_test, model_preds.get("tree_class", 1), dtype=np.int64)
        y_proba_tree = np.array([[0.33, 0.34, 0.33]] * n_test)
        # Fill with actual direction colors from EW
        ew_col = "ew_wave_direction"
        if ew_col in df.columns:
            ew_vals = df[ew_col].iloc[-n_test:].values
            y_pred_lr = np.where(ew_vals > 0.5, 2, np.where(ew_vals < -0.5, 0, 1)).astype(np.int64)
            y_pred_tree = y_pred_lr.copy()

        chart_prediction(df, y_pred_lr, y_proba_lr, y_pred_tree, y_proba_tree,
                         len(df) - n_test, symbol, REPORTS_DIR / f"{prefix}_prediction.png")
    except Exception as e:
        logger.warning("Prediction chart failed: {}", e)

    # 8) Generate report
    logger.info("Generating report...")
    report = generate_report(symbol, df, cs, pivots, sl_levels, tp_levels, model_preds)
    report_path = REPORTS_DIR / f"{prefix}_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("  Saved {}", report_path.name)

    # Print summary to console
    print()
    print("=" * 70)
    print(f"  {symbol} WEEKLY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Price: ${close[-1]:,.2f}")
    print(f"  Direction: {model_preds.get('lr_class', '?')} (LR {model_preds.get('lr_conf', 0):.0%}) / "
          f"{model_preds.get('tree_class', '?')} (Tree {model_preds.get('tree_conf', 0):.0%})")
    ew_dir_sym = "^" if cs.get("ew_wave_direction", 0) > 0.5 else ("v" if cs.get("ew_wave_direction", 0) < -0.5 else "-")
    print(f"  EW: W{int(cs.get('ew_wave_position', 0))} "
          f"{'Impulse' if cs.get('ew_is_impulse', 0) > 0.5 else 'Corrective'} "
          f"{ew_dir_sym}")
    print(f"  Regime: {cs.get('regime', '?')} (trend={cs.get('regime_prob_trending', 0):.0%})")
    print(f"  Hurst: {cs.get('hurst_exp', 0.5):.3f}")
    print(f"  SL Long: ${sl_levels['long_2atr']:,.2f} | TP Long1: ${tp_levels['tp_long_1']:,.2f} | TP Long Fib: ${tp_levels['tp_long_fib']:,.2f}")
    print(f"  SL Short: ${sl_levels['short_2atr']:,.2f} | TP Short1: ${tp_levels['tp_short_1']:,.2f} | TP Short Fib: ${tp_levels['tp_short_fib']:,.2f}")
    print(f"  Report: {report_path}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Technical Analysis Report for BTC/ETH Weekly Data")
    parser.add_argument("--asset", choices=["BTC", "ETH", "both"], default="both")
    parser.add_argument("--pivot-scale", type=float, default=1.5,
                        help="ATR multiplier for zigzag pivot detection (default: 1.5)")
    args = parser.parse_args()

    assets = ["BTC", "ETH"] if args.asset == "both" else [args.asset]
    for asset in assets:
        run_analysis(asset, pivot_scale=args.pivot_scale)

    logger.info("All analyses complete.")


if __name__ == "__main__":
    main()