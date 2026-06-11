"""Figure 5: Price chart with BOCPD regime background + changepoint lines.

Shows BTC/USDT 1h with:
- Price line on top subplot
- Regime background color (trending=green, ranging=grey, volatile=red)
- BOCPD changepoint vertical lines
- Run-length posterior heatmap on bottom subplot

Output: regime_detection_btc_1h.pdf + .png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

REGIME_COLORS = {
    "trending": (0.2, 0.8, 0.2, 0.25),
    "ranging": (0.5, 0.5, 0.5, 0.15),
    "volatile": (0.9, 0.2, 0.2, 0.25),
    "stressed": (0.8, 0.0, 0.0, 0.30),
}


def generate(
    real_data: dict,
    ablation_data: dict,
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate regime detection figure for BTC 1h."""
    paths = []

    try:
        from kairon.data.io import DataPaths, read_ohlcv
        from kairon.data.symbols import CryptoVenue, crypto_spot
        from kairon.features.regime import BOCPDConfig, BOCPDRegimeDetector

        sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
        table = read_ohlcv(symbol=sym, venue="binance", timeframe="1h",
                           paths=DataPaths.default())
        close = np.array(table.column("close").to_pylist(), dtype=np.float64)
        high = np.array(table.column("high").to_pylist(), dtype=np.float64)
        low = np.array(table.column("low").to_pylist(), dtype=np.float64)
        n = len(close)

        # Regime detection
        realized_vol = np.abs(np.diff(np.log(close)))
        spread_bps = (high[1:] - low[1:]) / close[1:] * 10000
        detector = BOCPDRegimeDetector(BOCPDConfig())
        states = detector.detect(realized_vol, spread_bps)
        regime_labels = np.array([s.regime for s in states])

        # Changepoints
        cps = detector.changepoints(realized_vol, spread_bps)

        # Plot
        fig, (ax_price, ax_rl) = plt.subplots(2, 1, figsize=(13, 7),
                                                gridspec_kw={"height_ratios": [3, 1]},
                                                sharex=True)
        fig.suptitle("BTC/USDT 1h — BOCPD Regime Detection", fontsize=14, fontweight="bold")

        # Top: price + regime shading
        ax_price.plot(close, color="#333333", linewidth=0.8, label="BTC close")

        # Shade regime backgrounds
        prev_regime = None
        seg_start = 0
        for i in range(len(regime_labels)):
            r = str(regime_labels[i])
            if r != prev_regime or i == len(regime_labels) - 1:
                if prev_regime is not None:
                    color = REGIME_COLORS.get(prev_regime, (0.5, 0.5, 0.5, 0.1))
                    ax_price.axvspan(seg_start, i, color=color, linewidth=0)
                seg_start = i
                prev_regime = r

        # Changepoint lines
        for cp in cps[:30]:  # limit to 30 for readability
            ax_price.axvline(x=cp, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

        ax_price.set_ylabel("Price (USDT)", fontsize=10)
        ax_price.legend(fontsize=8, loc="upper left")
        ax_price.grid(True, alpha=0.3)

        # Bottom: run-length posterior heatmap
        # Build a simplified run-length matrix
        max_rl = 50
        rl_matrix = np.zeros((max_rl, len(states)))
        for i, s in enumerate(states):
            rl = s.run_length_posterior
            use_len = min(len(rl), max_rl)
            rl_matrix[:use_len, i] = rl[:use_len]

        ax_rl.imshow(rl_matrix, aspect="auto", cmap="YlOrRd",
                     origin="lower", interpolation="nearest",
                     extent=[0, len(states), 0, max_rl])
        ax_rl.set_ylabel("Run length", fontsize=9)
        ax_rl.set_xlabel("Bar index", fontsize=9)

        # Regime legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=REGIME_COLORS["trending"], label="Trending"),
            Patch(facecolor=REGIME_COLORS["ranging"], label="Ranging"),
            Patch(facecolor=REGIME_COLORS["volatile"], label="Volatile"),
            Patch(facecolor=REGIME_COLORS["stressed"], label="Stressed"),
        ]
        ax_price.legend(handles=legend_elements, fontsize=7, loc="upper left")

    except Exception as exc:
        # Fallback: empty figure with error message
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.text(0.5, 0.5, f"Regime detection figure requires real data\n({exc})",
                transform=ax.transAxes, ha="center", va="center", fontsize=14, color="grey")
        ax.set_title("BTC/USDT 1h — BOCPD Regime Detection (data unavailable)", fontsize=14)

    plt.tight_layout()
    stem = "regime_detection_btc_1h"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths