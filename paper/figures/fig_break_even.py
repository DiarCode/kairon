"""Figure 9: Break-even accuracy heatmap.

Rows = assets (BTC, ETH, SOL), columns = horizons (1h, 5m).
Cell color = break-even accuracy (p*). Higher = harder to beat costs.

Output: break_even_heatmap.pdf + .png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns


def generate(
    real_data: dict,
    ablation_data: dict,
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate break-even accuracy heatmap."""
    paths = []
    cells = real_data.get("cells", [])

    symbols = ["BTC", "ETH", "SOL"]
    timeframes = ["1h", "5m"]

    # Build matrix
    matrix = np.full((3, 2), np.nan)
    annotations = [["" for _ in range(2)] for _ in range(3)]

    for cell in cells:
        if "error" in cell:
            continue
        sym = cell.get("symbol", "UNK")
        tf = cell.get("timeframe", "1h")
        be = cell.get("break_even", {})
        be_acc = be.get("break_even_accuracy", 0.55)
        move_bps = be.get("expected_move_bps", 0)

        row = symbols.index(sym) if sym in symbols else None
        col = timeframes.index(tf) if tf in timeframes else None
        if row is not None and col is not None:
            matrix[row, col] = be_acc
            annotations[row][col] = f"p*={be_acc:.3f}\nΔ={move_bps:.0f}bps"

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Break-Even Accuracy Heatmap", fontsize=14, fontweight="bold")

    # Use seaborn heatmap
    sns.heatmap(
        matrix,
        annot=np.array(annotations),
        fmt="",
        xticklabels=timeframes,
        yticklabels=symbols,
        cmap="RdYlGn_r",  # red=hard (high p*), green=easy (low p*)
        vmin=0.50,
        vmax=0.80,
        ax=ax,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Break-even accuracy (p*)"},
    )

    ax.set_xlabel("Timeframe", fontsize=11)
    ax.set_ylabel("Asset", fontsize=11)

    plt.tight_layout()
    stem = "break_even_heatmap"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths