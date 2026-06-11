"""Figure 6: Cost sensitivity grouped bar chart.

X = cost multiplier (0.5x, 1x, 2x, 5x), Y = Sharpe ratio.
One group per (asset, horizon) cell.

Output: cost_sensitivity.pdf + .png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSET_COLORS = {"BTC": "#F7931A", "ETH": "#627EEA", "SOL": "#14F195"}


def generate(
    real_data: dict,
    ablation_data: dict,
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate cost sensitivity figure."""
    paths = []
    cells = real_data.get("cells", [])

    multipliers = [0.5, 1.0, 2.0, 5.0]
    bar_width = 0.12

    fig, (ax_sharpe, ax_be) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cost Sensitivity Analysis", fontsize=14, fontweight="bold")

    # Collect data
    cell_data = {}
    for cell in cells:
        if "error" in cell:
            continue
        key = f"{cell.get('symbol', 'UNK')}_{cell.get('timeframe', '1h')}"
        cs = cell.get("cost_sensitivity", {})
        cell_data[key] = cs

    x_pos = np.arange(len(multipliers))

    for i, cell_key in enumerate(sorted(cell_data.keys())):
        cs = cell_data[cell_key]
        sym = cell_key.split("_")[0]
        color = ASSET_COLORS.get(sym, "#333333")
        tf = cell_key.split("_")[1]
        style = "--" if tf == "5m" else "-"

        sharpes = [cs.get(f"{m}x", {}).get("sharpe", 0.0) for m in multipliers]
        offset = (i - len(cell_data) / 2 + 0.5) * bar_width

        ax_sharpe.bar(x_pos + offset, sharpes, bar_width, color=color,
                      label=cell_key, alpha=0.85 if tf == "1h" else 0.6,
                      edgecolor="white", linewidth=0.5)

    ax_sharpe.set_xlabel("Cost Multiplier", fontsize=11)
    ax_sharpe.set_ylabel("Sharpe Ratio", fontsize=11)
    ax_sharpe.set_xticks(x_pos)
    ax_sharpe.set_xticklabels([f"{m}x" for m in multipliers])
    ax_sharpe.axhline(y=0, color="black", linewidth=0.5)
    ax_sharpe.legend(fontsize=7, loc="upper right", ncol=2)
    ax_sharpe.grid(True, alpha=0.3, axis="y")

    # Break-even accuracy subplot
    for cell in cells:
        if "error" in cell:
            continue
        sym = cell.get("symbol", "UNK")
        tf = cell.get("timeframe", "1h")
        color = ASSET_COLORS.get(sym, "#333333")
        be = cell.get("break_even", {})
        be_acc = be.get("break_even_accuracy", 0.55)
        expected_move = be.get("expected_move_bps", 100)

        label = f"{sym} {tf} (Δ={expected_move:.0f}bps, p*={be_acc:.2f})"
        ax_be.barh(f"{sym} {tf}", be_acc, color=color, alpha=0.7,
                   edgecolor="white", linewidth=0.5, label=label)

    ax_be.set_xlabel("Break-even Accuracy (p*)", fontsize=11)
    ax_be.axvline(x=0.5, color="red", linewidth=0.8, linestyle=":", label="p*=0.5 (random)")
    ax_be.set_xlim(0.45, 0.85)
    ax_be.legend(fontsize=7, loc="lower right")
    ax_be.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    stem = "cost_sensitivity"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths