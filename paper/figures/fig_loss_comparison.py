"""Figure 12: Loss function comparison.

Side-by-side bar chart comparing cross_entropy, SharpeLoss, and CostFocalLoss
for each (asset, horizon) cell.

Output: loss_comparison.pdf + .png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOSS_COLORS = {
    "cross_entropy": "#627EEA",
    "loss_sharpe": "#F7931A",
    "loss_cost_focal": "#14F195",
}


def generate(
    real_data: dict,
    ablation_data: dict,
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate loss function comparison figure."""
    paths = []
    results = ablation_data.get("results", [])

    # Collect loss variant data from Tier 3 ablations
    loss_variants = ["full_system", "loss_sharpe", "loss_cost_focal"]
    loss_labels = ["Cross-Entropy", "SharpeLoss", "CostFocalLoss"]
    loss_colors = [LOSS_COLORS["cross_entropy"], LOSS_COLORS["loss_sharpe"], LOSS_COLORS["loss_cost_focal"]]

    # Group by cell
    cells_data: dict[str, dict[str, dict[str, float]]] = {}
    for r in results:
        v = r.get("variant", "")
        if v not in loss_variants:
            continue
        cell_key = f"{r.get('symbol', 'UNK')}_{r.get('timeframe', '1h')}"
        cells_data.setdefault(cell_key, {})[v] = {
            "accuracy": r.get("accuracy", 0),
            "cas": r.get("cas", 0),
            "sharpe": r.get("sharpe", 0),
            "brier": r.get("brier", 0),
        }

    if not cells_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No loss ablation data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=14, color="grey")
        for stem in ("loss_comparison",):
            for ext in ("pdf", "png"):
                out_path = output_dir / f"{stem}.{ext}"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                paths.append(out_path)
        plt.close(fig)
        return paths

    cell_keys = sorted(cells_data.keys())
    n_cells = len(cell_keys)
    n_losses = len(loss_variants)
    bar_width = 0.25

    fig, (ax_acc, ax_cas) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Loss Function Comparison", fontsize=14, fontweight="bold")

    x = np.arange(n_cells)

    for j, (variant, label, color) in enumerate(zip(loss_variants, loss_labels, loss_colors)):
        accs = [cells_data[ck].get(variant, {}).get("accuracy", 0) for ck in cell_keys]
        cass = [cells_data[ck].get(variant, {}).get("cas", 0) for ck in cell_keys]

        ax_acc.bar(x + j * bar_width, accs, bar_width, color=color,
                   label=label, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_cas.bar(x + j * bar_width, cass, bar_width, color=color,
                   label=label, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax_acc.set_xlabel("Cell (Asset_Horizon)", fontsize=10)
    ax_acc.set_ylabel("Direction Accuracy", fontsize=11)
    ax_acc.set_xticks(x + bar_width)
    ax_acc.set_xticklabels(cell_keys, rotation=45, ha="right", fontsize=8)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3, axis="y")
    ax_acc.set_title("Accuracy", fontsize=11)

    ax_cas.set_xlabel("Cell (Asset_Horizon)", fontsize=10)
    ax_cas.set_ylabel("Cost-Adjusted Sharpe", fontsize=11)
    ax_cas.set_xticks(x + bar_width)
    ax_cas.set_xticklabels(cell_keys, rotation=45, ha="right", fontsize=8)
    ax_cas.legend(fontsize=8)
    ax_cas.grid(True, alpha=0.3, axis="y")
    ax_cas.axhline(y=0, color="black", linewidth=0.5)
    ax_cas.set_title("CAS", fontsize=11)

    plt.tight_layout()
    stem = "loss_comparison"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths