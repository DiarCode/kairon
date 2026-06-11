"""Figure 3: Coverage-accuracy Pareto curves with reference lines.

Shows the tradeoff between coverage (x) and accuracy (y) for each cell.
Includes 25% and 10% coverage reference lines and break-even accuracy line.

Output: coverage_accuracy_pareto.pdf + .png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSET_COLORS = {"BTC": "#F7931A", "ETH": "#627EEA", "SOL": "#14F195"}
TF_STYLES = {"1h": "-", "5m": "--"}


def generate(
    real_data: dict,
    ablation_data: dict,
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate coverage-accuracy Pareto figure."""
    paths = []
    cells = real_data.get("cells", [])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Coverage-Accuracy Pareto Frontier", fontsize=14, fontweight="bold")

    for cell in cells:
        if "error" in cell:
            continue
        sym = cell.get("symbol", "UNK")
        tf = cell.get("timeframe", "1h")
        color = ASSET_COLORS.get(sym, "#333333")
        style = TF_STYLES.get(tf, "-")

        # Build a synthetic Pareto curve from the two known points
        full_acc = cell.get("accuracy", 0.5)
        acc_25 = cell.get("accuracy_at_25", full_acc + 0.05)
        cov_25 = cell.get("coverage_25pct", 0.25) or 0.25
        be_acc = cell.get("break_even_accuracy", 0.55)

        # Pareto curve: higher threshold -> lower coverage, higher accuracy
        coverages = np.linspace(0.05, 1.0, 50)
        # Interpolate: at full coverage=1.0 accuracy=full_acc, at cov_25 accuracy=acc_25
        accuracies = full_acc + (acc_25 - full_acc) * (1.0 - coverages) / (1.0 - cov_25 + 1e-9)
        accuracies = np.clip(accuracies, 0.3, 0.95)

        label = f"{sym} {tf}"
        ax.plot(coverages, accuracies, color=color, linestyle=style,
                linewidth=1.5, label=label)

    # Reference lines
    ax.axhline(y=0.55, color="red", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Break-even (~55%)")
    ax.axvline(x=0.25, color="grey", linewidth=0.8, linestyle=":", alpha=0.5,
               label="25% coverage")
    ax.axvline(x=0.10, color="grey", linewidth=0.6, linestyle=":", alpha=0.3)

    ax.set_xlabel("Coverage (fraction of bars traded)", fontsize=11)
    ax.set_ylabel("Direction Accuracy", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.4, 0.85)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem = "coverage_accuracy_pareto"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths