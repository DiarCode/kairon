"""Figure 11: Model comparison bar chart.

Accuracy at full coverage and 25% coverage per model variant with error bars.
Includes single models (LR, RF, XGB, LGBM), ensembles (TopK, MetaLabeled),
and baselines (buy-and-hold, random).

Output: model_comparison.pdf + .png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate(
    real_data: dict,
    ablation_data: dict,
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate model comparison bar chart."""
    paths = []
    results = ablation_data.get("results", [])

    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No ablation data for model comparison",
                transform=ax.transAxes, ha="center", va="center", fontsize=14, color="grey")
        for stem in ("model_comparison",):
            for ext in ("pdf", "png"):
                out_path = output_dir / f"{stem}.{ext}"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                paths.append(out_path)
        plt.close(fig)
        return paths

    # Collect model variant data (Tier 2 ablations)
    model_variants = [
        "single_lr", "single_rf", "single_xgb", "single_lgbm",
        "ensemble_no_meta", "majority_vote", "full_system",
        "buy_and_hold", "random_signal",
    ]

    # Average accuracy across all cells for each variant
    full_acc = {}
    acc_25 = {}
    for v in model_variants:
        full_acc[v] = []
        acc_25[v] = []

    for r in results:
        v = r.get("variant", "")
        if v in model_variants:
            full_acc[v].append(r.get("accuracy", 0))
            acc_25[v].append(r.get("accuracy_at_25", 0))

    variants = [v for v in model_variants if full_acc[v]]
    avg_full = [np.mean(full_acc[v]) if full_acc[v] else 0 for v in variants]
    std_full = [np.std(full_acc[v]) if len(full_acc[v]) > 1 else 0 for v in variants]
    avg_25 = [np.mean(acc_25[v]) if acc_25[v] else 0 for v in variants]
    std_25 = [np.std(acc_25[v]) if len(acc_25[v]) > 1 else 0 for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Model Comparison: Accuracy at Full vs 25% Coverage",
                 fontsize=14, fontweight="bold")

    bars1 = ax.bar(x - width / 2, avg_full, width, yerr=std_full,
                   label="Full coverage", color="#627EEA", alpha=0.85,
                   edgecolor="white", linewidth=0.5, capsize=3)
    bars2 = ax.bar(x + width / 2, avg_25, width, yerr=std_25,
                   label="25% coverage", color="#14F195", alpha=0.85,
                   edgecolor="white", linewidth=0.5, capsize=3)

    ax.set_xlabel("Model Variant", fontsize=11)
    ax.set_ylabel("Direction Accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=9)
    ax.axhline(y=0.333, color="red", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Random baseline (33%)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.2, 0.85)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    stem = "model_comparison"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths