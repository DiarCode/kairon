"""Figures 7-8: Ablation study comparison charts.

Figure 7: Horizontal bar chart — y = ablation variant, x = delta CAS.
Figure 8: Radar chart — dimensions = key metrics, lines = top variants.

Output: ablation_cas_delta.pdf + .png, ablation_radar.pdf + .png
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
    """Generate ablation comparison figures."""
    paths = []
    results = ablation_data.get("results", [])

    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No ablation data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=14, color="grey")
        ax.set_title("Ablation Study (no data)")
        for stem in ("ablation_cas_delta", "ablation_radar"):
            for ext in ("pdf", "png"):
                out_path = output_dir / f"{stem}.{ext}"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                paths.append(out_path)
        plt.close(fig)
        return paths

    # --- Figure 7: Horizontal bar chart of delta CAS ---
    # Average deltas across all cells for each variant
    variant_deltas: dict[str, list[float]] = {}
    variant_sharpe_deltas: dict[str, list[float]] = {}
    for r in results:
        v = r.get("variant", "unknown")
        variant_deltas.setdefault(v, []).append(r.get("delta_cas_vs_full", 0.0))
        variant_sharpe_deltas.setdefault(v, []).append(r.get("delta_sharpe_vs_full", 0.0))

    avg_deltas = {v: np.mean(ds) for v, ds in variant_deltas.items()}
    avg_sharpe_deltas = {v: np.mean(ds) for v, ds in variant_sharpe_deltas.items()}

    # Sort by delta CAS
    sorted_variants = sorted(avg_deltas.keys(), key=lambda v: avg_deltas[v])
    variants = [v for v in sorted_variants if v != "full_system"]
    deltas = [avg_deltas[v] for v in variants]
    colors = ["#CC0000" if d < 0 else "#00875A" for d in deltas]

    fig, (ax_cas, ax_sharpe) = plt.subplots(1, 2, figsize=(14, max(6, len(variants) * 0.4)))
    fig.suptitle("Ablation Study: Component Impact", fontsize=14, fontweight="bold")

    ax_cas.barh(variants, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax_cas.axvline(x=0, color="black", linewidth=0.8)
    ax_cas.set_xlabel("Δ CAS vs Full System", fontsize=11)
    ax_cas.set_title("Cost-Adjusted Sharpe Delta", fontsize=11)
    ax_cas.grid(True, alpha=0.3, axis="x")

    sharpe_deltas = [avg_sharpe_deltas.get(v, 0.0) for v in variants]
    sharpe_colors = ["#CC0000" if d < 0 else "#00875A" for d in sharpe_deltas]
    ax_sharpe.barh(variants, sharpe_deltas, color=sharpe_colors, edgecolor="white", linewidth=0.5)
    ax_sharpe.axvline(x=0, color="black", linewidth=0.8)
    ax_sharpe.set_xlabel("Δ Sharpe vs Full System", fontsize=11)
    ax_sharpe.set_title("Sharpe Ratio Delta", fontsize=11)
    ax_sharpe.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    stem = "ablation_cas_delta"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    # --- Figure 8: Radar chart ---
    # Dimensions: accuracy, cas, sharpe, brier (inverted), ece (inverted)
    dimensions = ["Accuracy", "CAS", "Sharpe", "Calibration", "Coverage@25"]
    n_dims = len(dimensions)

    # Get top variants by absolute CAS delta
    key_variants = ["full_system"] + [v for v in sorted_variants[-5:] if v != "full_system"][-5:]
    radar_colors = ["#333333", "#F7931A", "#627EEA", "#14F195", "#E91E63", "#9C27B0"]

    # Normalize metrics to 0-1 scale for radar
    all_metrics = {v: {} for v in key_variants}
    for r in results:
        v = r.get("variant", "")
        if v not in key_variants:
            continue
        all_metrics[v].setdefault("accuracy", []).append(r.get("accuracy", 0))
        all_metrics[v].setdefault("cas", []).append(r.get("cas", 0))
        all_metrics[v].setdefault("sharpe", []).append(r.get("sharpe", 0))
        all_metrics[v].setdefault("brier_inv", []).append(1 - min(r.get("brier", 0.3), 1.0))
        all_metrics[v].setdefault("ece_inv", []).append(1 - min(r.get("ece", 0.1), 1.0))

    # Find global min/max for normalization
    metric_ranges = {}
    for dim_key in ["accuracy", "cas", "sharpe", "brier_inv", "ece_inv"]:
        all_vals = []
        for v in key_variants:
            all_vals.extend(all_metrics[v].get(dim_key, [0]))
        if all_vals:
            metric_ranges[dim_key] = (min(all_vals), max(all_vals))
        else:
            metric_ranges[dim_key] = (0, 1)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for i, v in enumerate(key_variants):
        color = radar_colors[i % len(radar_colors)]
        vals = []
        for dim_key in ["accuracy", "cas", "sharpe", "brier_inv", "ece_inv"]:
            raw = np.mean(all_metrics[v].get(dim_key, [0]))
            lo, hi = metric_ranges[dim_key]
            norm = (raw - lo) / (hi - lo + 1e-9)
            vals.append(norm)
        vals += vals[:1]

        ax.plot(angles, vals, "o-", linewidth=1.5, color=color, label=v)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Ablation Radar Chart", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    stem = "ablation_radar"
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths