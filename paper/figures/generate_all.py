"""Master figure generator: reads results JSON and calls all generators.

Usage::

    uv run python paper/figures/generate_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = Path(__file__).resolve().parent


def _load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    """Generate all publication-quality figures."""
    # Ensure repo root is on sys.path for `paper.*` imports
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    real_results_path = REPO_ROOT / "paper" / "real_results.json"
    ablation_results_path = REPO_ROOT / "paper" / "ablation_results.json"

    if not real_results_path.exists():
        logger.error("Missing {} — run paper/run_real_experiments.py first", real_results_path)
        return 1

    real_data = _load_results(real_results_path)
    ablation_data = _load_results(ablation_results_path) if ablation_results_path.exists() else {}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("candlestick_signals", "fig_candlestick_signals"),
        ("equity_curves", "fig_equity_curves"),
        ("coverage_accuracy", "fig_coverage_accuracy"),
        ("calibration", "fig_calibration"),
        ("regime_detection", "fig_regime_detection"),
        ("cost_sensitivity", "fig_cost_sensitivity"),
        ("ablation_comparison", "fig_ablation_comparison"),
        ("break_even", "fig_break_even"),
        ("confusion_matrices", "fig_confusion_matrices"),
        ("model_comparison", "fig_model_comparison"),
        ("loss_comparison", "fig_loss_comparison"),
    ]

    n_ok = 0
    n_fail = 0
    for label, module_name in generators:
        logger.info("Generating: {} ...", label)
        try:
            mod = __import__(f"paper.figures.{module_name}", fromlist=["generate"])
            fig_paths = mod.generate(real_data, ablation_data, output_dir=FIGURES_DIR)
            for p in fig_paths:
                logger.info("  -> {}", p)
            n_ok += 1
        except Exception as exc:
            logger.error("  FAILED: {} — {}", label, exc)
            n_fail += 1

    logger.info("Generated {}/{} figure groups", n_ok, n_ok + n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())