"""Execute all ablation variants across 6 cells and save results.

Usage::

    uv run python paper/run_ablation.py

Reads real Binance OHLCV, runs 23 ablation variants × 6 cells = 138 runs,
and writes ``paper/ablation_results.json``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from paper.ablation_study import AblationStudy

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ABLATION_PATH: Path = REPO_ROOT / "paper" / "ablation_results.json"


def _ablation_result_to_dict(r) -> dict[str, Any]:
    """Serialize an AblationResult to a JSON-compatible dict."""
    return {
        "variant": r.variant,
        "symbol": r.symbol,
        "timeframe": r.timeframe,
        "accuracy": r.accuracy,
        "cas": r.cas,
        "dsr": r.dsr,
        "sharpe": r.sharpe,
        "max_dd": r.max_dd,
        "brier": r.brier,
        "ece": r.ece,
        "n_trades": r.n_trades,
        "coverage_at_25": r.coverage_at_25,
        "accuracy_at_25": r.accuracy_at_25,
        "delta_cas_vs_full": r.delta_cas_vs_full,
        "delta_sharpe_vs_full": r.delta_sharpe_vs_full,
    }


def run_all_ablations() -> dict[str, Any]:
    """Execute the full ablation grid and save results."""
    all_results: list[dict[str, Any]] = []
    t0 = time.monotonic()

    symbols = ("BTC", "ETH", "SOL")
    timeframes = ("1h", "5m")

    for symbol in symbols:
        for tf in timeframes:
            cell_key = f"{symbol}_{tf}"
            logger.info("=" * 60)
            logger.info("ABLATION CELL: {} {}", symbol, tf)
            logger.info("=" * 60)

            study = AblationStudy(symbol=symbol, timeframe=tf)
            try:
                results = study.run_all()
                for r in results:
                    all_results.append(_ablation_result_to_dict(r))
            except Exception as exc:
                logger.error("ABLATION CELL {} {} FAILED: {}", symbol, tf, exc)

    elapsed = time.monotonic() - t0

    report = {
        "schema_version": "1",
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_source": "Binance public REST API (real OHLCV)",
        "elapsed_seconds": round(elapsed, 1),
        "n_results": len(all_results),
        "results": all_results,
    }

    ABLATION_PATH = REPO_ROOT / "paper" / "ablation_results.json"
    ABLATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )

    # Print summary table
    print("\n" + "=" * 120)
    print("ABLATION STUDY RESULTS")
    print("=" * 120)
    hdr = (
        f"{'Cell':<12} {'Variant':<25} {'Accuracy':>9} {'CAS':>8} "
        f"{'DSR':>8} {'Sharpe':>8} {'DCAS':>8} {'DSharpe':>9} "
        f"{'Cov25':>7} {'Acc25':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        print(
            f"{r['symbol']}_{r['timeframe']:<7} {r['variant']:<25} "
            f"{r['accuracy']:>9.4f} {r['cas']:>8.4f} {r['dsr']:>8.4f} "
            f"{r['sharpe']:>8.4f} {r['delta_cas_vs_full']:>+8.4f} "
            f"{r['delta_sharpe_vs_full']:>+9.4f} "
            f"{r['coverage_at_25']:>7.4f} {r['accuracy_at_25']:>7.4f}"
        )
    print("=" * 120)
    print(f"Total: {len(all_results)} results, {elapsed:.1f}s")
    print(f"Saved to: {ABLATION_PATH}")
    return report


if __name__ == "__main__":
    sys.exit(0 if run_all_ablations() else 1)