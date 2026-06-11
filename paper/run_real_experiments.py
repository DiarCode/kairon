"""Execute all 6 real-data experiment cells (3 assets × 2 horizons).

Usage::

    uv run python paper/run_real_experiments.py

Reads real Binance OHLCV from ``data/raw/ohlcv/binance/``, runs the
full pipeline per cell, and writes ``paper/real_results.json``.

Also computes baselines (buy-and-hold, random signal) for each cell.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS
from kairon.backtest.metrics import BARS_PER_YEAR_1H, BARS_PER_YEAR_5M, summarize
from kairon.data.io import DataPaths, read_ohlcv
from kairon.data.symbols import CryptoVenue, crypto_spot
from paper.real_data_experiment import (
    ASSET_CONFIGS,
    HORIZON_CONFIGS,
    RealDataExperiment,
    _brier_score,
    _compute_cas,
    _compute_dsr,
    _ece,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_PATH: Path = REPO_ROOT / "paper" / "real_results.json"


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------
def _buy_and_hold_baseline(
    prices: np.ndarray,
    *,
    bars_per_year: int,
    cost_model=DEFAULT_CRYPTO_COSTS,
) -> dict[str, float]:
    """Buy-and-hold: always long from bar 0, one round-trip cost at exit."""
    n = int(prices.size)
    if n < 2:
        return {"sharpe": 0.0, "return": 0.0, "max_dd": 0.0}
    rets = np.diff(prices) / prices[:-1]
    # Subtract one round-trip cost amortized over N bars
    amortized_cost_per_bar = float(cost_model.round_trip_bps) / 1e4 / n
    rets_net = rets - amortized_cost_per_bar
    equity = np.concatenate([np.array([1.0]), np.cumprod(1.0 + rets_net)])
    perf = summarize(equity, bars_per_year=bars_per_year)
    return {
        "sharpe": float(perf.sharpe),
        "return": float(perf.total_return),
        "max_dd": float(perf.max_drawdown),
    }


def _random_baseline(
    prices: np.ndarray,
    *,
    bars_per_year: int,
    seed: int = 20260608,
    n_trials: int = 20,
    cost_model=DEFAULT_CRYPTO_COSTS,
) -> dict[str, float]:
    """Random signal baseline: average over n_trials random signal streams."""
    rng = np.random.default_rng(seed)
    n = int(prices.size)
    if n < 2:
        return {"sharpe": 0.0, "accuracy": 0.5, "max_dd": 0.0}

    sharpes = []
    accs = []
    log_returns = np.diff(np.log(prices.astype(np.float64, copy=False)))
    # True direction: up=1, down=0
    true_dir = (log_returns > 0).astype(np.int64)

    for trial in range(n_trials):
        sig = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n)
        aligned = sig[:-1].astype(np.float64, copy=False)
        pnl = aligned * log_returns
        cost_per_bar = float(cost_model.round_trip_bps) / 1e4 / 2.0
        pnl = pnl - cost_per_bar
        if pnl.std(ddof=0) > 0:
            sharpes.append(float(pnl.mean() / pnl.std(ddof=0) * math.sqrt(bars_per_year)))
        else:
            sharpes.append(0.0)
        # Accuracy: does signal match direction?
        sig_dir = (sig[:-1] > 0).astype(np.int64)
        accs.append(float((sig_dir == true_dir).mean()))

    return {
        "sharpe": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes, ddof=0)),
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=0)),
        "max_dd": 0.0,  # not computed for random
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_all_cells() -> dict[str, Any]:
    """Execute all 6 experiment cells and return the results dict."""
    cells: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    t0 = time.monotonic()

    for symbol_name in ("BTC", "ETH", "SOL"):
        for tf in ("1h", "5m"):
            cell_key = f"{symbol_name}_{tf}"
            logger.info("=" * 60)
            logger.info("CELL: {} {}", symbol_name, tf)
            logger.info("=" * 60)

            # --- Main experiment ---
            exp = RealDataExperiment(symbol=symbol_name, timeframe=tf)
            try:
                result = exp.run_and_save(output_path=RESULTS_PATH)
                cells.append({
                    "cell_key": cell_key,
                    "symbol": result.symbol,
                    "timeframe": result.timeframe,
                    "accuracy": result.accuracy,
                    "cas": result.cas,
                    "dsr": result.dsr,
                    "pbo": result.pbo,
                    "sharpe": result.sharpe,
                    "max_dd": result.max_dd,
                    "brier": result.brier,
                    "ece": result.ece,
                    "n_trades": result.n_trades,
                    "n_bars": result.n_bars,
                    "break_even_accuracy": result.break_even.get("break_even_accuracy", 0.0),
                    "coverage_at_25": result.coverage.get("t_at_25pct_accuracy", 0.0),
                    "coverage_25pct": result.coverage.get("t_at_25pct_coverage_actual", 0.0),
                })
            except Exception as exc:
                logger.error("FAILED: {} {} — {}", symbol_name, tf, exc)
                cells.append({
                    "cell_key": cell_key,
                    "symbol": symbol_name,
                    "timeframe": tf,
                    "error": str(exc),
                })
                continue

            # --- Baselines on same data ---
            cfg = ASSET_CONFIGS[symbol_name]
            sym = crypto_spot(cfg["base"], cfg["quote"], CryptoVenue.BINANCE)
            hcfg = HORIZON_CONFIGS[tf]
            try:
                table = read_ohlcv(
                    symbol=sym,
                    venue="binance",
                    timeframe=tf,
                    paths=DataPaths.default(),
                )
                prices = np.array(table.column("close").to_pylist(), dtype=np.float64)

                bh = _buy_and_hold_baseline(prices, bars_per_year=hcfg["bars_per_year"])
                rnd = _random_baseline(prices, bars_per_year=hcfg["bars_per_year"])

                baselines.append({
                    "cell_key": cell_key,
                    "buy_and_hold": bh,
                    "random": rnd,
                })
            except Exception as exc:
                logger.warning("Baseline failed for {} {}: {}", symbol_name, tf, exc)

    elapsed = time.monotonic() - t0

    report = {
        "schema_version": "1",
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_source": "Binance public REST API (real OHLCV)",
        "elapsed_seconds": round(elapsed, 1),
        "n_cells": len(cells),
        "cells": cells,
        "baselines": baselines,
    }

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )

    # Print summary
    print("\n" + "=" * 100)
    print("REAL DATA EXPERIMENT RESULTS")
    print("=" * 100)
    hdr = (
        f"{'Cell':<12} {'Accuracy':>9} {'CAS':>8} {'DSR':>8} {'PBO':>8} "
        f"{'Sharpe':>8} {'MaxDD':>8} {'Brier':>7} {'ECE':>6} {'NTrades':>9} "
        f"{'BE_Acc':>7} {'Cov25':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        if "error" in c:
            print(f"{c['cell_key']:<12} ERROR: {c['error']}")
            continue
        print(
            f"{c['cell_key']:<12} {c['accuracy']:>9.4f} {c['cas']:>8.4f} "
            f"{c['dsr']:>8.4f} {c['pbo']:>8.4f} {c['sharpe']:>8.4f} "
            f"{c['max_dd']:>8.4f} {c['brier']:>7.4f} {c['ece']:>6.4f} "
            f"{c['n_trades']:>9} {c['break_even_accuracy']:>7.4f} "
            f"{c.get('coverage_25pct', 0.0):>7.4f}"
        )

    # Baselines summary
    if baselines:
        print("\n" + "-" * 100)
        print("BASELINES")
        print("-" * 100)
        for b in baselines:
            bh = b.get("buy_and_hold", {})
            rnd = b.get("random", {})
            print(
                f"{b['cell_key']:<12}  BH_Sharpe={bh.get('sharpe', 0):.4f}  "
                f"RND_Sharpe={rnd.get('sharpe', 0):.4f}  "
                f"RND_Acc={rnd.get('accuracy', 0):.4f}"
            )

    print("=" * 100)
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Elapsed: {elapsed:.1f}s")
    return report


if __name__ == "__main__":
    sys.exit(0 if run_all_cells() else 1)