"""Walk-forward / out-of-sample hold-out for the SOL mr_long edge (Step 1).

Temporal-split the 8-week SOL research store (train = first 75%, test = last
25% — the most-recent bars are held out, mimicking "the matrix was chosen on
older bars; does it still win on newer ones?"), then run the scalping backtest
with the LONG_ONLY and MEAN_REVERSION_ONLY matrices on **each split separately**
and emit a compact JSON summary of the mr_long / mr_short edge per split.

This is the one out-of-sample check the existing OHLCV store allows. It is
deterministic and offline (no network). The split is strictly temporal (the
store is sorted ascending by ts) so there is no leakage.

Usage: ``uv run python scripts/walkforward_sol.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from kairon.backtest.setup_analysis import analyze_setups
from kairon.data.history_store import read_history
from kairon.live.setup_matrix import LONG_ONLY, MEAN_REVERSION_ONLY
from kairon.live.strategy import ScalpingStrategy

SYMBOL = "SOL-USDT-PERP"
TIMEFRAMES = ("5m", "15m")
TRAIN_FRACTION = 0.75  # first 75% train, last 25% test (most-recent held out)
MATRICES = {
    "long-only": LONG_ONLY,
    "mean-reversion": MEAN_REVERSION_ONLY,
}


def _edge_row(edges: list, setup_id: str, side: str) -> dict | None:
    """Pick the edge bucket matching (setup_id, side); None if absent."""
    for e in edges:
        if e.setup_id == setup_id and e.side == side:
            return {
                "setup": e.setup_id, "side": e.side, "n": e.n,
                "win_rate": round(e.win_rate, 4), "hit_tp": e.hit_tp,
                "hit_sl": e.hit_sl, "sum_pnl": round(e.sum_pnl, 4),
                "avg_rr": round(e.avg_rr, 3),
            }
    return None


def _split_ts(table: pa.Table, split_idx: int) -> dict:
    """Boundary timestamps so the report can prove the split is temporal."""
    last_train = table.column("ts")[split_idx - 1].as_py()
    first_test = table.column("ts")[split_idx].as_py()
    return {
        "last_train_ts": last_train.isoformat(),
        "first_test_ts": first_test.isoformat(),
    }


def main() -> None:
    root = Path("data")
    out: dict = {
        "symbol": SYMBOL, "train_fraction": TRAIN_FRACTION, "timeframes": {},
    }
    for tf in TIMEFRAMES:
        table = read_history(root, SYMBOL, tf)
        n = table.num_rows
        if n < 800:
            out["timeframes"][tf] = {"error": f"too few bars ({n})"}
            continue
        split_idx = int(n * TRAIN_FRACTION)
        train = table.slice(0, split_idx)
        test = table.slice(split_idx)
        tf_out: dict = {
            "total_bars": n, "train_bars": train.num_rows, "test_bars": test.num_rows,
            "split": _split_ts(table, split_idx), "matrices": {},
        }
        for mode, matrix in MATRICES.items():
            strat = ScalpingStrategy(setup_matrix=matrix)
            train_edges = analyze_setups(bars=train, strategy=strat, symbol=SYMBOL, timeframe=tf)
            strat = ScalpingStrategy(setup_matrix=matrix)  # fresh state per run
            test_edges = analyze_setups(bars=test, strategy=strat, symbol=SYMBOL, timeframe=tf)
            tf_out["matrices"][mode] = {
                "train": {
                    "mr_long": _edge_row(train_edges, "mr_long", "long"),
                    "mr_short": _edge_row(train_edges, "mr_short", "short"),
                },
                "test": {
                    "mr_long": _edge_row(test_edges, "mr_long", "long"),
                    "mr_short": _edge_row(test_edges, "mr_short", "short"),
                },
            }
        out["timeframes"][tf] = tf_out
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
