"""Per-(symbol, timeframe) edge probe for Phase 4 breadth/ensemble scoping.

Runs the scalping backtest on one (symbol, tf) from the local research store
and prints a compact JSON line of per-setup edges. Designed to be run in
parallel across the universe (the backtest engine is single-threaded per call,
so one process per symbol uses a separate core).

Usage: ``uv run python scripts/analyze_symbol_edge.py <SYMBOL> <TF> [matrix|legacy]``

* ``matrix`` (default): the MEAN_REVERSION_ONLY setup-selection matrix
  (regime-gated, MR-only) — the live edge path.
* ``long-only``: the LONG_ONLY preset (mr_long only, mr_short killed) — the
  Phase 4 data-driven tightening; surfaces the mr_long edge without mr_short
  drag.
* ``legacy``: no matrix (all setups fire) — surfaces momentum/breakout/
  breakdown edges for the ensemble scoping decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kairon.backtest.setup_analysis import analyze_setups
from kairon.data.history_store import read_history
from kairon.live.setup_matrix import LONG_ONLY, MEAN_REVERSION_ONLY
from kairon.live.strategy import ScalpingStrategy


def _strategy_for(mode: str) -> ScalpingStrategy:
    if mode == "matrix":
        return ScalpingStrategy(setup_matrix=MEAN_REVERSION_ONLY)
    if mode == "long-only":
        return ScalpingStrategy(setup_matrix=LONG_ONLY)
    return ScalpingStrategy()


def main() -> None:
    symbol, tf = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "matrix"
    bars = read_history(Path("data"), symbol, tf)
    if bars.num_rows < 500:
        print(json.dumps({"symbol": symbol, "tf": tf, "mode": mode,
                          "bars": bars.num_rows, "edges": []}))
        return
    strat = _strategy_for(mode)
    edges = analyze_setups(bars=bars, strategy=strat, symbol=symbol, timeframe=tf)
    print(json.dumps({
        "symbol": symbol,
        "tf": tf,
        "mode": mode,
        "bars": bars.num_rows,
        "edges": [
            {"setup": e.setup_id, "side": e.side, "n": e.n, "win_rate": e.win_rate,
             "tp": e.hit_tp, "sl": e.hit_sl, "sum_pnl": e.sum_pnl}
            for e in edges if e.n >= 5
        ],
    }))


if __name__ == "__main__":
    main()
