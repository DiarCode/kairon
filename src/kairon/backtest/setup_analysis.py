"""Setup-selection analysis — per-setup edge breakdown across the research store.

Runs :func:`kairon.backtest.scalping_engine.run_scalp_backtest` over each
(symbol, timeframe) in the research parquet store and aggregates the recorded
trades by ``setup_id`` to produce the edge table that drives the setup-selection
matrix (Phase 2). This is the research tool behind
``memory/scalping-setup-edge-findings``: it surfaces which setups have positive
expectancy and which to kill.

No network — reads the local ``data/history`` store. Deterministic given the
store contents.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from kairon.backtest.scalping_engine import ScalpBacktestConfig, run_scalp_backtest
from kairon.data.history_store import read_history

__all__ = ["SetupEdge", "analyze_setups", "analyze_universe"]


@dataclass(frozen=True, slots=True)
class SetupEdge:
    """Aggregated edge for one (symbol, timeframe, setup_id, side) bucket."""

    symbol: str
    timeframe: str
    setup_id: str
    side: str
    n: int
    wins: int
    hit_tp: int
    hit_sl: int
    sum_pnl: float
    avg_rr: float
    win_rate: float

    @property
    def expectancy(self) -> float:
        return self.sum_pnl / self.n if self.n else 0.0


def _config_from(**overrides: Any) -> ScalpBacktestConfig:
    base: dict = {
        "bankroll_start": 10.0, "leverage": 10.0, "allocation": 1.0,
        "risk_per_trade": 0.025, "buffer_bars": 200, "min_qty": 0.1,
        "qty_step": 0.1, "max_drawdown": None,
    }
    base.update(overrides)
    return ScalpBacktestConfig(**base)


def analyze_setups(
    bars: pa.Table, *, strategy: Any, symbol: str, timeframe: str,
    config: ScalpBacktestConfig | None = None,
) -> list[SetupEdge]:
    """Backtest one (symbol, tf) and aggregate trades by setup_id."""
    cfg = config or _config_from()
    res = run_scalp_backtest(bars=bars, strategy=strategy, symbol=symbol, config=cfg)
    buckets: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for t in res.trades:
        buckets[(t.setup_id, t.side.value)].append(t)
    edges: list[SetupEdge] = []
    for (setup_id, side), trades in sorted(buckets.items()):
        n = len(trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)
        hit_tp = sum(1 for t in trades if t.hit_tp)
        hit_sl = sum(1 for t in trades if t.hit_sl)
        sum_pnl = float(sum(t.net_pnl for t in trades))
        avg_rr = float(sum(t.realized_rr for t in trades) / n) if n else 0.0
        edges.append(SetupEdge(
            symbol=symbol, timeframe=timeframe, setup_id=setup_id, side=side,
            n=n, wins=wins, hit_tp=hit_tp, hit_sl=hit_sl, sum_pnl=sum_pnl,
            avg_rr=avg_rr, win_rate=wins / n if n else 0.0,
        ))
    return edges


def analyze_universe(
    *,
    root: Path,
    symbols: list[str],
    timeframes: list[str],
    strategy: Any,
    config: ScalpBacktestConfig | None = None,
) -> list[SetupEdge]:
    """Analyze every (symbol, tf) in the research store; skips missing/empty."""
    out: list[SetupEdge] = []
    for sym in symbols:
        for tf in timeframes:
            bars = read_history(root, sym, tf)
            if bars.num_rows < 500:
                continue
            out.extend(analyze_setups(
                bars=bars, strategy=strategy, symbol=sym, timeframe=tf, config=config,
            ))
    return out


def edge_table(edges: list[SetupEdge]) -> str:
    """Render the edge breakdown as a fixed-width text table for reports."""
    header = (
        f"{'symbol':<15}{'tf':<4}{'setup':<16}{'side':<6}"
        f"{'n':>5}{'win%':>6}{'TP':>5}{'SL':>5}{'avgR':>8}{'sumPnL':>10}"
    )
    lines = [header, "-" * len(header)]
    for e in sorted(edges, key=lambda x: (x.symbol, x.timeframe, x.setup_id, x.side)):
        lines.append(
            f"{e.symbol:<15}{e.timeframe:<4}{e.setup_id:<16}{e.side:<6}"
            f"{e.n:>5}{e.win_rate*100:>6.0f}{e.hit_tp:>5}{e.hit_sl:>5}"
            f"{e.avg_rr:>8.2f}{e.sum_pnl:>10.3f}"
        )
    return "\n".join(lines)
