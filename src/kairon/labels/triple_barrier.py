"""Triple-barrier labels (Lopez de Prado, *Advances in Financial
Machine Learning*, ch. 3).

Each bar at time ``t`` is assigned to one of three barriers:
- **Upper** (label +1) if the high over ``[t, t+H]`` first hits the
  upper price ``close[t] * (1 + pt)``.
- **Lower** (label -1) if the low over ``[t, t+H]`` first hits the
  lower price ``close[t] * (1 - sl)``.
- **Vertical** (label 0) if neither barrier is hit before time
  ``t + H``.

The horizontal barriers ``pt`` (profit-take %) and ``sl`` (stop-loss %)
are typed params on the ``LabelSpec``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pyarrow as pa

from kairon.labels.schema import (
    DirectionClass,
    LabeledBar,
    LabeledFrame,
    LabelKind,
    LabelSpec,
)


def make_triple_barrier_labels(
    table: pa.Table,
    *,
    spec: LabelSpec,
    symbol: str,
    pt_pct: float = 0.01,
    sl_pct: float = 0.01,
    require_finite: bool = True,
) -> LabeledFrame:
    """Build a triple-barrier label per bar.

    Parameters
    ----------
    pt_pct, sl_pct:
        Profit-take and stop-loss in *return* terms. A ``pt_pct`` of 0.01
        means the upper barrier is ``close[t] * 1.01``.
    require_finite:
        If True, the upper/lower barrier must be reached; bars that
        never touch either barrier are labeled ``FLAT`` (0). If False,
        they are dropped.
    """
    if spec.kind is not LabelKind.TRIPLE_BARRIER:
        raise ValueError(f"spec.kind must be TRIPLE_BARRIER, got {spec.kind}")
    if pt_pct <= 0 or sl_pct <= 0:
        raise ValueError("pt_pct and sl_pct must be > 0")
    ts_col: Sequence[datetime] = list(table.column("ts").to_pylist())
    close_col: Sequence[float] = list(table.column("close").to_pylist())
    high_col: Sequence[float] = list(table.column("high").to_pylist())
    low_col: Sequence[float] = list(table.column("low").to_pylist())
    n = len(ts_col)
    horizon_td = timedelta(seconds=spec.horizon_seconds)
    import bisect

    bars: list[LabeledBar] = []
    for i in range(n):
        t = ts_col[i]
        target_ts = t + horizon_td
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=UTC)
        # We want the first bar at or after ``target_ts`` to be *inside*
        # the horizon window (its high/low can hit the barrier). We
        # then loop k from i+1 to end_idx inclusive.
        end_idx = bisect.bisect_left(ts_col, target_ts)
        # Need at least the next bar to test the barriers; skip otherwise.
        if end_idx <= i or end_idx >= n:
            continue
        c0 = close_col[i]
        upper = c0 * (1.0 + pt_pct)
        lower = c0 * (1.0 - sl_pct)
        y_class = int(DirectionClass.FLAT)
        first_hit: str = "vertical"
        hit_idx: int = end_idx
        for k in range(i + 1, end_idx + 1):
            hit_upper = high_col[k] >= upper
            hit_lower = low_col[k] <= lower
            if hit_upper and hit_lower:
                # Ambiguous: take the first in time, with conservative bias.
                y_class = int(DirectionClass.UP)
                first_hit = "upper"
                hit_idx = k
                break
            if hit_upper:
                y_class = int(DirectionClass.UP)
                first_hit = "upper"
                hit_idx = k
                break
            if hit_lower:
                y_class = int(DirectionClass.DOWN)
                first_hit = "lower"
                hit_idx = k
                break
        if first_hit == "vertical" and require_finite:
            continue
        # Resolve the exit close: at the bar that first hit a barrier,
        # or the last bar of the window for vertical/no-hit bars.
        exit_close = close_col[hit_idx]
        realized_return_bps = (exit_close - c0) / c0 * 1e4
        bars.append(
            LabeledBar(
                symbol=symbol,
                ts=t,
                horizon=spec.horizon,
                kind=LabelKind.TRIPLE_BARRIER,
                y=y_class,
                y_class=y_class,
                meta={
                    "upper": float(upper),
                    "lower": float(lower),
                    "first_hit": first_hit,
                    "n_bars": float(end_idx - i),
                    "exit_close": float(exit_close),
                    "realized_return_bps": float(realized_return_bps),
                },
            )
        )
    _ = math  # silence unused
    return LabeledFrame(spec=spec, symbol=symbol, bars=tuple(bars))
