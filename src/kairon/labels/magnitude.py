"""Magnitude labels: log-return at the horizon.

The label is ``log(close[t + H] / close[t])``, a continuous value
that is symmetric around 0 and approximately normal in the small
returns regime. For tail risk, a quantile label can be added on top
in a future version.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pyarrow as pa

from kairon.labels.schema import LabeledBar, LabeledFrame, LabelKind, LabelSpec


def make_magnitude_labels(
    table: pa.Table,
    *,
    spec: LabelSpec,
    symbol: str,
) -> LabeledFrame:
    """Build a continuous log-return label per bar."""
    if spec.kind is not LabelKind.MAGNITUDE:
        raise ValueError(f"spec.kind must be MAGNITUDE, got {spec.kind}")
    ts_col: Sequence[datetime] = list(table.column("ts").to_pylist())
    close_col: Sequence[float] = list(table.column("close").to_pylist())
    n = len(ts_col)
    horizon_td = timedelta(seconds=spec.horizon_seconds)
    bars: list[LabeledBar] = []
    for i in range(n):
        t = ts_col[i]
        target_ts = t + horizon_td
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=UTC)
        import bisect

        idx = bisect.bisect_left(ts_col, target_ts)
        if idx == n:
            continue
        future_close = close_col[idx]
        current_close = close_col[i]
        if current_close <= 0 or future_close <= 0:
            continue
        y = math.log(future_close / current_close)
        bars.append(
            LabeledBar(
                symbol=symbol,
                ts=t,
                horizon=spec.horizon,
                kind=LabelKind.MAGNITUDE,
                y=y,
                meta={"ret": future_close / current_close - 1.0},
            )
        )
    return LabeledFrame(spec=spec, symbol=symbol, bars=tuple(bars))
