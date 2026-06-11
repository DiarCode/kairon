"""Direction labels: binary up/down and 3-class {-1, 0, +1}.

The label at time ``t`` is computed from the close at time
``t + horizon``. The forward-looking point is determined *causally*:
we use the close at the *first* bar whose ts >= t + horizon, never
anything that could leak future information.

The threshold for the "FLAT" class (3-class only) defaults to 0.05% of
the close at ``t``; this is configurable via
``LabelSpec.params["flat_threshold_pct"]``.
"""

from __future__ import annotations

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


def _index_at_or_after(ts_arr: Sequence[datetime], ts: datetime) -> int | None:
    """Return the first index at or after ``ts``, or None if past the end."""
    import bisect

    idx = bisect.bisect_left(ts_arr, ts)
    if idx == len(ts_arr):
        return None
    return idx


def make_direction_labels(
    table: pa.Table,
    *,
    spec: LabelSpec,
    symbol: str,
    flat_threshold_pct: float = 0.0005,
) -> LabeledFrame:
    """Build a 3-class {-1, 0, +1} direction label per bar.

    Parameters
    ----------
    table:
        OHLCV table. The ``ts`` column must be sorted ascending.
    spec:
        ``LabelSpec(kind=DIRECTION, horizon=...)``.
    symbol:
        Canonical symbol for ``LabeledBar`` provenance.
    flat_threshold_pct:
        Half-width of the "FLAT" band around 0% return. Default 5 bps.
    """
    if spec.kind is not LabelKind.DIRECTION:
        raise ValueError(f"spec.kind must be DIRECTION, got {spec.kind}")
    ts_col = list(table.column("ts").to_pylist())
    close_col = list(table.column("close").to_pylist())
    if len(ts_col) != len(close_col):
        raise ValueError("ts and close columns must have equal length")
    if not all(ts_col[i] <= ts_col[i + 1] for i in range(len(ts_col) - 1)):
        raise ValueError("ts column must be sorted ascending")
    n = len(ts_col)
    horizon_td = timedelta(seconds=spec.horizon_seconds)
    bars: list[LabeledBar] = []
    for i in range(n):
        t = ts_col[i]
        target_ts = t + horizon_td
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=UTC)
        future_idx = _index_at_or_after(ts_col, target_ts)
        if future_idx is None:
            # The horizon goes past the end of the table; no label.
            continue
        future_close = close_col[future_idx]
        current_close = close_col[i]
        ret = (future_close - current_close) / current_close
        if ret > flat_threshold_pct:
            y_class = int(DirectionClass.UP)
        elif ret < -flat_threshold_pct:
            y_class = int(DirectionClass.DOWN)
        else:
            y_class = int(DirectionClass.FLAT)
        bars.append(
            LabeledBar(
                symbol=symbol,
                ts=t,
                horizon=spec.horizon,
                kind=LabelKind.DIRECTION,
                y=y_class,
                y_class=y_class,
                meta={"return": ret, "future_close": future_close},
            )
        )
    return LabeledFrame(spec=spec, symbol=symbol, bars=tuple(bars))
