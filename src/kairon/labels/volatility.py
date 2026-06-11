"""Volatility labels: realized volatility over the horizon.

We compute log returns over the bars in ``[t, t + H]`` and take the
sample standard deviation. This is a *realized* volatility proxy;
for a more accurate ex-ante measure, use a GARCH model's conditional
variance (added in the models layer).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pyarrow as pa

from kairon.labels.schema import LabeledBar, LabeledFrame, LabelKind, LabelSpec


def _realized_vol(prices: Sequence[float]) -> float:
    """Sample standard deviation of log returns (annualization is the
    caller's responsibility)."""
    n = len(prices)
    if n < 3:
        return math.nan
    log_rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, n)]
    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
    return math.sqrt(var)


def make_volatility_labels(
    table: pa.Table,
    *,
    spec: LabelSpec,
    symbol: str,
) -> LabeledFrame:
    """Build a realized-vol label per bar."""
    if spec.kind is not LabelKind.VOLATILITY:
        raise ValueError(f"spec.kind must be VOLATILITY, got {spec.kind}")
    ts_col: Sequence[datetime] = list(table.column("ts").to_pylist())
    close_col: Sequence[float] = list(table.column("close").to_pylist())
    n = len(ts_col)
    horizon_td = timedelta(seconds=spec.horizon_seconds)
    bars: list[LabeledBar] = []
    import bisect

    for i in range(n):
        t = ts_col[i]
        target_ts = t + horizon_td
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=UTC)
        end_idx = bisect.bisect_left(ts_col, target_ts)
        if end_idx - i < 3:
            continue
        window = close_col[i : end_idx + 1]
        rv = _realized_vol(window)
        if math.isnan(rv):
            continue
        bars.append(
            LabeledBar(
                symbol=symbol,
                ts=t,
                horizon=spec.horizon,
                kind=LabelKind.VOLATILITY,
                y=rv,
                meta={"n_bars": len(window)},
            )
        )
    return LabeledFrame(spec=spec, symbol=symbol, bars=tuple(bars))
