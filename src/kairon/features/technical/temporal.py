"""Temporal features: cyclical hour-of-day and trading session indicators.

Crypto markets trade 24/7, but intraday patterns persist because of
overlapping global trading sessions. The "Bitcoin Never Sleeps" paper
documented that 21:00-23:00 UTC has structurally higher returns.

- hour_sin/cos: cyclical encoding of UTC hour (preserves circularity)
- session_asia/europe/us: binary indicators for major trading sessions

These features provide conditioning context that changes the interpretation
of all other features (e.g., RSI 70 during peak hours means something
different than RSI 70 during off-peak).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def hour_of_day(table: pa.Table) -> pa.Table:
    """Add cyclical hour-of-day and trading session indicator columns.

    Requires a ``ts`` column (UTC timestamp) on the input table.

    Output columns:
        hour_sin: sin(2π × hour / 24) — cyclical hour encoding
        hour_cos: cos(2π × hour / 24) — cyclical hour encoding
        session_asia: 1 during Asian session (0:00-8:00 UTC), 0 otherwise
        session_europe: 1 during European session (7:00-16:00 UTC), 0 otherwise
        session_us: 1 during US session (13:00-22:00 UTC), 0 otherwise

    Note: sessions overlap (7-8 UTC is both Asia and Europe;
    13-16 UTC is both Europe and US). This is intentional — the
    overlapping periods are when global liquidity is highest.
    """
    n = table.num_rows

    # Extract hour from timestamp column
    if "ts" not in table.column_names:
        raise ValueError("hour_of_day requires a 'ts' (timestamp) column on the input table")

    ts_col = table.column("ts")
    hours = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ts_val = ts_col[i].as_py()
        if hasattr(ts_val, "hour"):
            hours[i] = float(ts_val.hour)
        else:
            # Fallback: try to extract from timestamp
            hours[i] = float(np.floor(ts_val / 3_600_000_000) % 24)

    # Cyclical encoding
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)

    # Trading session indicators (with overlaps)
    session_asia = np.where((hours >= 0) & (hours < 8), 1.0, 0.0)
    session_europe = np.where((hours >= 7) & (hours < 16), 1.0, 0.0)
    session_us = np.where((hours >= 13) & (hours < 22), 1.0, 0.0)

    out = table
    out = out.append_column("hour_sin", pa.array(hour_sin, type=pa.float64()))
    out = out.append_column("hour_cos", pa.array(hour_cos, type=pa.float64()))
    out = out.append_column("session_asia", pa.array(session_asia, type=pa.float64()))
    out = out.append_column("session_europe", pa.array(session_europe, type=pa.float64()))
    out = out.append_column("session_us", pa.array(session_us, type=pa.float64()))
    return out


__all__ = ["hour_of_day"]