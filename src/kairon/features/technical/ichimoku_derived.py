"""Ichimoku-derived signals: cloud position, TK cross, cloud twist, chikou displacement.

The raw Ichimoku indicator (trend.ichimoku) outputs 5 price levels. The ML
model needs *relative position* features that capture the relationship
between price and the cloud structure.

- cloud_position: where price sits relative to the Kumo (cloud)
    > 0 above cloud, < 0 below cloud, normalized by cloud width
- tk_cross: Tenkan-Kijun crossover signal (+1 bullish, -1 bearish)
- cloud_twist: Senkou A/Senkou B crossover (trend change signal)
- chikou_displacement: distance from chikou to price (normalized by ATR)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def ichimoku_derived(table: pa.Table) -> pa.Table:
    """Add Ichimoku-derived signal columns.

    Requires that the raw Ichimoku columns already exist on the table
    (ichi_tenkan, ichi_kijun, ichi_senkou_a, ichi_senkou_b, ichi_chikou).
    These are produced by the ``trend.ichimoku`` feature builder.

    Output columns:
        cloud_position: (close - mid_cloud) / cloud_width
            Values > 1 = above cloud, < -1 = below cloud, in-cloud ≈ 0
        tk_cross: sign(tenkan - kijun) as float (+1 bullish, -1 bearish)
        cloud_twist: sign(senkou_a - senkou_b) as float (+1 bullish, -1 bearish)
        chikou_displacement: (close - chikou) / ATR_14 if available, else raw diff
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    tenkan = np.array([float(v) for v in table.column("ichi_tenkan").to_pylist()], dtype=np.float64)
    kijun = np.array([float(v) for v in table.column("ichi_kijun").to_pylist()], dtype=np.float64)
    senkou_a = np.array([float(v) for v in table.column("ichi_senkou_a").to_pylist()], dtype=np.float64)
    senkou_b = np.array([float(v) for v in table.column("ichi_senkou_b").to_pylist()], dtype=np.float64)
    chikou = np.array([float(v) for v in table.column("ichi_chikou").to_pylist()], dtype=np.float64)
    n = len(close)

    # Try to get ATR for normalization; fall back to raw diff
    atr_available = "atr_14" in table.column_names
    if atr_available:
        atr_14 = np.array([float(v) for v in table.column("atr_14").to_pylist()], dtype=np.float64)
    else:
        atr_14 = np.ones(n, dtype=np.float64)

    # 1. Cloud position: (close - cloud_mid) / cloud_width
    #    cloud_mid = (senkou_a + senkou_b) / 2
    #    cloud_width = |senkou_a - senkou_b|
    cloud_mid = (senkou_a + senkou_b) / 2.0
    cloud_width = np.abs(senkou_a - senkou_b)
    # Avoid division by zero for very thin clouds
    cloud_width = np.where(cloud_width > 1e-10, cloud_width, 1e-10)
    cloud_position = (close - cloud_mid) / cloud_width
    # Clamp extreme values (price far from cloud)
    cloud_position = np.clip(cloud_position, -10.0, 10.0)
    # Replace NaN from warm-up period
    cloud_position = np.where(np.isfinite(cloud_position), cloud_position, 0.0)

    # 2. TK cross: Tenkan-Kijun crossover
    tk_diff = tenkan - kijun
    tk_cross = np.sign(tk_diff).astype(np.float64)
    tk_cross = np.where(np.isfinite(tk_cross), tk_cross, 0.0)

    # 3. Cloud twist: current and previous Senkou A vs B
    sa_sb_diff = senkou_a - senkou_b
    cloud_twist = np.sign(sa_sb_diff).astype(np.float64)
    cloud_twist = np.where(np.isfinite(cloud_twist), cloud_twist, 0.0)

    # 4. Chikou displacement: (close - chikou) / ATR_14
    chikou_disp = (close - chikou) / np.where(atr_14 > 1e-10, atr_14, 1e-10)
    chikou_disp = np.clip(chikou_disp, -10.0, 10.0)
    chikou_disp = np.where(np.isfinite(chikou_disp), chikou_disp, 0.0)

    out = table
    out = out.append_column("cloud_position", pa.array(cloud_position, type=pa.float64()))
    out = out.append_column("tk_cross", pa.array(tk_cross, type=pa.float64()))
    out = out.append_column("cloud_twist", pa.array(cloud_twist, type=pa.float64()))
    out = out.append_column("chikou_displacement", pa.array(chikou_disp, type=pa.float64()))
    return out


__all__ = ["ichimoku_derived"]