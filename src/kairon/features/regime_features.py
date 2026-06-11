"""BOCPD regime probability features.

Exposes the BOCPD detector's continuous-valued state as feature columns
for the ML model. The BOCPD detector is already computed in the regime
module but only the hard label string is used — the probability vector,
run-length statistics, and changepoint detection are discarded.

These features give the model direct access to:
- Soft regime membership probabilities (trending/ranging/volatile/stressed)
- Regime persistence via run-length statistics (short = new regime, long = stable)
- Changepoint detection (structural breaks in the market)

All features are computed walk-forward safe (no look-ahead): the BOCPD
detector processes bars sequentially via ``update()``.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from kairon.features.regime import BOCPDConfig, BOCPDRegimeDetector


def bocpd_regime_probabilities(table: pa.Table, *, config: BOCPDConfig | None = None) -> pa.Table:
    """Add BOCPD regime probability columns.

    Walk-forward safe: processes bars sequentially using ``update()``.
    Each bar's probabilities are based ONLY on data up to and including
    that bar — no future information leaks in.

    Output columns:
        regime_prob_trending: P(trending) from BOCPD run-length posterior
        regime_prob_ranging: P(ranging) from BOCPD run-length posterior
        regime_prob_volatile: P(volatile) from BOCPD run-length posterior
        regime_prob_stressed: P(stressed) from BOCPD run-length posterior

    Requires: ``close``, ``high``, ``low`` columns on the input table.
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
    n = len(close)

    # Compute realized vol and spread from OHLCV (same inputs as the
    # experiment runner uses for BOCPD).
    realized_vol = np.zeros(n, dtype=np.float64)
    realized_vol[1:] = np.abs(np.diff(np.log(close)))
    realized_vol[0] = realized_vol[1] if n > 1 else 0.0

    spread_bps = np.zeros(n, dtype=np.float64)
    spread_bps = (high - low) / close * 10000.0
    # Guard against division by zero for constant-price bars
    spread_bps = np.where(np.isfinite(spread_bps), spread_bps, 0.0)

    detector = BOCPDRegimeDetector(config=config)

    prob_trending = np.zeros(n, dtype=np.float64)
    prob_ranging = np.zeros(n, dtype=np.float64)
    prob_volatile = np.zeros(n, dtype=np.float64)
    prob_stressed = np.zeros(n, dtype=np.float64)

    for i in range(n):
        state = detector.update(float(realized_vol[i]), float(spread_bps[i]))
        probs = state.regime_probabilities
        prob_trending[i] = probs.get("trending", 0.0)
        prob_ranging[i] = probs.get("ranging", 0.0)
        prob_volatile[i] = probs.get("volatile", 0.0)
        prob_stressed[i] = probs.get("stressed", 0.0)

    out = table
    out = out.append_column("regime_prob_trending", pa.array(prob_trending, type=pa.float64()))
    out = out.append_column("regime_prob_ranging", pa.array(prob_ranging, type=pa.float64()))
    out = out.append_column("regime_prob_volatile", pa.array(prob_volatile, type=pa.float64()))
    out = out.append_column("regime_prob_stressed", pa.array(prob_stressed, type=pa.float64()))
    return out


def bocpd_run_length(table: pa.Table, *, config: BOCPDConfig | None = None) -> pa.Table:
    """Add BOCPD run-length statistics as feature columns.

    Walk-forward safe: processes bars sequentially using ``update()``.

    Output columns:
        bocpd_run_length_mean: Expected run-length from BOCPD posterior
            (high = stable regime, low = recent changepoint)
        bocpd_run_length_map: MAP (most probable) run-length
            (integer count of bars since last regime change)

    Requires: ``close``, ``high``, ``low`` columns on the input table.
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
    n = len(close)

    realized_vol = np.zeros(n, dtype=np.float64)
    realized_vol[1:] = np.abs(np.diff(np.log(close)))
    realized_vol[0] = realized_vol[1] if n > 1 else 0.0

    spread_bps = (high - low) / close * 10000.0
    spread_bps = np.where(np.isfinite(spread_bps), spread_bps, 0.0)

    detector = BOCPDRegimeDetector(config=config)

    rl_mean = np.zeros(n, dtype=np.float64)
    rl_map = np.zeros(n, dtype=np.float64)

    for i in range(n):
        state = detector.update(float(realized_vol[i]), float(spread_bps[i]))
        rl_mean[i] = state.run_length_mean
        rl_map[i] = float(state.run_length_map)

    out = table
    out = out.append_column("bocpd_run_length_mean", pa.array(rl_mean, type=pa.float64()))
    out = out.append_column("bocpd_run_length_map", pa.array(rl_map, type=pa.float64()))
    return out


def bocpd_changepoint(table: pa.Table, *, config: BOCPDConfig | None = None) -> pa.Table:
    """Add BOCPD changepoint detection as a binary feature column.

    A changepoint is detected at bar ``i`` when the MAP run-length
    drops significantly below the previous bar's MAP run-length, indicating
    a structural break in the market regime.

    Walk-forward safe: processes bars sequentially using ``update()``.

    Output columns:
        bocpd_is_changepoint: 1 if a changepoint was detected at this bar,
            0 otherwise. Uses a simple heuristic: MAP run-length drops
            by more than 50% from the previous bar.

    Requires: ``close``, ``high``, ``low`` columns on the input table.
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    high = np.array([float(v) for v in table.column("high").to_pylist()], dtype=np.float64)
    low = np.array([float(v) for v in table.column("low").to_pylist()], dtype=np.float64)
    n = len(close)

    realized_vol = np.zeros(n, dtype=np.float64)
    realized_vol[1:] = np.abs(np.diff(np.log(close)))
    realized_vol[0] = realized_vol[1] if n > 1 else 0.0

    spread_bps = (high - low) / close * 10000.0
    spread_bps = np.where(np.isfinite(spread_bps), spread_bps, 0.0)

    detector = BOCPDRegimeDetector(config=config)

    is_cp = np.zeros(n, dtype=np.float64)
    prev_rl_map = 0.0

    for i in range(n):
        state = detector.update(float(realized_vol[i]), float(spread_bps[i]))
        cur_rl_map = float(state.run_length_map)
        # Changepoint: MAP run-length drops by more than 50% from previous
        if i > 0 and prev_rl_map > 5.0 and cur_rl_map < prev_rl_map * 0.5:
            is_cp[i] = 1.0
        prev_rl_map = cur_rl_map

    return table.append_column("bocpd_is_changepoint", pa.array(is_cp, type=pa.float64()))


__all__ = [
    "bocpd_regime_probabilities",
    "bocpd_run_length",
    "bocpd_changepoint",
]