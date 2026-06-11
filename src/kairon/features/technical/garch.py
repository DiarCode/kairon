"""GARCH(1,1) conditional variance.

Closed-form recursive computation of conditional variance, walk-forward safe.
No external dependencies required.

The GARCH(1,1) model:
    sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1]

Default parameters are calibrated for crypto (high volatility persistence):
    omega = 1e-6, alpha = 0.1, beta = 0.85

Output columns:
- garch_var: conditional variance sigma^2
- garch_vol: conditional volatility sigma (sqrt of garch_var)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def garch_variance(
    table: pa.Table,
    *,
    omega: float = 1e-6,
    alpha: float = 0.1,
    beta: float = 0.85,
) -> pa.Table:
    """Add GARCH(1,1) conditional variance and volatility columns.

    Walk-forward safe: uses only past returns (r[t-1]) and past variance
    (sigma2[t-1]) to compute current variance.

    Parameters
    ----------
    omega : float
        Long-run variance weight (default 1e-6, crypto-calibrated).
    alpha : float
        Squared return weight (default 0.1, crypto-calibrated).
    beta : float
        Lagged variance weight (default 0.85, crypto-calibrated).
        Note: alpha + beta < 1 is required for stationarity.

    Output columns:
        garch_var: conditional variance (sigma^2)
        garch_vol: conditional volatility (sigma, sqrt of garch_var)
    """
    close = np.array([float(v) for v in table.column("close").to_pylist()], dtype=np.float64)
    n = len(close)

    # Log returns
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = np.log(close[1:] / close[:-1])
    returns = np.where(np.isfinite(returns), returns, 0.0)

    # GARCH(1,1) recursion
    garch_var = np.zeros(n, dtype=np.float64)
    # Initialize with unconditional variance
    if alpha + beta < 1.0:
        unconditional_var = omega / (1.0 - alpha - beta)
    else:
        unconditional_var = np.var(returns) if n > 1 else 1e-4
    garch_var[0] = unconditional_var

    for i in range(1, n):
        garch_var[i] = omega + alpha * returns[i - 1] ** 2 + beta * garch_var[i - 1]
        # Prevent negative variance (shouldn't happen with positive params)
        garch_var[i] = max(garch_var[i], 1e-12)

    # Conditional volatility
    garch_vol = np.sqrt(garch_var)

    out = table
    out = out.append_column("garch_var", pa.array(garch_var, type=pa.float64()))
    out = out.append_column("garch_vol", pa.array(garch_vol, type=pa.float64()))
    return out


__all__ = ["garch_variance"]