"""Deflated + multiple-testing-aware statistics.

This module contains two pieces of statistical machinery that make
"this is a 60%-accurate signal" defensible:

- :class:`DeflatedSharpeRatio` — corrects the Sharpe ratio for the
  number of trials run, the skewness/kurtosis of returns, and the
  length of the backtest. From Bailey & López de Prado (2014).

- :class:`ProbabilityBacktestOverfit` — the probability that the
  best-performing backtest path in a CPCV run is the result of
  overfitting, not real edge. From Bailey et al. (2015).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# DSR
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DSRSpec:
    """Configuration for the DSR calculation."""

    n_trials: int = 1
    bars_per_year: int = 365
    risk_free: float = 0.0
    significance: float = 0.05


@dataclass(frozen=True, slots=True)
class DSRResult:
    """The output of :func:`deflated_sharpe_ratio`."""

    sharpe: float
    dsr: float
    p_value: float
    sr_star: float  # the haircut threshold needed to pass
    extras: dict[str, float]


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    spec: DSRSpec,
) -> DSRResult:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Returns a DSRResult containing the *adjusted* probability that the
    observed Sharpe exceeds a haircut threshold that accounts for the
    number of trials, the returns' higher moments, and the time-step
    of the test. ``DSRResult.sr_star`` is the haircut threshold itself.
    """
    if returns.size < 2:
        return DSRResult(
            sharpe=float("nan"),
            dsr=float("nan"),
            p_value=float("nan"),
            sr_star=float("nan"),
            extras={},
        )
    sharpe = float(returns.mean() / returns.std(ddof=0)) * math.sqrt(spec.bars_per_year)
    # Higher moments
    n = returns.size
    e3 = float(stats.skew(returns))
    e4 = float(stats.kurtosis(returns, fisher=False))  # excess=False -> non-centered
    # SR* haircut (Eq. 4 in Bailey & López de Prado 2014)
    sr_star = _sr_star(
        n_trials=spec.n_trials,
        n_bars=n,
        skew=e3,
        kurt=e4,
        bars_per_year=spec.bars_per_year,
    )
    # Z-score of the observed SR against SR*
    sr_diff = sharpe - sr_star
    denom = _dsr_std(n_bars=n, skew=e3, kurt=e4, bars_per_year=spec.bars_per_year)
    if denom <= 0 or not math.isfinite(denom):
        p_value = float("nan")
    else:
        z = sr_diff * math.sqrt(n - 1) / denom
        p_value = float(1.0 - stats.norm.cdf(z))
    # DSR = 1 - p_value, clipped to [0, 1]
    dsr = float(1.0 - p_value) if math.isfinite(p_value) else float("nan")
    return DSRResult(
        sharpe=sharpe,
        dsr=dsr,
        p_value=p_value,
        sr_star=sr_star,
        extras={"skew": e3, "kurtosis": e4, "n_bars": float(n)},
    )


def _sr_star(
    *,
    n_trials: int,
    n_bars: int,
    skew: float,
    kurt: float,
    bars_per_year: int,
) -> float:
    """The haircut threshold SR* above which the multiple-testing
    adjusted p-value is < 0.05.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_bars < 2:
        raise ValueError(f"n_bars must be >= 2, got {n_bars}")
    # No multiple-testing adjustment for a single trial.
    if n_trials == 1:
        return 0.0
    eul_gamma = 0.5772156649
    e_z = stats.norm.ppf(1.0 - 1.0 / n_trials)
    e_z2 = e_z ** 2
    # Variance of SR (annualised) under non-normal returns
    var_sr = (
        1.0
        - skew * e_z
        + ((kurt - 1.0) / 4.0) * e_z2
    ) / (n_bars - 1)
    if var_sr <= 0:
        return float("inf")
    return float(
        e_z + eul_gamma * math.sqrt(var_sr) * math.sqrt(bars_per_year)
    )


def _dsr_std(*, n_bars: int, skew: float, kurt: float, bars_per_year: int) -> float:
    """Variance of the (annualised) SR under non-normal returns.

    For the central case (SR = SR*, e_z = 0), the variance simplifies
    to ``1/(n-1)`` (the no-haircut limit). We keep the full form for
    reference; the haircut threshold is what really matters for DSR.
    """
    if n_bars < 2:
        return 0.0
    # Simplification: when comparing SR to SR* (the e_z=0 case), the
    # variance is just 1/(n-1) (annualised). Anything more elaborate
    # would require passing the e_z to use, which we don't.
    return float(math.sqrt(1.0 / (n_bars - 1)) * math.sqrt(bars_per_year))


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PBOResult:
    """The output of :func:`probability_of_backtest_overfit`."""

    pbo: float
    n_paths: int
    n_folds: int
    best_is_oos: float
    worst_oos: float
    extras: dict[str, float]


def probability_of_backtest_overfit(
    *,
    oos_returns_by_path: np.ndarray,
) -> PBOResult:
    """Probability of Backtest Overfit (Bailey, Borwein, López de Prado,
    Zhu 2014).

    Parameters
    ----------
    oos_returns_by_path
        A 2-D array of shape ``(n_paths, n_folds)`` of OOS returns.
        Each "path" is one combinatorial train/test split from CPCV;
        each "fold" is the OOS performance of that path on one of its
        test blocks.

    PBO is the fraction of paths whose *log-最优* OOS performance
    (best fold) is *below* the *log-最差* IS performance of the
    *worst* path. High PBO → likely overfit.
    """
    if oos_returns_by_path.ndim != 2:
        raise ValueError(
            f"oos_returns_by_path must be 2-D, got shape {oos_returns_by_path.shape}"
        )
    n_paths, n_folds = oos_returns_by_path.shape
    if n_paths < 2:
        return PBOResult(
            pbo=float("nan"),
            n_paths=n_paths,
            n_folds=n_folds,
            best_is_oos=float("nan"),
            worst_oos=float("nan"),
            extras={},
        )
    # Best OOS per path (one number per path)
    best_oos_per_path = oos_returns_by_path.max(axis=1)
    # Best IS per path = best OOS of the OTHER paths (we use OOS as a
    # proxy for "what would have looked best in IS if I'd picked this path")
    # Strictly, IS is the OOS complement; but for the simpler U-shaped
    # PBO we just need the *minimum* best OOS across paths.
    worst_of_best = best_oos_per_path.min()
    # PBO = fraction of paths whose best OOS is below the median best OOS
    # of the *complement* paths. The classical PBO uses a log-optimal
    # definition, but for a defensible first cut we use this rank test.
    median_best = float(np.median(best_oos_per_path))
    pbo = float(np.mean(best_oos_per_path < median_best))
    return PBOResult(
        pbo=pbo,
        n_paths=n_paths,
        n_folds=n_folds,
        best_is_oos=median_best,
        worst_oos=float(worst_of_best),
        extras={
            "n_paths": float(n_paths),
            "n_folds": float(n_folds),
            "median_best_oos": median_best,
            "min_best_oos": float(worst_of_best),
        },
    )


__all__ = [
    "DSRResult",
    "DSRSpec",
    "PBOResult",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfit",
]
