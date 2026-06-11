"""Stacked-meta vs primary CAS-dominance comparison (W6.3).

Story W6.3 is the *gate* between the W6 stacked-meta release
(W6.1 + W6.2) and the W6.4/W6.5 deliverables. The function
:func:`pareto_compare_cas` measures the CAS (Cost-Adjusted Sharpe)
of the primary ensemble (the W3.1 ``TopKConfidenceEnsemble``) and
the stacked-generalization ensemble (the W6.2
``StackedGeneralizationEnsemble``) on a feature matrix and target
vector. If the stacked CAS strictly dominates the primary CAS on
at least one asset (and the difference is statistically
significant, paired t-test p < 0.1), ``dominates=True`` and the
stacked meta is shipped. Otherwise the W6 FALLBACK fires: the
W6.2 stacked meta is documented as 'not worth shipping' and the
W6.4 multi-head + W6.5 sizer path is the v1 release.

CAS definition
--------------
CAS (Cost-Adjusted Sharpe) is the Sharpe ratio of the per-bar
strategy returns, net of the W1.3 ``DEFAULT_CRYPTO_COSTS``
round-trip cost. Concretely::

    pnl_per_bar[i] = signal[i] * (price[i+1] - price[i]) / price[i]
                   - cost_per_bar_drag
    sharpe(pnl_per_bar) -> CAS

The function uses synthetic per-asset price walks (BTCUSDT,
ETHUSDT, SOLUSDT — the W2.2 universe) so the comparison is
hermetic and does not require a live data feed. The W0
BTC-only fallback is honoured: the W6.3 fallback is documented
in the W6.3 status file and the W6.2 stacked meta's CAS is
measured on the same 3-asset universe.

W6 FALLBACK contract
--------------------
The W6 FALLBACK fires when the stacked meta fails CAS-dominance
on ALL of the 3 supplied assets. The fallback is the
load-bearing pre-mortem scenario #2 from the W6.3 plan: "if
the stacked meta does not strictly beat the primary on any
asset, the meta is not worth shipping and the W6.4 multi-head
+ sizer path is the v1 release."

The runner script (``scripts/run_pareto_compare.py``) writes
the ``W6_FALLBACK_DECISION: 'skip_stacked_meta'`` marker to
the report's headline when the fallback fires.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.models.base import Prediction


# Default assets per plan §W6.3 / W2.2. The 3-asset set is the
# W2.2 universe: BTCUSDT, ETHUSDT, SOLUSDT.
DEFAULT_ASSETS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

# Per-asset vol multiplier (BTC=1.0, ETH=1.3, SOL=1.8 — matches
# the W3.5 / W2.2 synthetic-sigma multipliers).
ASSET_VOL_MULT: dict[str, float] = {
    "BTCUSDT": 1.0,
    "ETHUSDT": 1.3,
    "SOLUSDT": 1.8,
}

# Default fixtures: n_bars bars of synthetic BTC-like price walk.
DEFAULT_N_BARS: int = 1440
DEFAULT_BASE_SIGMA: float = 0.005
DEFAULT_BASE_PRICE: float = 50_000.0
DEFAULT_SEED: int = 20260608

# W6.3 paired t-test p-value threshold. p < 0.1 is the
# documented gate per plan §W6.3: the stacked CAS must be
# significantly higher than the primary CAS (p < 0.1) on
# AT LEAST ONE asset for the stacked meta to ship.
DEFAULT_TTEST_P_THRESHOLD: float = 0.1


# ---------------------------------------------------------------------------
# Synthetic price + signal generation
# ---------------------------------------------------------------------------
def _synthesize_prices(
    *,
    n_bars: int,
    sigma: float,
    seed: int,
    base_price: float = DEFAULT_BASE_PRICE,
) -> np.ndarray:
    """Return a deterministic BTC-like log-normal price walk."""
    if n_bars <= 0:
        raise ValueError(f"n_bars must be > 0, got {n_bars}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    rng: np.random.Generator = np.random.default_rng(seed)
    log_returns: np.ndarray = rng.normal(loc=0.0, scale=sigma, size=n_bars)
    log_prices: np.ndarray = np.empty(n_bars, dtype=np.float64)
    log_prices[0] = math.log(base_price)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns[:-1])
    return np.exp(log_prices).astype(np.float64)


def _prediction_to_signal(pred: Prediction) -> np.ndarray:
    """Convert a :class:`Prediction` to a 1-D {-1, 0, +1} signal.

    The mapping is::

        y_class[i] > 0  -> +1
        y_class[i] < 0  -> -1
        y_class[i] == 0 -> 0

    For binary-class labels in {0, 1}, ``1 -> +1`` and ``0 -> 0``.
    """
    y_class: np.ndarray = np.asarray(pred.y_class, dtype=np.int64)
    return np.sign(y_class).astype(np.int8)


# ---------------------------------------------------------------------------
# CAS computation
# ---------------------------------------------------------------------------
def _cas_from_signals(
    *,
    prices: np.ndarray,
    signals: np.ndarray,
    cost_model: CostModel,
) -> float:
    """Compute the cost-adjusted Sharpe of a (price, signal) pair.

    The per-bar PnL is ``signal[i] * log_return[i]`` net of the
    cost model's per-bar cost drag (approximated as
    ``round_trip_bps / 1e4 / 4`` — a 1-trade-per-4-bars
    attribution; the exact cost model is the W8 backtest's
    job). The result is the mean-scaled Sharpe of the
    cost-adjusted return series.
    """
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got ndim={prices.ndim}")
    if signals.ndim != 1:
        raise ValueError(f"signals must be 1-D, got ndim={signals.ndim}")
    n: int = int(prices.size)
    if signals.size < n:
        # Pad with zeros (FLAT) so the signal length matches
        # the price length.
        padded: np.ndarray = np.zeros(n, dtype=np.int8)
        padded[: signals.size] = signals.astype(np.int8, copy=False)
        signals = padded
    elif signals.size > n:
        signals = signals[:n]
    if n < 2:
        return 0.0
    log_prices: np.ndarray = np.log(prices.astype(np.float64, copy=False))
    log_returns: np.ndarray = np.diff(log_prices)
    aligned_signals: np.ndarray = signals[:-1].astype(np.float64, copy=False)
    pnl: np.ndarray = aligned_signals * log_returns
    cost_per_bar: float = float(cost_model.round_trip_bps) / 1e4 / 4.0
    pnl = pnl - cost_per_bar
    if pnl.std(ddof=0) == 0.0:
        return 0.0
    bars_per_year: int = 365 * 24  # hourly bar assumption
    return float(
        pnl.mean() / pnl.std(ddof=0) * math.sqrt(float(bars_per_year))
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def pareto_compare_cas(
    primary: Any,
    stacked: Any,
    features: Any,
    y: np.ndarray,
    *,
    n_assets: int = 3,
) -> dict[str, Any]:
    """Run the W6.3 CAS-dominance comparison.

    The function takes a primary ensemble and a stacked
    generalization ensemble, fits both on the supplied
    (features, y), and evaluates each on a per-asset synthetic
    price walk to compute the CAS at the per-asset level. The
    ensembles are expected to expose the standard
    ``Model`` surface (``fit`` and ``predict``).

    Parameters
    ----------
    primary
        The primary ensemble. Any object with
        ``fit(features, y) -> trained`` and
        ``predict(trained, features) -> Prediction`` works
        (the v1 path is :class:`TopKConfidenceEnsemble`).
    stacked
        The stacked-generalization ensemble. Same surface
        (the W6.2 path is :class:`StackedGeneralizationEnsemble`).
    features
        The training feature matrix. The exact type is
        backend-agnostic; the v1 path is :class:`FeatureMatrix`.
    y
        1-D target vector aligned to ``features``.
    n_assets
        Number of per-asset CAS evaluations. Default ``3`` (the
        W2.2 universe of BTCUSDT, ETHUSDT, SOLUSDT).

    Returns
    -------
    dict[str, Any]
        Result dict with the documented W6.3 shape::

            {
              "primary_cas": list[float],   # per-asset primary CAS
              "stacked_cas": list[float],   # per-asset stacked CAS
              "ttest_pvalue": float,        # paired t-test p-value
              "dominates": bool,            # stacked CAS > primary CAS on >= 1 asset with p < 0.1
              "w6_fallback": bool,          # stacked_cas <= primary_cas on ALL n_assets assets
              "assets": list[str],          # asset names
              "n_assets": int,              # n_assets arg
              "n_dominating_assets": int,   # count of assets where stacked > primary
            }
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1, got {n_assets}")
    if features is None:
        raise ValueError("features must not be None")

    primary_trained: Any = primary.fit(features, y)
    stacked_trained: Any = stacked.fit(features, y)

    primary_pred: Prediction = primary.predict(primary_trained, features)
    stacked_pred: Prediction = stacked.predict(stacked_trained, features)
    primary_signal: np.ndarray = _prediction_to_signal(primary_pred)
    stacked_signal: np.ndarray = _prediction_to_signal(stacked_pred)

    assets: list[str] = list(DEFAULT_ASSETS)[:n_assets]
    primary_cas: list[float] = []
    stacked_cas: list[float] = []

    cost_model: CostModel = DEFAULT_CRYPTO_COSTS
    for i, asset in enumerate(assets):
        vol_mult: float = ASSET_VOL_MULT.get(asset, 1.0)
        prices: np.ndarray = _synthesize_prices(
            n_bars=DEFAULT_N_BARS,
            sigma=DEFAULT_BASE_SIGMA * vol_mult,
            seed=DEFAULT_SEED + i,
        )
        primary_cas.append(
            _cas_from_signals(
                prices=prices, signals=primary_signal, cost_model=cost_model
            )
        )
        stacked_cas.append(
            _cas_from_signals(
                prices=prices, signals=stacked_signal, cost_model=cost_model
            )
        )

    p_value: float = _paired_ttest_pvalue(primary_cas, stacked_cas)
    n_dominating_assets: int = sum(
        1 for p, s in zip(primary_cas, stacked_cas) if s > p
    )
    dominates: bool = bool(
        n_dominating_assets >= 1
        and p_value < DEFAULT_TTEST_P_THRESHOLD
    )
    w6_fallback: bool = bool(
        all(s <= p for p, s in zip(primary_cas, stacked_cas))
    )

    return {
        "primary_cas": primary_cas,
        "stacked_cas": stacked_cas,
        "ttest_pvalue": float(p_value),
        "dominates": dominates,
        "w6_fallback": w6_fallback,
        "assets": assets,
        "n_assets": n_assets,
        "n_dominating_assets": n_dominating_assets,
    }


def _paired_ttest_pvalue(
    primary_cas: list[float],
    stacked_cas: list[float],
) -> float:
    """Compute the paired t-test p-value of the per-asset CAS.

    Uses :func:`scipy.stats.ttest_rel` if scipy is available;
    falls back to a 2-sided t-test approximation otherwise.
    Returns ``1.0`` (no significance) when the two CAS vectors
    are degenerate (e.g. zero variance) so the result is
    well-formed for the W6.3 acceptance criterion.
    """
    n: int = min(len(primary_cas), len(stacked_cas))
    if n < 2:
        return 1.0
    a: np.ndarray = np.asarray(primary_cas[:n], dtype=np.float64)
    b: np.ndarray = np.asarray(stacked_cas[:n], dtype=np.float64)
    diff: np.ndarray = b - a
    if diff.std(ddof=0) == 0.0:
        return 1.0
    try:
        from scipy import stats  # type: ignore[import-not-found]

        result: Any = stats.ttest_rel(b, a)
        p: float = float(result.pvalue)
        if not math.isfinite(p):
            return 1.0
        return p
    except ImportError:
        t_stat: float = float(
            diff.mean() / (diff.std(ddof=1) / math.sqrt(float(n)))
        )
        p_two_sided: float = 2.0 * (
            1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0)))
        )
        return float(p_two_sided)


__all__ = [
    "DEFAULT_ASSETS",
    "DEFAULT_N_BARS",
    "DEFAULT_TTEST_P_THRESHOLD",
    "pareto_compare_cas",
]
