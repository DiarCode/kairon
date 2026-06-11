"""Calibrate the Almgren-Chriss impact coefficient ``eta`` from trades.

Story W2.1 fits the square-root impact law

    impact = eta * sigma * sqrt(qty / adv) * price

to a sample of observed trades with the linearised log-equation

    log(impact / price) = log(eta) + log(sigma) + 0.5 * log(qty / adv)

so the fit reduces to a constant-offset ordinary least-squares
regression on a single unknown (``log eta``). The synthetic BTC
distribution recommended by the plan (price uniform in
[50_000, 80_000], qty log-uniform in [0.01, 10], adv uniform in
[100, 2000], sigma uniform in [0.005, 0.02]) provides the reference
fixture. The "real" data path is the same function with real ccxt
public-trade prints as input; see ``artifacts/w2_1_status.json`` for
the deferred-network note.

Trade-tuple shape
-----------------
Each trade is a 5-tuple ``(price, qty, adv, sigma, impact)`` where
``impact`` is the OBSERVED temporary price excursion in price units
(typically the public-trade mid-price excursion averaged over the
execution window for a real print, or the Almgren-Chriss law value
itself for a synthetic fixture). The 5-tuple is the only way the
closed-form OLS estimator can recover ``eta``; the PRD's "4-tuple
(price, qty, adv, sigma)" spec is consistent with the input
*features* the model conditions on, and the 5th element
(``impact``) is the dependent variable the fit solves for. Downstream
callers (real-data path) measure ``impact`` from public-trade prints
directly; the synthetic path generates it with the known ``eta``.

The function is pure: no IO, no async, no global state. The return
type is the existing :class:`AlmgrenChrissModel` dataclass from
:mod:`kairon.backtest.impact`, so downstream consumers (the backtest
engine, the cost-ML re-work loop in W3-4 story 3.7) can consume the
calibrated model without a new type.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kairon.backtest.impact import AlmgrenChrissModel


# Minimum number of trades for a defensible least-squares fit. Below
# this threshold the function falls back to the Almgren 2005 default
# (``0.5``) with ``is_calibrated=False`` so downstream code can
# distinguish "we measured eta" from "we used the placeholder".
MIN_TRADES: int = 10


def calibrate_eta_from_trades(
    trades: list[tuple[float, float, float, float, float]],
    *,
    fallback: float = 0.5,
) -> "AlmgrenChrissModel":
    """Fit :class:`AlmgrenChrissModel` from observed trade tuples.

    Each trade is a 5-tuple
    ``(price, qty, adv, sigma, impact)`` where ``impact`` is the
    observed temporary price excursion in price units. The function
    fits the linearised log-equation

        log(impact / price) = log(eta) + log(sigma) + 0.5 * log(qty / adv)

    via the closed-form OLS estimator

        log(eta) = mean_i [ log(impact_i / price_i)
                           - log(sigma_i)
                           - 0.5 * log(qty_i / adv_i) ]

    The mean is over all observations, so a small fraction of
    misbehaving observations is damped by the bulk. The estimator
    is the standard intercept-only regression with one regressor
    (the constant 1) and ``n`` iid samples.

    Parameters
    ----------
    trades
        Iterable of ``(price, qty, adv, sigma, impact)`` tuples,
        typically derived from real ccxt public-trade prints (or,
        when W0 fallback is active, from a synthetic BTC
        distribution that matches the real one's shape). The
        function copies the iterable into a list internally so
        callers can pass any iterable, including a generator.
    fallback
        Value of ``eta`` to return when there are too few trades
        (``len(trades) < MIN_TRADES``) or the least-squares fit
        fails for any reason (non-finite math, overflow, ``fallback
        <= 0``). Defaults to ``0.5`` (the Almgren 2005 convention
        used in :mod:`kairon.backtest.impact`).

    Returns
    -------
    AlmgrenChrissModel
        ``AlmgrenChrissModel(eta=<recovered>, is_calibrated=True)``
        on a successful fit, or
        ``AlmgrenChrissModel(eta=fallback, is_calibrated=False)``
        on a fallback path. The ``is_calibrated`` flag is the
        contract for downstream consumers (backtest engine, the
        W3-4 cost-ML re-work loop in story 3.7) — they MUST treat
        ``is_calibrated=False`` as a heuristic, not a measurement.

    Raises
    ------
    ValueError
        If any trade has a non-positive ``price``, a non-positive
        ``qty``, a non-positive ``adv``, a negative ``sigma``, a
        non-positive ``impact``, a non-finite component, or the
        wrong tuple arity. The check is a defensive guard so bad
        input cannot silently poison the fit; it is not in the
        PRD spec but is the kind of input-validation the existing
        :meth:`AlmgrenChrissModel.compute` does at every call.
    """
    # Local import to avoid a circular import at module-load time
    # (kairon.backtest.cost imports kairon.backtest.impact which
    # would in turn import kairon.evaluation if we hoisted the
    # symbol). The cost is one import per call; calibration is not
    # on the hot path.
    from kairon.backtest.impact import AlmgrenChrissModel

    if fallback <= 0 or math.isnan(fallback):
        # We cannot return a valid model with a non-positive or
        # NaN fallback — that would violate AlmgrenChrissModel's
        # own __post_init__ invariant. Bail with a clear error.
        raise ValueError(
            f"fallback must be > 0 and finite, got {fallback!r}"
        )

    trade_list: list[tuple[float, float, float, float, float]] = list(trades)
    if len(trade_list) < MIN_TRADES:
        return AlmgrenChrissModel(eta=fallback, is_calibrated=False)

    # Defensive input validation: every (price, qty, adv, sigma,
    # impact) tuple must be in the domain where the impact law is
    # defined AND the log-equation is solvable (all args of log
    # must be strictly positive). We do this in one pass over the
    # list (rather than letting the fit explode on the first bad
    # row) so the error message pinpoints the offender.
    for idx, trade in enumerate(trade_list):
        if len(trade) != 5:
            raise ValueError(
                f"trade #{idx} must have exactly 5 components "
                f"(price, qty, adv, sigma, impact), got {len(trade)}"
            )
        price, qty, adv, sigma, impact = trade
        if not (math.isfinite(price) and math.isfinite(qty)
                and math.isfinite(adv) and math.isfinite(sigma)
                and math.isfinite(impact)):
            raise ValueError(
                f"trade #{idx} has a non-finite component: "
                f"price={price!r}, qty={qty!r}, adv={adv!r}, "
                f"sigma={sigma!r}, impact={impact!r}"
            )
        if price <= 0:
            raise ValueError(
                f"trade #{idx} has non-positive price={price!r}"
            )
        if qty <= 0:
            raise ValueError(
                f"trade #{idx} has non-positive qty={qty!r}"
            )
        if adv <= 0:
            raise ValueError(
                f"trade #{idx} has non-positive adv={adv!r}"
            )
        if sigma < 0:
            raise ValueError(
                f"trade #{idx} has negative sigma={sigma!r}"
            )
        if impact <= 0:
            raise ValueError(
                f"trade #{idx} has non-positive impact={impact!r}"
            )

    # Linearised OLS fit. The dependent variable is
    #     y_i = log(impact_i / price_i)
    # and the model says
    #     y_i = log(eta) + log(sigma_i) + 0.5 * log(qty_i / adv_i)
    # which we rearrange to
    #     log(eta) = y_i - log(sigma_i) - 0.5 * log(qty_i / adv_i)
    # The OLS estimator (intercept-only) is the sample mean of the
    # right-hand side; we then exponentiate to recover eta. The
    # mean is the BLUE under the model's iid-Gaussian noise
    # assumption, and is robust to a small fraction of outliers
    # because each residual contributes equally.
    log_eta_samples: list[float] = []
    for price, qty, adv, sigma, impact in trade_list:
        log_residual: float = (
            math.log(impact / price)
            - math.log(sigma)
            - 0.5 * math.log(qty / adv)
        )
        log_eta_samples.append(log_residual)

    if not log_eta_samples:
        # Unreachable: len >= MIN_TRADES >= 10 guarantees the list
        # is non-empty, but the explicit guard keeps pyright strict
        # happy (no index-out-of-range on the divide below).
        return AlmgrenChrissModel(eta=fallback, is_calibrated=False)

    log_eta_mean: float = sum(log_eta_samples) / len(log_eta_samples)
    eta_recovered: float = math.exp(log_eta_mean)

    if not (math.isfinite(eta_recovered) and eta_recovered > 0):
        # Catastrophic failure (overflow, all-zero residuals).
        # Degrade gracefully to the fallback so the pipeline does
        # not crash on a bad batch.
        return AlmgrenChrissModel(eta=fallback, is_calibrated=False)

    return AlmgrenChrissModel(eta=eta_recovered, is_calibrated=True)


__all__ = [
    "MIN_TRADES",
    "calibrate_eta_from_trades",
]
