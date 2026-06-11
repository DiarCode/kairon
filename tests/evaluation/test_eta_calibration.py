"""Tests for :mod:`kairon.evaluation.eta_calibration`.

These tests pin the W2.1 Almgren-Chriss ``eta`` calibrator so that
downstream test authors can rely on it. The fixture generator
implements the synthetic BTC distribution recommended by the plan
(price uniform in [50_000, 80_000], qty log-uniform in [0.01, 10],
adv uniform in [100, 2000], sigma uniform in [0.005, 0.02]) and
uses the closed-form Almgren-Chriss law to synthesize the observed
``impact`` for each trade. The "real" data path is the same
function with real ccxt public-trade prints as input; per W0
BTC-only fallback the synthetic path is what we ship in this
iteration.

The four tests cover:

1. ``test_calibrate_recovers_known_eta``: 200 synthetic trades
   generated from a known ``eta=0.6`` round-trip through the
   calibrator must recover ``eta`` within 5% relative error
   (``0.57 < eta < 0.63``).
2. ``test_calibration_sets_is_calibrated_true``: the returned
   model has ``is_calibrated=True`` (not the W1 placeholder).
3. ``test_calibration_with_few_data_returns_fallback``: with
   fewer than ``MIN_TRADES=10`` trades, the function returns
   ``AlmgrenChrissModel(eta=fallback, is_calibrated=False)`` so
   we do not pretend a calibration we don't have.
4. ``test_calibration_with_zero_or_negative_values_raises``:
   defensive — bad input (negative price, zero qty, negative
   sigma) raises ``ValueError`` rather than poisoning the fit.
"""

from __future__ import annotations

import math
import random
from typing import Final

import pytest

from kairon.evaluation.eta_calibration import (
    MIN_TRADES,
    calibrate_eta_from_trades,
)


# ---------------------------------------------------------------------------
# Synthetic-trade fixture (matches the W2.1 plan distribution)
# ---------------------------------------------------------------------------
_KNOWN_ETA: Final[float] = 0.6
_KNOWN_ETA_RELATIVE_TOLERANCE: Final[float] = 0.05  # 5% relative error
_N_TRADES: Final[int] = 200
_SEED: Final[int] = 20260607  # deterministic; the ralph state date


def _synth_trade(price: float, qty: float, adv: float, sigma: float, eta: float) -> float:
    """Return the closed-form Almgren-Chriss impact in price units.

    Helper for the test fixture. Mirrors
    :meth:`AlmgrenChrissModel.compute` so the synthetic
    generator and the calibrator are solving the same equation.
    """
    return eta * sigma * math.sqrt(qty / adv) * price


def _synthetic_btc_trades(
    n: int, *, eta: float, seed: int = _SEED
) -> list[tuple[float, float, float, float, float]]:
    """Generate ``n`` synthetic BTC trades with known ``eta``.

    The plan's W2.1 distribution is:

    - ``price`` ~ Uniform[50_000, 80_000]
    - ``qty``   ~ LogUniform[0.01, 10]   (i.e. ``log(qty)`` is uniform)
    - ``adv``   ~ Uniform[100, 2000]
    - ``sigma`` ~ Uniform[0.005, 0.02]

    Each trade's observed ``impact`` is the closed-form
    Almgren-Chriss value with the supplied ``eta``, so the OLS
    estimator can recover ``eta`` exactly (modulo a small sampling
    error for ``n=200``).
    """
    rng = random.Random(seed)
    out: list[tuple[float, float, float, float, float]] = []
    for _ in range(n):
        price: float = rng.uniform(50_000.0, 80_000.0)
        # log-uniform: sample log10(qty) uniformly in [log10(0.01), log10(10)]
        # and exponentiate. Equivalent to sampling qty from
        # 10**Uniform[-2, +1].
        log10_qty: float = rng.uniform(-2.0, 1.0)
        qty: float = 10.0 ** log10_qty
        adv: float = rng.uniform(100.0, 2000.0)
        sigma: float = rng.uniform(0.005, 0.02)
        impact: float = _synth_trade(price, qty, adv, sigma, eta)
        out.append((price, qty, adv, sigma, impact))
    return out


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_calibrate_recovers_known_eta() -> None:
    """200 synthetic trades from ``eta=0.6`` must round-trip within 5%.

    The PRD W2.1 acceptance criterion #2 is "recovered eta within
    5% relative error of 0.6", which translates to
    ``0.57 < eta < 0.63``. The closed-form OLS estimator is the
    sample mean of the log-residuals, so 200 samples converge
    tightly (the standard error of the mean shrinks like
    ``1/sqrt(n)``).
    """
    trades = _synthetic_btc_trades(_N_TRADES, eta=_KNOWN_ETA)
    model = calibrate_eta_from_trades(trades)
    assert model.is_calibrated is True
    assert _KNOWN_ETA * (1.0 - _KNOWN_ETA_RELATIVE_TOLERANCE) < model.eta < _KNOWN_ETA * (1.0 + _KNOWN_ETA_RELATIVE_TOLERANCE), (
        f"recovered eta={model.eta} is outside the 5% relative "
        f"tolerance band around the known eta={_KNOWN_ETA}"
    )


def test_calibration_sets_is_calibrated_true() -> None:
    """The returned model must have ``is_calibrated=True`` on a successful fit.

    The W1 placeholder has ``is_calibrated=False``. The W2.1
    calibrator must flip the flag to ``True`` so downstream
    consumers (backtest engine, the W3-4 cost-ML re-work loop in
    story 3.7) know the value was measured, not assumed.
    """
    trades = _synthetic_btc_trades(_N_TRADES, eta=_KNOWN_ETA)
    model = calibrate_eta_from_trades(trades)
    assert model.is_calibrated is True
    # Sticky marker: a calibrated run never silently degrades to
    # the placeholder.
    assert model.eta > 0
    assert math.isfinite(model.eta)


def test_calibration_with_few_data_returns_fallback() -> None:
    """With ``< MIN_TRADES`` trades, the function returns the fallback model.

    The PRD W2.1 acceptance criterion #4 says: "with < 10 trades,
    returns ``AlmgrenChrissModel(eta=fallback, is_calibrated=False)``
    so we don't pretend a calibration we don't have". The
    fallback is the Almgren 2005 default of ``0.5``; the
    ``is_calibrated=False`` flag is the contract for downstream
    consumers.
    """
    few_trades: list[tuple[float, float, float, float, float]] = _synthetic_btc_trades(
        MIN_TRADES - 1, eta=_KNOWN_ETA
    )
    model = calibrate_eta_from_trades(few_trades)
    assert model.is_calibrated is False
    assert model.eta == pytest.approx(0.5)


def test_calibration_with_zero_or_negative_values_raises() -> None:
    """Bad input (negative price, zero qty, negative sigma) raises ``ValueError``.

    Defensive guard: the impact law is undefined outside the
    positive-price / positive-qty / positive-adv / non-negative-
    sigma domain, and a silent fit on a poisoned batch would
    produce a meaningless ``eta`` that downstream consumers would
    treat as a measurement. The check is per-trade so the error
    message pinpoints the offender.
    """
    base_trade: tuple[float, float, float, float, float] = (
        50_000.0, 1.0, 1000.0, 0.01, 0.5 * 0.01 * math.sqrt(1.0 / 1000.0) * 50_000.0,
    )
    # Pad to MIN_TRADES so we cross the size threshold and reach
    # the validation loop; the offending trade is the last one.
    padding: list[tuple[float, float, float, float, float]] = [
        (60_000.0, 1.0, 1000.0, 0.01, 0.5 * 0.01 * math.sqrt(1.0 / 1000.0) * 60_000.0)
        for _ in range(MIN_TRADES)
    ]

    # Negative price
    bad_price_trade: tuple[float, float, float, float, float] = (
        -1.0, 1.0, 1000.0, 0.01, 1.0,
    )
    with pytest.raises(ValueError, match="non-positive price"):
        calibrate_eta_from_trades(padding + [bad_price_trade])

    # Zero qty
    bad_qty_trade: tuple[float, float, float, float, float] = (
        50_000.0, 0.0, 1000.0, 0.01, 1.0,
    )
    with pytest.raises(ValueError, match="non-positive qty"):
        calibrate_eta_from_trades(padding + [bad_qty_trade])

    # Negative sigma
    bad_sigma_trade: tuple[float, float, float, float, float] = (
        50_000.0, 1.0, 1000.0, -0.01, 1.0,
    )
    with pytest.raises(ValueError, match="negative sigma"):
        calibrate_eta_from_trades(padding + [bad_sigma_trade])

    # Sanity check: the same padding without any bad trade does
    # NOT raise — guards against a false-positive in the matcher
    # above (e.g. a regex that matches the baseline error).
    calibrate_eta_from_trades(padding + [base_trade])
