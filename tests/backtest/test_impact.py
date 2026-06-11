"""Tests for the Almgren-Chriss market impact placeholder (W1.3)."""

from __future__ import annotations

import math

import pytest

from kairon.backtest.cost import CostModel, DEFAULT_CRYPTO_COSTS, DEFAULT_STOCK_COSTS
from kairon.backtest.impact import DEFAULT_ETA, AlmgrenChrissModel


# ---------------------------------------------------------------------------
# Core placeholder contract (PRD W1.3 acceptance criterion #2)
# ---------------------------------------------------------------------------
def test_placeholder_eta() -> None:
    """W1.3 load-bearing test: with eta=0.5, the model returns the
    exact Almgren-Chriss square-root impact formula on a BTC fixture.
    """
    model = AlmgrenChrissModel(eta=0.5, is_calibrated=False)
    # is_calibrated must be False on the placeholder
    assert model.is_calibrated is False
    assert model.eta == 0.5

    price, qty, adv, sigma = 50_000.0, 1.0, 1_000.0, 0.01
    expected = 0.5 * 0.01 * math.sqrt(1.0 / 1000.0) * 50_000.0
    assert expected == pytest.approx(7.90569, abs=1e-3)

    impact = model.compute(price=price, qty=qty, adv=adv, sigma=sigma)
    assert impact == pytest.approx(expected, abs=1e-9)
    # Same number via the bps convenience: 10000 * 7.90569 / 50000
    assert model.compute_bps(price=price, qty=qty, adv=adv, sigma=sigma) == pytest.approx(
        1.581138, abs=1e-6
    )
    # Placeholder marker is sticky
    assert model.is_calibrated is False


def test_default_eta_is_half() -> None:
    """The placeholder ships at eta=0.5 (Almgren 2005 convention)."""
    assert DEFAULT_ETA == 0.5
    model = AlmgrenChrissModel()
    assert model.eta == 0.5
    assert model.is_calibrated is False


def test_is_calibrated_can_be_set_true_for_calibrated_run() -> None:
    """W2 will flip is_calibrated to True after a successful calibration."""
    model = AlmgrenChrissModel(eta=0.314, is_calibrated=True)
    assert model.is_calibrated is True
    assert model.eta == 0.314


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_eta_must_be_positive() -> None:
    with pytest.raises(ValueError, match="eta"):
        AlmgrenChrissModel(eta=0.0)
    with pytest.raises(ValueError, match="eta"):
        AlmgrenChrissModel(eta=-0.5)


def test_is_calibrated_must_be_bool() -> None:
    with pytest.raises(ValueError, match="is_calibrated"):
        AlmgrenChrissModel(is_calibrated=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is_calibrated"):
        AlmgrenChrissModel(is_calibrated=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is_calibrated"):
        AlmgrenChrissModel(is_calibrated="yes")  # type: ignore[arg-type]


def test_eta_nan_rejected() -> None:
    with pytest.raises(ValueError, match="eta"):
        AlmgrenChrissModel(eta=float("nan"))


def test_compute_rejects_negative_qty() -> None:
    model = AlmgrenChrissModel()
    with pytest.raises(ValueError, match="qty"):
        model.compute(price=50_000.0, qty=-1.0, adv=1_000.0, sigma=0.01)


def test_compute_rejects_zero_adv() -> None:
    model = AlmgrenChrissModel()
    with pytest.raises(ValueError, match="adv"):
        model.compute(price=50_000.0, qty=1.0, adv=0.0, sigma=0.01)


def test_compute_rejects_negative_sigma() -> None:
    model = AlmgrenChrissModel()
    with pytest.raises(ValueError, match="sigma"):
        model.compute(price=50_000.0, qty=1.0, adv=1_000.0, sigma=-0.01)


def test_compute_rejects_zero_price() -> None:
    model = AlmgrenChrissModel()
    with pytest.raises(ValueError, match="price"):
        model.compute(price=0.0, qty=1.0, adv=1_000.0, sigma=0.01)


def test_compute_rejects_negative_price() -> None:
    model = AlmgrenChrissModel()
    with pytest.raises(ValueError, match="price"):
        model.compute(price=-50_000.0, qty=1.0, adv=1_000.0, sigma=0.01)


def test_compute_rejects_invalid_side() -> None:
    model = AlmgrenChrissModel()
    with pytest.raises(ValueError, match="side"):
        model.compute(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.01, side="hold")  # type: ignore[arg-type]


def test_compute_accepts_zero_sigma() -> None:
    """A flat market has zero impact (no vol → no excursion)."""
    model = AlmgrenChrissModel()
    assert model.compute(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.0) == 0.0
    assert model.compute_bps(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.0) == 0.0


# ---------------------------------------------------------------------------
# Symmetry & monotonicity
# ---------------------------------------------------------------------------
def test_buy_sell_symmetry() -> None:
    """buy and sell must produce identical magnitude (asymmetry lives in spread)."""
    model = AlmgrenChrissModel()
    common = dict(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.01)
    assert model.compute(**common, side="buy", regime=None) == model.compute(
        **common, side="sell", regime=None
    )
    assert model.compute_bps(**common, side="buy", regime=None) == model.compute_bps(
        **common, side="sell", regime=None
    )


def test_monotonic_in_qty() -> None:
    """Larger order → larger impact (sqrt(qty) is strictly increasing)."""
    model = AlmgrenChrissModel()
    small = model.compute(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.01)
    big = model.compute(price=50_000.0, qty=10.0, adv=1_000.0, sigma=0.01)
    bigger = model.compute(price=50_000.0, qty=100.0, adv=1_000.0, sigma=0.01)
    assert small < big < bigger
    # 10x qty should give sqrt(10) ≈ 3.162x impact
    assert big == pytest.approx(small * math.sqrt(10.0), abs=1e-9)
    assert bigger == pytest.approx(small * math.sqrt(100.0), abs=1e-9)


def test_monotonic_in_sigma() -> None:
    """Higher vol → larger impact (linear in sigma)."""
    model = AlmgrenChrissModel()
    calm = model.compute(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.005)
    base = model.compute(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.01)
    vol = model.compute(price=50_000.0, qty=1.0, adv=1_000.0, sigma=0.05)
    assert calm < base < vol
    # 2x sigma should give 2x impact
    assert base == pytest.approx(calm * 2.0, abs=1e-9)
    assert vol == pytest.approx(calm * 10.0, abs=1e-9)


def test_monotonic_in_eta() -> None:
    """Larger eta → larger impact (linear in eta)."""
    price, qty, adv, sigma = 50_000.0, 1.0, 1_000.0, 0.01
    small = AlmgrenChrissModel(eta=0.1).compute(price=price, qty=qty, adv=adv, sigma=sigma)
    base = AlmgrenChrissModel(eta=0.5).compute(price=price, qty=qty, adv=adv, sigma=sigma)
    big = AlmgrenChrissModel(eta=1.0).compute(price=price, qty=qty, adv=adv, sigma=sigma)
    assert small < base < big
    assert base == pytest.approx(small * 5.0, abs=1e-9)
    assert big == pytest.approx(small * 10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Dataclass structural contract
# ---------------------------------------------------------------------------
def test_model_is_frozen() -> None:
    """AlmgrenChrissModel must be immutable (frozen dataclass)."""
    model = AlmgrenChrissModel()
    with pytest.raises((AttributeError, Exception)):
        model.eta = 0.6  # type: ignore[misc]


def test_model_equality_is_value_based() -> None:
    """Same eta + is_calibrated → equal (default dataclass eq)."""
    a = AlmgrenChrissModel(eta=0.5, is_calibrated=False)
    b = AlmgrenChrissModel(eta=0.5, is_calibrated=False)
    c = AlmgrenChrissModel(eta=0.5, is_calibrated=True)
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# CostModel integration (PRD W1.3 acceptance criterion #3)
# ---------------------------------------------------------------------------
def test_cost_model_default_unchanged() -> None:
    """The W1.3 acceptance criterion #3 load-bearing invariant:
    DEFAULT_CRYPTO_COSTS and DEFAULT_STOCK_COSTS must still have
    impact_model=None so the 394 baseline cost-model tests do not
    regress. ``total_cost`` with the 2-arg call must also produce
    byte-identical results.
    """
    assert DEFAULT_CRYPTO_COSTS.impact_model is None
    assert DEFAULT_STOCK_COSTS.impact_model is None
    # Byte-identical legacy 2-arg call: crypto bps=10+2+2=14, stock bps=2+1+2=5
    assert DEFAULT_CRYPTO_COSTS.total_cost(10_000.0, "buy") == pytest.approx(14.0)
    assert DEFAULT_STOCK_COSTS.total_cost(10_000.0, "sell") == pytest.approx(5.0)


def test_cost_model_with_impact_model_adds_bps_term() -> None:
    """When impact_model + adv + sigma + price + qty are all supplied,
    total_cost adds the Almgren-Chriss bps term to the legacy total.
    """
    cm = CostModel(
        commission_bps=10.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
        impact_model=AlmgrenChrissModel(eta=0.5, is_calibrated=False),
    )
    # No impact inputs → legacy behaviour, no impact term.
    legacy = cm.total_cost(10_000.0, "buy")
    assert legacy == pytest.approx(10.0)  # 10 bps * 10_000 / 10_000
    # With impact inputs, an extra bps term is added. For the BTC
    # fixture: 10000 * 7.90569 / 50000 ≈ 1.581138 bps.
    with_impact = cm.total_cost(
        10_000.0,
        "buy",
        adv=1_000.0,
        sigma=0.01,
        price=50_000.0,
        qty=1.0,
    )
    expected_bps = 10.0 + 1.581138
    assert with_impact == pytest.approx(10_000.0 * expected_bps / 10_000.0, abs=1e-6)


def test_cost_model_impact_model_rejects_non_algren_object() -> None:
    """The impact_model field must hold an AlmgrenChrissModel or None."""
    with pytest.raises(ValueError, match="impact_model"):
        CostModel(impact_model="not-a-model")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="impact_model"):
        CostModel(impact_model=42)  # type: ignore[arg-type]


def test_cost_model_impact_model_buy_vs_sell_symmetry() -> None:
    """Side parameter 'buy' vs 'sell' must produce identical impact
    magnitude when the model itself is symmetric.
    """
    cm = CostModel(
        commission_bps=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
        impact_model=AlmgrenChrissModel(eta=0.5, is_calibrated=False),
    )
    kwargs = dict(adv=1_000.0, sigma=0.01, price=50_000.0, qty=1.0)
    buy = cm.total_cost(10_000.0, "buy", **kwargs)
    sell = cm.total_cost(10_000.0, "sell", **kwargs)
    assert buy == pytest.approx(sell, abs=1e-9)
