"""Tests for the backtest cost model."""

from __future__ import annotations

import pytest

from kairon.backtest.cost import (
    DEFAULT_CRYPTO_COSTS,
    DEFAULT_STOCK_COSTS,
    CostModel,
)


def test_cost_model_validates() -> None:
    with pytest.raises(ValueError):
        CostModel(commission_bps=-1)
    with pytest.raises(ValueError):
        CostModel(slippage_bps=-1)
    with pytest.raises(ValueError):
        CostModel(half_spread_bps=-1)
    with pytest.raises(ValueError):
        CostModel(impact_coefficient=-1)
    with pytest.raises(ValueError):
        CostModel(min_trade_bps=-1)


def test_round_trip_bps() -> None:
    cm = CostModel(commission_bps=10, slippage_bps=2, half_spread_bps=3)
    # (10 + 2 + 3) * 2 = 30
    assert cm.round_trip_bps == 30.0


def test_total_cost_basic() -> None:
    cm = CostModel(commission_bps=10, slippage_bps=0, half_spread_bps=0)
    assert cm.total_cost(10_000.0, "buy") == pytest.approx(10.0)
    assert cm.total_cost(10_000.0, "sell") == pytest.approx(10.0)


def test_total_cost_rejects_negative_notional() -> None:
    cm = CostModel()
    with pytest.raises(ValueError, match="notional"):
        cm.total_cost(-1.0, "buy")


def test_total_cost_rejects_bad_side() -> None:
    cm = CostModel()
    with pytest.raises(ValueError, match="side"):
        cm.total_cost(100.0, "sideways")  # type: ignore[arg-type]


def test_should_trade_round_trip_filter() -> None:
    cm = CostModel(commission_bps=10, slippage_bps=2, half_spread_bps=3)
    # round_trip = 30 bps
    assert cm.should_trade(10.0) is False
    assert cm.should_trade(30.0) is True
    assert cm.should_trade(50.0) is True


def test_should_trade_rejects_non_positive() -> None:
    cm = CostModel()
    assert cm.should_trade(0.0) is False
    assert cm.should_trade(-1.0) is False


def test_default_crypto_costs() -> None:
    # 0.10% taker commission per side
    assert DEFAULT_CRYPTO_COSTS.commission_bps == 10.0
    # round trip is 28 bps
    assert DEFAULT_CRYPTO_COSTS.round_trip_bps == 28.0


def test_default_stock_costs() -> None:
    assert DEFAULT_STOCK_COSTS.commission_bps == 2.0
