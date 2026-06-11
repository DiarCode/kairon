"""Tests for the W7.3 maker rebate model.

Two tests pin the W7.3 acceptance criteria:

1. ``test_limit_order_gets_rebate`` — a 1 BTC limit order at
   50,000 USDT with ``commission_bps=10`` and ``rebate_bps=0.2``
   produces a net cost of ``(10 - 0.2) = 9.8`` bps of notional.
   The cash cost is ``1 * 50_000 * 9.8 / 1e4 = 49.0`` USDT.
2. ``test_taker_order_no_rebate`` — a market order with
   ``commission_bps=10`` produces a net cost of ``10`` bps
   (no rebate). The cash cost is ``1 * 50_000 * 10 / 1e4
   = 50.0`` USDT.
"""

from __future__ import annotations

import pytest

from kairon.backtest.cost import CostModel
from kairon.paper.cost import (
    DEFAULT_REBATE_BPS,
    MakerRebateModel,
    RebateConfig,
)


# ---------------------------------------------------------------------------
# W7.3 acceptance criterion #1: limit order gets the rebate
# ---------------------------------------------------------------------------
def test_limit_order_gets_rebate() -> None:
    """A 1 BTC limit order at 50,000 USDT gets the 0.2 bps rebate.

    With ``commission_bps=10`` and ``rebate_bps=0.2``, the net
    cost for a limit order is ``10 - 0.2 = 9.8`` bps. The
    notional is ``1 * 50_000 = 50_000`` USDT, so the cash
    cost is ``50_000 * 9.8 / 1e4 = 49.0`` USDT.
    """
    cost: CostModel = CostModel(commission_bps=10.0)
    cfg: RebateConfig = RebateConfig(rebate_bps=0.2, cost=cost)
    model: MakerRebateModel = MakerRebateModel(cfg)

    # Sanity: the v1 defaults are the v1 contract.
    assert cfg.rebate_bps == DEFAULT_REBATE_BPS
    assert cost.commission_bps == 10.0

    # The net cost in bps is the commission minus the rebate.
    net_bps: float = model.net_cost_bps("limit")
    assert net_bps == pytest.approx(10.0 - 0.2, abs=1e-12)
    assert net_bps == pytest.approx(9.8, abs=1e-12)

    # The cash cost for a 1 BTC limit order at 50,000 USDT is
    # ``notional * net_bps / 1e4 = 50_000 * 9.8 / 1e4 = 49.0``.
    notional: float = 1.0 * 50_000.0  # = 50_000 USDT
    cash_cost: float = model.net_cost_cash(notional=notional, order_kind="limit")
    assert cash_cost == pytest.approx(49.0, abs=1e-9)

    # And: the rebate is +0.2 bps (positive), so the net cost
    # is strictly LESS than the commission (the trader is paid
    # to add liquidity).
    assert model.net_cost_bps("limit") < model.commission_bps
    # The magnitude of the reduction equals the rebate.
    assert (
        model.commission_bps - model.net_cost_bps("limit")
    ) == pytest.approx(cfg.rebate_bps, abs=1e-12)


# ---------------------------------------------------------------------------
# W7.3 acceptance criterion #2: taker order does not get the rebate
# ---------------------------------------------------------------------------
def test_taker_order_no_rebate() -> None:
    """A market order does not get a rebate.

    With ``commission_bps=10``, a market (taker) order's net
    cost is exactly ``10`` bps of notional. The cash cost for
    a 1 BTC market order at 50,000 USDT is
    ``50_000 * 10 / 1e4 = 50.0`` USDT.
    """
    cost: CostModel = CostModel(commission_bps=10.0)
    cfg: RebateConfig = RebateConfig(rebate_bps=0.2, cost=cost)
    model: MakerRebateModel = MakerRebateModel(cfg)

    # The net cost in bps is exactly the commission (no
    # rebate applied to taker orders).
    net_bps: float = model.net_cost_bps("market")
    assert net_bps == pytest.approx(10.0, abs=1e-12)
    assert net_bps == pytest.approx(cost.commission_bps, abs=1e-12)

    # The cash cost for a 1 BTC market order at 50,000 USDT is
    # ``50_000 * 10 / 1e4 = 50.0``.
    notional: float = 1.0 * 50_000.0  # = 50_000 USDT
    cash_cost: float = model.net_cost_cash(notional=notional, order_kind="market")
    assert cash_cost == pytest.approx(50.0, abs=1e-9)

    # And: the limit-order rebate is the ONLY difference
    # between a limit and a market order at the same venue.
    # The differential is ``rebate_bps`` in bps, or
    # ``notional * rebate_bps / 1e4`` in cash.
    diff_bps: float = model.net_cost_bps("market") - model.net_cost_bps("limit")
    assert diff_bps == pytest.approx(cfg.rebate_bps, abs=1e-12)
    diff_cash: float = (
        model.net_cost_cash(notional=notional, order_kind="market")
        - model.net_cost_cash(notional=notional, order_kind="limit")
    )
    assert diff_cash == pytest.approx(notional * cfg.rebate_bps / 1e4, abs=1e-9)

    # Defensive: an unknown order kind raises ValueError.
    with pytest.raises(ValueError, match="order_kind"):
        model.net_cost_bps("stop")  # type: ignore[arg-type]

    # Defensive: a negative notional raises ValueError.
    with pytest.raises(ValueError, match="notional"):
        model.net_cost_cash(notional=-1.0, order_kind="market")
