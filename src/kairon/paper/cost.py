"""Maker rebate model for the paper trading engine (W7.3).

The v1 paper trader assumes a single per-side commission
(``CostModel.commission_bps``) and ignores venue-specific
maker-taker fee schedules. In production, **maker** orders
(orders that add liquidity to the book) are often *rebated*
(paid a small fee) while **taker** orders (orders that remove
liquidity) are charged the standard commission. The net cost
of a maker trade is therefore ``commission - rebate``, and the
net cost of a taker trade is ``commission`` (no rebate).

Why a separate model?
---------------------
The :class:`kairon.backtest.cost.CostModel` is the v1
single-number cost model (commission + slippage + half-spread
in bps). Adding a maker-rebate field to it would break the
v1 contract (the W2 calibration, the W3 backtest engine, the
W6 sizer). The W7.3 model is a *thin* additive layer that
takes a :class:`CostModel` and an order kind and returns the
*net* cost in bps. The paper trader composes the two:
``net_bps = MakerRebateModel(cost).net_cost_bps(order_kind)``.

The net cost is expressed in bps so the paper trader can
multiply by notional to get the cash cost, exactly like the
v1 ``CostModel.total_cost`` path. The rebate is +0.2 bps by
default (the v1 BTC venue default for limit orders on
Binance/Bybit maker programmes); the commission is read from
the supplied :class:`CostModel` (``commission_bps``).

This module is pure: no IO, no async, no global state. The
``RebateConfig`` is a frozen dataclass; the
``MakerRebateModel`` is a thin wrapper around the cost
model's commission + the rebate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

if TYPE_CHECKING:
    from kairon.backtest.cost import CostModel


# Order kind enum. ``limit`` orders are makers (they add
# liquidity); ``market`` orders are takers (they remove
# liquidity). The string values match the
# :class:`kairon.paper.OrderType` values so the paper trader
# can pass ``order.order_type.value`` directly.
OrderKind = Literal["limit", "market"]


# Default rebate: +0.2 bps for limit orders (the v1 BTC venue
# default for the maker programme). The rebate is expressed in
# bps of notional, matching the :class:`CostModel` convention.
DEFAULT_REBATE_BPS: Final[float] = 0.2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RebateConfig:
    """Configuration for the :class:`MakerRebateModel`.

    The v1 contract has one knob plus a reference to a
    :class:`CostModel`:

    - ``rebate_bps=0.2``  — the per-side rebate for limit
      orders, in bps of notional. The rebate is *added* to the
      limit order's net cost (i.e. it REDUCES the cost). The
      default is +0.2 bps (the v1 BTC venue default).
    - ``cost``            — the :class:`CostModel` to read
      ``commission_bps`` from. Default ``None`` means "use a
      zero-commission cost model" (the v1 contract for tests
      that don't care about commission). The ``cost`` field
      is stored by reference (the :class:`CostModel` is
      frozen, so reference sharing is safe).
    """

    rebate_bps: float = DEFAULT_REBATE_BPS
    cost: "CostModel | None" = None
    extras: dict[str, Any] = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        if self.rebate_bps < 0.0:
            raise ValueError(
                f"rebate_bps must be >= 0, got {self.rebate_bps!r}"
            )
        # ``cost`` is optional (None -> zero commission).
        # The :class:`CostModel` validates its own fields
        # at construction, so we don't re-validate here.


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MakerRebateModel:
    """A thin maker-taker rebate layer over a :class:`CostModel`.

    The model's single public method is :meth:`net_cost_bps`,
    which takes an order kind (``"limit"`` or ``"market"``)
    and returns the *net* cost in bps:

    - ``"limit"``  : ``commission_bps - rebate_bps`` (a limit
      order gets the rebate; the net cost is the commission
      minus the rebate). The result is in bps of notional.
    - ``"market"`` : ``commission_bps`` (a taker order does
      not get a rebate; the net cost is the commission).

    The v1 contract is symmetric: a taker order's cost is the
    commission; a maker order's cost is the commission minus
    the rebate. A negative net cost is allowed (it would mean
    the venue pays the trader to add liquidity), but the v1
    defaults (commission=10 bps, rebate=0.2 bps) keep the net
    cost positive.

    Usage::

        from kairon.backtest.cost import CostModel
        cfg = RebateConfig(rebate_bps=0.2, cost=CostModel(commission_bps=10.0))
        model = MakerRebateModel(cfg)
        # Limit order: net cost = 10 - 0.2 = 9.8 bps.
        net_bps_limit = model.net_cost_bps("limit")
        # Market order: net cost = 10 bps.
        net_bps_market = model.net_cost_bps("market")
    """

    def __init__(self, config: RebateConfig | None = None) -> None:
        self.config: RebateConfig = config or RebateConfig()

    @property
    def rebate_bps(self) -> float:
        """The per-side rebate for limit orders, in bps of notional."""
        return float(self.config.rebate_bps)

    @property
    def commission_bps(self) -> float:
        """The commission read from the underlying :class:`CostModel`.

        Returns ``0.0`` when no :class:`CostModel` is configured
        (the v1 contract for tests that don't care about
        commission). This makes the
        ``net_cost_bps("market")`` path return 0.0 in the
        "no commission" case, matching the
        ``test_taker_order_no_rebate`` spec.
        """
        if self.config.cost is None:
            return 0.0
        return float(self.config.cost.commission_bps)

    def net_cost_bps(self, order_kind: OrderKind) -> float:
        """Return the net cost in bps for a given order kind.

        Parameters
        ----------
        order_kind
            ``"limit"`` (maker) or ``"market"`` (taker). Any
            other value raises ``ValueError``.

        Returns
        -------
        float
            The net cost in bps of notional. For ``"limit"``
            orders, this is ``commission_bps - rebate_bps``.
            For ``"market"`` orders, this is ``commission_bps``
            (no rebate).
        """
        if order_kind not in ("limit", "market"):
            raise ValueError(
                f"order_kind must be 'limit' or 'market', got "
                f"{order_kind!r}"
            )
        if order_kind == "limit":
            # Maker: commission - rebate. The rebate is *added*
            # to the trader's pocket (i.e. it REDUCES the cost).
            return float(self.commission_bps - self.config.rebate_bps)
        # Taker: commission only.
        return float(self.commission_bps)

    # -- convenience: cash cost for a given notional -------------------
    def net_cost_cash(
        self,
        notional: float,
        order_kind: OrderKind,
    ) -> float:
        """Return the net cost in cash units for a given notional.

        Equivalent to ``notional * net_cost_bps(order_kind) / 1e4``.
        Matches the v1 :meth:`CostModel.total_cost` convention.
        """
        if notional < 0.0:
            raise ValueError(f"notional must be >= 0, got {notional!r}")
        return float(notional * self.net_cost_bps(order_kind) / 1e4)


__all__ = [
    "DEFAULT_REBATE_BPS",
    "MakerRebateModel",
    "OrderKind",
    "RebateConfig",
]
