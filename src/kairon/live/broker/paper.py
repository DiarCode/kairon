"""PaperBroker: deterministic simulation broker implementing the Broker protocol.

Uses the existing :class:`kairon.paper.cost.CostModel` for realistic
commission + slippage, and mirrors the position-lifecycle logic from
:class:`kairon.paper.PaperTrader` (weighted-average entry on same-side adds,
pro-rata cost on reductions).

Unlike :class:`FakeBroker` (which fills at the exact mark price),
PaperBroker applies a configurable slippage (default 5bps) and the
project's crypto cost model. This makes it suitable for integration
tests that verify the TradingLoop's sizing, exposure, and PnL
accounting match the backtest engine within tolerance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from kairon.live.broker.base import (
    Balance,
    Broker,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid7() -> str:
    return uuid.uuid4().hex[:24]


@dataclass
class _PaperPosition:
    """Internal position tracking for PaperBroker."""

    symbol: str
    side: OrderSide
    qty: float
    avg_entry: float
    entry_cost_cash: float  # total cash spent on entry (including costs)


class PaperBroker:
    """Deterministic simulation broker for integration tests.

    Fills market orders at ``mark_price * (1 ± slippage_bps/10000)``
    where the sign depends on the order side (buy → +slippage, sell → −slippage).
    Limit orders fill when the mark price crosses the limit. SL/TP conditional
    orders are tracked and filled when the mark crosses the trigger price.

    Costs are applied via :class:`kairon.paper.cost.CostModel` (default
    ``DEFAULT_CRYPTO_COSTS`` with 10bps commission + 2bps slippage).
    """

    def __init__(
        self,
        *,
        initial_balance: float = 10_000.0,
        currency: str = "USDT",
        slippage_bps: float = 5.0,
        cost_model: CostModel | None = None,
    ) -> None:
        self._initial_balance = initial_balance
        self._currency = currency
        self._slippage_bps = slippage_bps
        self._cost_model = cost_model or DEFAULT_CRYPTO_COSTS

        # Mutable state
        self._balances: dict[str, float] = {currency: initial_balance}
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._positions: dict[str, _PaperPosition] = {}
        self._leverage: dict[str, int] = {}

        # Mark prices per symbol (set by caller or test)
        self._marks: dict[str, float] = {}

    # --- Mark price management (for tests) -------------------------------

    def set_mark_price(self, symbol: str, price: float) -> None:
        """Set the current mark price for a symbol."""
        self._marks[symbol] = price

    # --- Broker protocol implementation -------------------------------------

    async def place_order(self, order: Order) -> Order:
        """Fill a market order immediately or record a limit/conditional order."""
        if order.id in self._orders and self._orders[order.id].status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        ):
            # Idempotent: return existing filled/cancelled order
            return self._orders[order.id]

        mark = self._marks.get(order.symbol)
        if mark is None and order.order_type == OrderType.MARKET:
            msg = f"No mark price set for {order.symbol}; cannot fill market order"
            raise ValueError(msg)

        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, mark)
        # Limit and conditional orders are recorded as PENDING
        pending = order.model_copy(update={"status": OrderStatus.PENDING})
        self._orders[order.id] = pending
        return pending

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            msg = f"Order {order_id} not found"
            raise KeyError(msg)
        cancelled = order.model_copy(update={"status": OrderStatus.CANCELLED, "ts": _utc_now_iso()})
        self._orders[order_id] = cancelled
        return cancelled

    async def cancel_all(self, symbol: str) -> list[Order]:
        cancelled: list[Order] = []
        for oid, order in list(self._orders.items()):
            if order.symbol == symbol and order.status in (
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
            ):
                cancelled_order = order.model_copy(
                    update={"status": OrderStatus.CANCELLED, "ts": _utc_now_iso()}
                )
                self._orders[oid] = cancelled_order
                cancelled.append(cancelled_order)
        return cancelled

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        positions = [
            Position(
                symbol=p.symbol,
                side=p.side,
                qty=p.qty,
                avg_entry=p.avg_entry,
                unrealized_pnl=self._unrealized_pnl(p),
                ts=_utc_now_iso(),
            )
            for p in self._positions.values()
            if symbol is None or p.symbol == symbol
        ]
        return positions

    async def get_balances(self) -> list[Balance]:
        return [
            Balance(currency=ccy, available=amt, total=amt, ts=_utc_now_iso())
            for ccy, amt in self._balances.items()
        ]

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        orders = [
            o
            for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
        ]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        self._leverage[symbol] = leverage

    async def place_conditional(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        trigger_price: float,
        order_type: OrderType = OrderType.MARKET,
        sl: float | None = None,
        tp: float | None = None,
    ) -> Order:
        order = Order(
            id=_uuid7(),
            intent_id=_uuid7(),
            trace_id=_uuid7(),
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=trigger_price,
            sl=sl,
            tp=tp,
            status=OrderStatus.PENDING,
            broker_id=f"paper-cond-{_uuid7()[:8]}",
            ts=_utc_now_iso(),
        )
        self._orders[order.id] = order
        return order

    # --- Tick: check limit/SL/TP orders against current mark --------------

    def tick(self) -> list[Fill]:
        """Check all pending limit and conditional orders against mark prices.

        Call this after setting mark prices via ``set_mark_price()``.
        Returns a list of new fills generated this tick.

        Conditional order semantics:
        - SELL-side (closing a long): triggers when mark <= trigger_price (price dropped)
        - BUY-side (closing a short): triggers when mark >= trigger_price (price rose)
        - Limit BUY: triggers when mark <= limit_price
        - Limit SELL: triggers when mark >= limit_price
        - SL on SELL-side order: triggers when mark <= sl (price dropped further)
        - SL on BUY-side order: triggers when mark >= sl (price rose further)
        - TP on SELL-side order: triggers when mark >= tp (price rose to target)
        - TP on BUY-side order: triggers when mark <= tp (price dropped to target)
        """
        new_fills: list[Fill] = []
        for oid, order in list(self._orders.items()):
            if order.status != OrderStatus.PENDING:
                continue
            mark = self._marks.get(order.symbol)
            if mark is None:
                continue

            filled = False

            # Check limit order fill condition
            if order.price is not None and order.order_type == OrderType.LIMIT:
                should_fill = (
                    (order.side == OrderSide.BUY and mark <= order.price)
                    or (order.side == OrderSide.SELL and mark >= order.price)
                )
                if should_fill:
                    fill = self._execute_fill(order, mark)
                    new_fills.append(fill)
                    filled = True

            # Check conditional (trigger_price) order fill condition.
            # Conditional orders with a trigger_price but no SL/TP activate when
            # the mark price reaches the trigger level. Direction:
            # - SELL-side: triggers when mark <= trigger (stop-loss, closing a long)
            # - BUY-side: triggers when mark >= trigger (stop-loss, closing a short)
            # For take-profit, use the tp field directly (checked below).
            if not filled and order.broker_id is not None and order.broker_id.startswith("paper-cond-"):
                if order.price is not None and order.sl is None and order.tp is None:
                    triggered = (
                        (order.side == OrderSide.SELL and mark <= order.price)
                        or (order.side == OrderSide.BUY and mark >= order.price)
                    )
                    if triggered:
                        fill = self._execute_fill(order, mark)
                        new_fills.append(fill)
                        filled = True

            # Check SL/TP conditions on any pending order
            if not filled and (order.sl is not None or order.tp is not None):
                # SL: SELL-side triggers when mark <= sl; BUY-side triggers when mark >= sl
                if order.sl is not None:
                    sl_triggered = (
                        (order.side == OrderSide.SELL and mark <= order.sl)
                        or (order.side == OrderSide.BUY and mark >= order.sl)
                    )
                    if sl_triggered:
                        fill = self._execute_fill(order, mark)
                        new_fills.append(fill)
                        continue
                # TP: SELL-side triggers when mark >= tp; BUY-side triggers when mark <= tp
                if order.tp is not None:
                    tp_triggered = (
                        (order.side == OrderSide.SELL and mark >= order.tp)
                        or (order.side == OrderSide.BUY and mark <= order.tp)
                    )
                    if tp_triggered:
                        fill = self._execute_fill(order, mark)
                        new_fills.append(fill)

        return new_fills

    # --- Internal helpers ----------------------------------------------------

    def _fill_market_order(self, order: Order, mark: float) -> Order:
        """Fill a market order with slippage and costs."""
        # Apply slippage: buy orders pay more, sell orders receive less
        slippage_factor = self._slippage_bps / 10_000.0
        if order.side == OrderSide.BUY:
            fill_price = mark * (1.0 + slippage_factor)
        else:
            fill_price = mark * (1.0 - slippage_factor)

        fill = self._create_fill(order, fill_price)
        self._update_position_and_balance(order, fill_price, fill.qty, fill.fee)
        filled = order.model_copy(
            update={
                "status": OrderStatus.FILLED,
                "broker_id": f"paper-{order.id[:8]}",
                "ts": _utc_now_iso(),
            }
        )
        self._orders[order.id] = filled
        return filled

    def _execute_fill(self, order: Order, fill_price: float) -> Fill:
        """Execute a fill for a pending order and update state."""
        fill = self._create_fill(order, fill_price)
        self._update_position_and_balance(order, fill_price, fill.qty, fill.fee)
        filled = order.model_copy(
            update={
                "status": OrderStatus.FILLED,
                "broker_id": f"paper-{order.id[:8]}",
                "ts": _utc_now_iso(),
            }
        )
        self._orders[order.id] = filled
        return fill

    def _create_fill(self, order: Order, fill_price: float) -> Fill:
        """Create a Fill record with cost model applied."""
        notional = order.qty * fill_price
        # CostModel expects lowercase "buy"/"sell", OrderSide values are "Buy"/"Sell"
        side_str = order.side.value.lower()
        fee = self._cost_model.total_cost(notional, side_str)

        fill = Fill(
            id=_uuid7(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=fee,
            fee_ccy=self._currency,
            ts=_utc_now_iso(),
        )
        self._fills.append(fill)
        return fill

    def _update_position_and_balance(
        self, order: Order, fill_price: float, qty: float, fee: float
    ) -> None:
        """Update position and balance after a fill, mirroring PaperTrader logic."""
        notional = qty * fill_price

        if order.symbol in self._positions:
            pos = self._positions[order.symbol]
            if order.side == pos.side:
                # Same-side add: weighted-average entry
                new_qty = pos.qty + qty
                new_avg = (pos.avg_entry * pos.qty + fill_price * qty) / new_qty
                new_cost = pos.entry_cost_cash + notional + fee
                self._positions[order.symbol] = _PaperPosition(
                    symbol=order.symbol,
                    side=pos.side,
                    qty=new_qty,
                    avg_entry=new_avg,
                    entry_cost_cash=new_cost,
                )
            else:
                # Opposite side: close or reduce
                close_qty = min(qty, pos.qty)
                remaining = pos.qty - close_qty
                # Realize PnL on closed portion
                pnl = close_qty * (fill_price - pos.avg_entry) * (
                    1.0 if pos.side == OrderSide.BUY else -1.0
                )
                # Return cash for closed portion
                self._balances[self._currency] += close_qty * fill_price - fee

                if remaining <= 0:
                    del self._positions[order.symbol]
                else:
                    self._positions[order.symbol] = _PaperPosition(
                        symbol=order.symbol,
                        side=pos.side,
                        qty=remaining,
                        avg_entry=pos.avg_entry,
                        entry_cost_cash=pos.entry_cost_cash * (remaining / pos.qty),
                    )
                # If qty > pos.qty, we've reversed — open opposite side
                open_qty = qty - close_qty
                if open_qty > 0:
                    self._positions[order.symbol] = _PaperPosition(
                        symbol=order.symbol,
                        side=order.side,
                        qty=open_qty,
                        avg_entry=fill_price,
                        entry_cost_cash=open_qty * fill_price,
                    )
        else:
            # New position
            self._balances[self._currency] -= notional + fee
            self._positions[order.symbol] = _PaperPosition(
                symbol=order.symbol,
                side=order.side,
                qty=qty,
                avg_entry=fill_price,
                entry_cost_cash=notional + fee,
            )

    def _unrealized_pnl(self, pos: _PaperPosition) -> float:
        """Compute unrealized PnL for a position using current mark price."""
        mark = self._marks.get(pos.symbol, pos.avg_entry)
        pnl_per_unit = mark - pos.avg_entry if pos.side == OrderSide.BUY else pos.avg_entry - mark
        return pnl_per_unit * pos.qty

    # --- Helpers for tests --------------------------------------------------

    @property
    def cash(self) -> float:
        return self._balances.get(self._currency, 0.0)

    @property
    def total_fills(self) -> int:
        return len(self._fills)

    def get_fills(self) -> list[Fill]:
        return list(self._fills)


__all__ = ["PaperBroker"]