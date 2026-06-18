"""FakeBroker: in-memory Broker implementation for unit tests.

Deterministic fills at the mark price. Tracks orders, positions, and balances
in dicts. No network calls. Suitable for unit testing the Guardian,
Reconciler, and TradingLoop without touching a real exchange.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from kairon.live.broker.base import (
    Balance,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid7() -> str:
    """Generate a unique ID (simplified UUID4 for tests)."""
    return uuid.uuid4().hex[:24]


class FakeBroker:
    """In-memory Broker for unit tests.

    Fills market orders instantly at ``mark_price`` and limit orders when
    ``mark_price`` crosses the limit price. The caller can set ``mark_price``
    between ticks to simulate price changes.
    """

    def __init__(
        self,
        *,
        initial_balance: float = 10_000.0,
        mark_price: float = 50_000.0,
        currency: str = "USDT",
    ) -> None:
        self.mark_price = mark_price
        self._balances: dict[str, float] = {currency: initial_balance}
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._currency = currency
        self._leverage: dict[str, int] = {}

    async def place_order(self, order: Order) -> Order:
        """Submit an order. Market orders fill instantly at mark_price."""
        if order.id in self._orders:
            # Idempotent: return existing order
            return self._orders[order.id]

        filled = order.model_copy(
            update={
                "status": OrderStatus.FILLED,
                "broker_id": f"fake-{order.id[:8]}",
                "ts": _utc_now_iso(),
            }
        )
        self._orders[order.id] = filled

        # Create fill
        fill = Fill(
            id=_uuid7(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=self.mark_price,
            fee=order.qty * self.mark_price * 0.001,  # 10bps fee
            fee_ccy=self._currency,
            ts=_utc_now_iso(),
        )
        self._fills.append(fill)

        # Update position
        self._update_position(filled, fill)

        return filled

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel a pending order."""
        order = self._orders.get(order_id)
        if order is None:
            msg = f"Order {order_id} not found"
            raise KeyError(msg)
        cancelled = order.model_copy(
            update={"status": OrderStatus.CANCELLED, "ts": _utc_now_iso()}
        )
        self._orders[order_id] = cancelled
        return cancelled

    async def cancel_all(self, symbol: str) -> list[Order]:
        """Cancel all open orders for a symbol."""
        cancelled: list[Order] = []
        for oid, order in list(self._orders.items()):
            if order.symbol == symbol and order.status in (
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIAL,
            ):
                cancelled_order = order.model_copy(
                    update={"status": OrderStatus.CANCELLED, "ts": _utc_now_iso()}
                )
                self._orders[oid] = cancelled_order
                cancelled.append(cancelled_order)
        return cancelled

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Return open positions, optionally filtered by symbol."""
        if symbol:
            pos = self._positions.get(symbol)
            return [pos] if pos else []
        return list(self._positions.values())

    async def get_balances(self) -> list[Balance]:
        """Return account balances."""
        return [
            Balance(
                currency=ccy,
                available=amt,
                total=amt,
                ts=_utc_now_iso(),
            )
            for ccy, amt in self._balances.items()
        ]

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Return open orders, optionally filtered by symbol."""
        orders = [
            o
            for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
        ]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol. Idempotent."""
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
        """Place a conditional order (simplified: stores as pending)."""
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
            broker_id=f"fake-cond-{_uuid7()[:8]}",
            ts=_utc_now_iso(),
        )
        self._orders[order.id] = order
        return order

    # ------------------------------------------------------------------
    # Helpers for tests
    # ------------------------------------------------------------------

    def _update_position(self, order: Order, fill: Fill) -> None:
        """Update position dict after a fill."""
        symbol = order.symbol
        if symbol in self._positions:
            existing = self._positions[symbol]
            new_qty = existing.qty + fill.qty if order.side == OrderSide.BUY else existing.qty - fill.qty
            if new_qty <= 0:
                del self._positions[symbol]
            else:
                self._positions[symbol] = existing.model_copy(
                    update={"qty": new_qty, "ts": _utc_now_iso()}
                )
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                side=order.side,
                qty=fill.qty,
                avg_entry=fill.price,
                unrealized_pnl=0.0,
                ts=_utc_now_iso(),
            )


__all__ = ["FakeBroker"]