"""Broker protocol and domain dataclasses for live trading.

The :class:`Broker` protocol defines the async interface that both
:class:`BybitBroker` (production) and :class:`PaperBroker` (simulation)
implement. Everything above the broker is broker-agnostic; swapping
testnet → mainnet is ``broker=BybitBroker(testnet=False)`` and nothing else.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "Buy"
    SELL = "Sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "Market"
    LIMIT = "Limit"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ORPHAN = "orphan"


# ---------------------------------------------------------------------------
# Dataclasses (pydantic v2 frozen/strict)
# ---------------------------------------------------------------------------


class Order(BaseModel, frozen=True, strict=True):
    """A trade order intent, persisted *before* submission to the broker."""

    id: str = Field(description="UUIDv7 intent ID")
    intent_id: str = Field(description="Idempotency key for retry logic")
    trace_id: str = Field(description="Correlation ID for observability")
    symbol: str = Field(description="Canonical symbol, e.g. BTC-USDT-PERP")
    side: OrderSide
    qty: float = Field(gt=0, description="Order quantity in base currency")
    order_type: OrderType
    price: float | None = Field(default=None, description="Limit price (None for market)")
    sl: float | None = Field(default=None, description="Stop-loss price")
    tp: float | None = Field(default=None, description="Take-profit price")
    reduce_only: bool = Field(
        default=False,
        description="If True, the order may only reduce an existing position (no new/opening exposure).",
    )
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    broker_id: str | None = Field(default=None, description="Broker-assigned order ID")
    ts: str = Field(description="ISO-8601 UTC timestamp when intent was created")


class Fill(BaseModel, frozen=True, strict=True):
    """A fill (partial or full) received from the broker."""

    id: str = Field(description="Fill UUID")
    order_id: str = Field(description="Links back to :class:`Order.id`")
    symbol: str
    side: OrderSide
    qty: float = Field(gt=0)
    price: float = Field(gt=0)
    fee: float = Field(default=0.0)
    fee_ccy: str = Field(default="USDT")
    ts: str = Field(description="ISO-8601 UTC timestamp of fill")


class Position(BaseModel, frozen=True, strict=True):
    """A snapshot of an open position."""

    symbol: str
    side: OrderSide
    qty: float = Field(gt=0)
    avg_entry: float = Field(gt=0, description="Average entry price")
    unrealized_pnl: float = Field(default=0.0)
    ts: str = Field(description="ISO-8601 UTC timestamp of snapshot")


class Balance(BaseModel, frozen=True, strict=True):
    """Account balance for a single currency."""

    currency: str = Field(default="USDT")
    available: float = Field(ge=0)
    total: float = Field(ge=0)
    ts: str = Field(description="ISO-8601 UTC timestamp of snapshot")


# ---------------------------------------------------------------------------
# Broker protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Broker(Protocol):
    """Async interface for order execution.

    Everything above this protocol is broker-agnostic. Swapping
    testnet → mainnet is ``BybitBroker(testnet=False)`` and nothing else.
    """

    async def place_order(self, order: Order) -> Order:
        """Submit an order to the broker. Returns the order with ``broker_id`` filled."""

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel a single order. Returns the updated order."""

    async def cancel_all(self, symbol: str) -> list[Order]:
        """Cancel all open orders for a symbol. Returns cancelled orders."""

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Fetch open positions. If ``symbol`` is None, return all."""

    async def get_balances(self) -> list[Balance]:
        """Fetch account balances across all currencies."""

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Fetch open orders. If ``symbol`` is None, return all."""

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol. Idempotent."""

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
        """Place a conditional (stop/limit) order with a trigger price."""


__all__ = [
    "Balance",
    "Broker",
    "Fill",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
]
