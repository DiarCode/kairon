"""Shared Bybit helpers used by both pybit and raw-request broker implementations.

This module contains pure, broker-agnostic utilities: symbol mapping, HMAC
signing helpers, id generation, timestamp formatting, response coercion, and
WebSocket message parsing. Keeping them in one place ensures the ``BybitBroker``
(pybit) and ``BybitRawBroker`` (httpx/websockets) implementations stay
consistent.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from kairon.data.symbols import CryptoVenue, Symbol
from kairon.live.broker.base import (
    Balance,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)


class BybitPermissionError(RuntimeError):
    """Raised when Bybit rejects an order due to regulatory/account restrictions.

    This typically maps to Bybit error code 10024: the API key's account is
    not permitted to trade the requested product (e.g. linear perpetuals in
    the current region).
    """

    def __init__(self, message: str, ret_code: int | None = None) -> None:
        super().__init__(message)
        self.ret_code = ret_code


class BybitAPIError(RuntimeError):
    """Raised for non-10024 Bybit API errors."""

    def __init__(self, message: str, ret_code: int | None = None) -> None:
        super().__init__(message)
        self.ret_code = ret_code


def symbol_to_bybit(symbol: Symbol) -> tuple[str, str]:
    """Convert a Kairon Symbol to (bybit_symbol, category).

    Returns:
        (bybit_symbol, category) — e.g. ("BTCUSDT", "linear")

    Raises:
        ValueError: if the symbol is not a crypto perpetual on Bybit.
    """
    if not symbol.is_perp:
        msg = f"Only perpetual symbols are supported for Bybit trading, got {symbol.canonical}"
        raise ValueError(msg)

    if symbol.venue != CryptoVenue.BYBIT:
        msg = f"Symbol venue must be BYBIT for BybitBroker, got {symbol.venue}"
        raise ValueError(msg)

    # Strip dashes and suffix: BTC-USDT-PERP → BTCUSDT
    # The canonical format is "BASE-QUOTE-PERP"
    parts = symbol.canonical.split("-")
    if len(parts) == 3 and parts[2] == "PERP":
        bybit_symbol = parts[0] + parts[1]
    elif len(parts) == 2:
        # Fallback: just concatenate base + quote
        bybit_symbol = parts[0] + parts[1]
    else:
        msg = f"Cannot map symbol {symbol.canonical} to Bybit format"
        raise ValueError(msg)

    return bybit_symbol, "linear"


def symbol_str_to_bybit(symbol_str: str) -> tuple[str, str]:
    """Convert a symbol string like 'BTC-USDT-PERP' to Bybit format.

    Returns (bybit_symbol, category).
    """
    if symbol_str.endswith("-PERP"):
        base_quote = symbol_str.replace("-PERP", "")
        bybit_symbol = base_quote.replace("-", "")
        return bybit_symbol, "linear"
    else:
        # Fallback: strip dashes
        bybit_symbol = symbol_str.replace("-", "")
        return bybit_symbol, "linear"


def bybit_to_symbol_str(bybit_symbol: str) -> str:
    """Convert a Bybit symbol like 'BTCUSDT' back to Kairon canonical format.

    Returns 'BTC-USDT-PERP'.
    """
    # Known USDT perpetuals
    known = {
        "BTCUSDT": "BTC-USDT-PERP",
        "ETHUSDT": "ETH-USDT-PERP",
        "SOLUSDT": "SOL-USDT-PERP",
        "XRPUSDT": "XRP-USDT-PERP",
        "DOGEUSDT": "DOGE-USDT-PERP",
        "ADAUSDT": "ADA-USDT-PERP",
        "AVAXUSDT": "AVAX-USDT-PERP",
        "DOTUSDT": "DOT-USDT-PERP",
        "MATICUSDT": "MATIC-USDT-PERP",
        "LINKUSDT": "LINK-USDT-PERP",
    }
    if bybit_symbol in known:
        return known[bybit_symbol]

    # Fallback: try to split at known quote currencies
    for quote in ["USDT", "USD", "BTC", "ETH"]:
        if bybit_symbol.endswith(quote) and len(bybit_symbol) > len(quote):
            base = bybit_symbol[: -len(quote)]
            return f"{base}-{quote}-PERP"

    # Last resort: return as-is
    return bybit_symbol


def order_side_to_bybit(side: OrderSide) -> str:
    """Convert OrderSide to Bybit's string format."""
    return side.value  # "Buy" or "Sell" — matches Bybit v5 API


def order_type_to_bybit(order_type: OrderType) -> str:
    """Convert OrderType to Bybit's string format."""
    return order_type.value  # "Market" or "Limit" — matches Bybit v5 API


def bybit_to_order_status(status: str) -> OrderStatus:
    """Map a Bybit orderStatus string to a Kairon OrderStatus."""
    mapping = {
        "Created": OrderStatus.SUBMITTED,
        "New": OrderStatus.SUBMITTED,
        "Active": OrderStatus.SUBMITTED,
        "PartiallyFilled": OrderStatus.PARTIAL,
        "Filled": OrderStatus.FILLED,
        "Cancelled": OrderStatus.CANCELLED,
        "Rejected": OrderStatus.REJECTED,
    }
    return mapping.get(status, OrderStatus.PENDING)




def parse_position(data: dict[str, Any]) -> Position | None:
    """Parse a Bybit position message into a Position dataclass.

    Returns ``None`` when the position size is zero (Bybit still emits an empty
    position record for symbols with no open exposure).
    """
    size = to_float(data.get("size", data.get("contracts", 0)))
    if size == 0:
        return None
    side = OrderSide.BUY if data.get("side") == "Buy" else OrderSide.SELL
    return Position(
        symbol=bybit_to_symbol_str(data.get("symbol", "")),
        side=side,
        qty=size,
        avg_entry=to_float(data.get("entryPrice", data.get("avgPrice", 0))) or 0.0,
        unrealized_pnl=to_float(data.get("unrealisedPnl", data.get("unrealisedPnl", 0))),
        ts=data.get("updatedTime", utc_now_iso()),
    )


def parse_fill_from_order(
    data: dict[str, Any],
    intent_to_local: dict[str, str],
    prev_cum_qty: dict[str, float],
) -> Fill | None:
    """Parse a Bybit order/execution message into an incremental Fill.

    Uses the ``orderLinkId`` (Kairon ``intent_id``) to resolve the local
    ``Order.id``. Computes incremental fill quantity by subtracting the
    previously recorded cumulative executed quantity for that order. Generates
    a unique fill id so repeated status updates do not overwrite the same row.

    Parameters
    ----------
    data:
        A Bybit v5 order or execution message. Expected keys include
        ``orderStatus``, ``orderLinkId``, ``orderId``, ``cumExecQty``,
        ``avgPrice``, ``cumExecFee``, ``execPrice``, ``execFee``,
        ``execSeq``, ``updatedTime``.
    intent_to_local:
        Mapping from Bybit ``orderLinkId`` to Kairon local ``Order.id``.
    prev_cum_qty:
        Mapping from local ``Order.id`` to the last recorded cumulative
        executed quantity. Updated in-place when a new incremental fill is
        detected.

    Returns
    -------
    Fill | None
        An incremental fill if the order status is ``Filled`` or
        ``PartiallyFilled`` and the cumulative quantity increased, otherwise
        ``None``.
    """
    order_status = data.get("orderStatus", "")
    if order_status not in ("Filled", "PartiallyFilled"):
        return None

    intent_id = data.get("orderLinkId", "")
    local_order_id = intent_to_local.get(intent_id)
    if local_order_id is None:
        logger.debug(
            "Ignoring Bybit fill: no local order mapping for orderLinkId %s",
            intent_id,
        )
        return None

    current_cum_qty = to_float(data.get("cumExecQty", 0))
    previous_cum_qty = prev_cum_qty.get(local_order_id, 0.0)
    incremental_qty = max(0.0, current_cum_qty - previous_cum_qty)
    if incremental_qty <= 0:
        return None

    prev_cum_qty[local_order_id] = current_cum_qty

    side = OrderSide.BUY if data.get("side") == "Buy" else OrderSide.SELL
    bybit_order_id = data.get("orderId", "")
    updated_time = data.get("updatedTime", utc_now_iso())
    exec_seq = data.get("execSeq", "0")
    fill_id = f"{bybit_order_id}-{updated_time}-{exec_seq}" if bybit_order_id else f"fill-{intent_id}-{updated_time}"

    # Prefer per-execution price/fee when available, otherwise fall back to
    # order-level cumulative values.
    price = to_float(data.get("execPrice", data.get("avgPrice", data.get("price", 0))))
    fee = to_float(data.get("execFee", data.get("cumExecFee", 0)))

    return Fill(
        id=fill_id,
        order_id=local_order_id,
        symbol=bybit_to_symbol_str(data.get("symbol", "")),
        side=side,
        qty=incremental_qty,
        price=price if price > 0 else to_float(data.get("avgPrice", 0)),
        fee=fee,
        fee_ccy="USDT",
        ts=updated_time,
    )


def parse_balances(response: dict[str, Any]) -> list[Balance]:
    """Parse Bybit wallet balance response into a list of Balance objects."""
    result = response.get("result", {})
    coins = result.get("list", [])
    if coins:
        # Unified account: first item has coin list
        coin_list = coins[0].get("coin", [])
    else:
        coin_list = []

    balances = []
    for coin in coin_list:
        balances.append(
            Balance(
                currency=coin.get("coin", "USDT"),
                available=to_float(coin.get("availableToWithdraw", coin.get("free", 0))),
                total=to_float(coin.get("walletBalance", coin.get("equity", 0))),
                ts=utc_now_iso(),
            )
        )

    # If the unified account has no coins yet (fresh testnet account),
    # still return USDT so downstream equity checks can reason about equity.
    if not balances:
        balances.append(
            Balance(
                currency="USDT",
                available=0.0,
                total=0.0,
                ts=utc_now_iso(),
            )
        )

    return balances


def parse_positions(response: dict[str, Any]) -> list[Position]:
    """Parse Bybit position list response into a list of Position objects."""
    result_list = response.get("result", {}).get("list", [])
    positions: list[Position] = []
    for item in result_list:
        pos = parse_position(item)
        if pos is not None:
            positions.append(pos)
    return positions


def parse_open_orders(response: dict[str, Any]) -> list[Order]:
    """Parse Bybit open orders response into a list of Order objects."""
    result_list = response.get("result", {}).get("list", [])
    orders = []
    for item in result_list:
        side = OrderSide.BUY if item.get("side") == "Buy" else OrderSide.SELL
        order_type = (
            OrderType.MARKET if item.get("orderType") == "Market" else OrderType.LIMIT
        )
        kairon_symbol = bybit_to_symbol_str(item.get("symbol", ""))
        orders.append(
            Order(
                id=item.get("orderId", ""),
                intent_id="",
                trace_id="",
                symbol=kairon_symbol,
                side=side,
                qty=to_float(item.get("qty", 0)),
                order_type=order_type,
                price=(to_float(item.get("price")) or None),
                sl=(to_float(item.get("stopLoss")) or None),
                tp=(to_float(item.get("takeProfit")) or None),
                status=bybit_to_order_status(item.get("orderStatus", "")),
                broker_id=item.get("orderId", ""),
                ts=item.get("updatedTime", utc_now_iso()),
            )
        )
    return orders


def to_float(raw: str | float | None) -> float:
    """Coerce a Bybit numeric string (or empty string) to float."""
    if raw is None or raw == "":
        return 0.0
    return float(raw)


def sleep_for_backoff(attempt: int, base_delay: float, max_delay: float) -> None:
    """Sleep for an exponentially increasing duration capped at ``max_delay``."""
    delay = min(base_delay * (2**attempt), max_delay)
    time.sleep(delay)


def uuid7() -> str:
    """Generate a unique ID (simplified UUID4 for tests)."""
    return uuid.uuid4().hex[:24]


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def tiny_positive_qty() -> float:
    """Return a tiny positive quantity usable as a placeholder in cancel returns.

    The :class:`Order` model requires ``qty > 0``, so cancelled orders that do
    not include a quantity from Bybit use this sentinel instead of ``0``.
    """
    return 1e-12


def format_bybit_error_message(ret_code: int, ret_msg: str) -> str:
    """Return a human-readable message for a Bybit API error."""
    return f"Bybit API error {ret_code}: {ret_msg}"


def is_permission_error(ret_code: int | None, ret_msg: str) -> bool:
    """Return True if Bybit response indicates a 10024 permission restriction."""
    if ret_code == 10024:
        return True
    return "regulatory restriction" in (ret_msg or "")


def is_insufficient_balance(ret_code: int | None, ret_msg: str) -> bool:
    """Return True if Bybit response indicates insufficient balance/margin."""
    if ret_code in (170131, 110007):
        return True
    msg = ret_msg or ""
    return any(
        marker in msg
        for marker in (
            "Insufficient balance",
            "InsufficientAB",
            "CheckMarginRatio",
            "Margin ratio",
            "margin",
        )
    )


def permission_error_message() -> str:
    """Standard explanation for Bybit error 10024."""
    return (
        "Bybit account is blocked by error 10024 (regulatory / "
        "product-eligibility restriction). Complete KYC/eligibility "
        "in the Bybit app, switch to demo-trading API keys, or use "
        "production keys after regional verification."
    )


__all__ = [
    "BybitAPIError",
    "BybitPermissionError",
    "bybit_to_order_status",
    "bybit_to_symbol_str",
    "format_bybit_error_message",
    "is_insufficient_balance",
    "is_permission_error",
    "order_side_to_bybit",
    "order_type_to_bybit",
    "parse_balances",
    "parse_fill_from_order",
    "parse_open_orders",
    "parse_position",
    "parse_positions",
    "permission_error_message",
    "sleep_for_backoff",
    "symbol_str_to_bybit",
    "symbol_to_bybit",
    "tiny_positive_qty",
    "to_float",
    "utc_now_iso",
    "uuid7",
]
