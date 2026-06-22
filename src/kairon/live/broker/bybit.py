"""BybitBroker: live broker implementation wrapping pybit v5 HTTP and WebSocket.

Uses ``pybit.unified_trading.HTTP`` for order management and
``pybit.unified_trading.WebSocket`` for private channel streaming
(position updates, order fills). Falls back to REST polling when
WebSocket is disconnected.

This module is the *only* place in ``kairon.live`` that imports pybit;
everything above the Broker protocol boundary is broker-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

from kairon.data.symbols import CryptoVenue, crypto_perp
from kairon.live.broker.base import (
    Balance,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.broker.bybit_shared import (
    BybitPermissionError,
    bybit_to_order_status,
    bybit_to_symbol_str,
    is_insufficient_balance,
    is_permission_error,
    order_side_to_bybit,
    order_type_to_bybit,
    parse_balances,
    parse_fill_from_order,
    parse_open_orders,
    parse_position,
    parse_positions,
    permission_error_message,
    sleep_for_backoff,
    symbol_str_to_bybit,
    symbol_to_bybit,
    tiny_positive_qty,
    to_float,
    utc_now_iso,
    uuid7,
)

logger = logging.getLogger(__name__)

# Known lot-size filters for common USDT linear perpetuals on Bybit.
# Maps Kairon canonical symbol -> (qty_step, min_order_qty).
_QTY_LOT_SIZES: dict[str, tuple[float, float]] = {
    "BTC-USDT-PERP": (0.001, 0.001),
    "ETH-USDT-PERP": (0.01, 0.01),
    "XRP-USDT-PERP": (0.1, 0.1),
    "SOL-USDT-PERP": (0.1, 0.1),
    "DOGEUSDT": (1.0, 1.0),  # legacy canonical
    "ADA-USDT-PERP": (1.0, 1.0),
    "AVAX-USDT-PERP": (0.01, 0.01),
    "DOT-USDT-PERP": (0.1, 0.1),
    "LINK-USDT-PERP": (0.01, 0.01),
    "MATIC-USDT-PERP": (1.0, 1.0),
}

# Known price tick sizes for limit-order pricing in close_position fallback.
_PRICE_TICK_SIZES: dict[str, float] = {
    "BTC-USDT-PERP": 0.1,
    "ETH-USDT-PERP": 0.01,
    "XRP-USDT-PERP": 0.0001,
    "SOL-USDT-PERP": 0.001,
    "DOGEUSDT": 0.00001,
    "ADA-USDT-PERP": 0.00001,
    "AVAX-USDT-PERP": 0.001,
    "DOT-USDT-PERP": 0.001,
    "LINK-USDT-PERP": 0.001,
    "MATIC-USDT-PERP": 0.00001,
}


def _qty_step_and_min(symbol: str) -> tuple[float, float]:
    """Return (qty_step, min_order_qty) for a Kairon canonical symbol."""
    return _QTY_LOT_SIZES.get(symbol, (0.001, 0.001))


def _round_qty(symbol: str, qty: float) -> float:
    """Round quantity down to Bybit's lot size and enforce minimum order size."""
    step, min_qty = _qty_step_and_min(symbol)
    if not math.isfinite(qty) or qty <= 0:
        return 0.0
    rounded = math.floor(qty / step) * step
    # Round to a sane number of decimals to avoid float noise.
    decimals = max(0, int(-math.log10(step)))
    rounded = round(rounded, decimals)
    if rounded < min_qty:
        return 0.0
    return rounded


def _round_price(symbol: str, price: float) -> float:
    """Round a limit price to Bybit's price tick size.

    Rounds away from zero to avoid crossing the spread, then clips to the
    tick-size precision.
    """
    tick = _PRICE_TICK_SIZES.get(symbol, 0.01)
    if not math.isfinite(price) or tick <= 0:
        return price
    decimals = max(0, int(-math.log10(tick)))
    return round(price, decimals)


class BybitBroker:
    """Live broker implementation wrapping pybit v5 HTTP and WebSocket.

    Uses ``pybit.unified_trading.HTTP`` for synchronous REST operations
    (place_order, cancel, get_positions, etc.) and
    ``pybit.unified_trading.WebSocket`` for private channel streaming
    (position updates, order fills).

    All REST calls are wrapped in ``asyncio.to_thread`` to avoid blocking
    the event loop since pybit's HTTP client is synchronous.

    Args:
        api_key: Bybit API key.
        api_secret: Bybit API secret.
        testnet: If True, connect to Bybit testnet instead of mainnet.
        tld: Bybit top-level domain suffix (com, kz, hk, eu, nl). Defaults to
            "com"; use "kz" for testnet.bybit.kz accounts.
        max_reconnect_attempts: Maximum consecutive reconnection attempts.
        reconnect_base_delay: Base delay (seconds) for exponential backoff.
        reconnect_max_delay: Maximum delay (seconds) between reconnection attempts.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        tld: str = "com",
        max_reconnect_attempts: int = 10,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._tld = tld
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay

        # Lazy imports — pybit is an optional [live] dependency
        self._http: Any = None  # pybit.unified_trading.HTTP
        self._ws: Any = None  # pybit.unified_trading.WebSocket (private)
        self._ws_connected: bool = False
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue()
        self._position_queue: asyncio.Queue[Position] = asyncio.Queue()

        # Clock skew detection
        self._server_time_offset: float | None = None  # seconds

        # Intent / cumulative-qty tracking for incremental fill capture
        self._intent_to_local: dict[str, str] = {}
        self._order_cum_qty: dict[str, float] = {}

    # --- Lazy initialization --------------------------------------------------

    def _ensure_http(self) -> Any:
        """Lazily initialize the HTTP session."""
        if self._http is None:
            from pybit.unified_trading import HTTP

            self._http = HTTP(
                testnet=self._testnet,
                api_key=self._api_key,
                api_secret=self._api_secret,
                tld=self._tld,
                log_requests=False,
            )
            logger.info(
                "BybitBroker HTTP session initialized (testnet=%s, tld=%s)",
                self._testnet,
                self._tld,
            )
        return self._http

    # --- WebSocket lifecycle --------------------------------------------------

    async def connect_websocket(self) -> None:
        """Connect to Bybit's private WebSocket for order and position streams.

        On disconnect, automatic reconnection with exponential backoff
        is handled internally. Falls back to REST polling when disconnected.
        """
        try:
            from pybit.unified_trading import WebSocket
        except ImportError:
            logger.error("pybit is not installed. Install with: pip install pybit")
            raise

        if self._ws is not None:
            return  # Already connected

        attempt = 0
        while attempt < self._max_reconnect_attempts:
            try:
                self._ws = WebSocket(
                    testnet=self._testnet,
                    channel_type="private",
                    api_key=self._api_key,
                    api_secret=self._api_secret,
                    tld=self._tld,
                )
                # Subscribe to position and order streams
                self._ws.position_stream(self._on_position_message)
                self._ws.order_stream(self._on_order_message)
                self._ws_connected = True
                logger.info(
                    "BybitBroker WebSocket connected (testnet=%s, tld=%s)",
                    self._testnet,
                    self._tld,
                )
                return
            except Exception as e:
                attempt += 1
                delay = min(
                    self._reconnect_base_delay * (2 ** (attempt - 1)),
                    self._reconnect_max_delay,
                )
                logger.warning(
                    "WebSocket connection failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self._max_reconnect_attempts,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

        logger.error("WebSocket connection failed after %d attempts", self._max_reconnect_attempts)
        self._ws_connected = False

    def _on_position_message(self, message: dict[str, Any]) -> None:
        """Callback for WebSocket position stream messages."""
        try:
            data = message.get("data", {})
            if isinstance(data, list):
                for pos_data in data:
                    position = parse_position(pos_data)
                    if position is not None:
                        self._position_queue.put_nowait(position)
            elif isinstance(data, dict):
                position = parse_position(data)
                if position is not None:
                    self._position_queue.put_nowait(position)
        except Exception as e:
            logger.error("Error parsing position message: %s", e)

    def _on_order_message(self, message: dict[str, Any]) -> None:
        """Callback for WebSocket order stream messages."""
        try:
            data = message.get("data", {})
            if isinstance(data, list):
                for order_data in data:
                    fill = parse_fill_from_order(
                        order_data, self._intent_to_local, self._order_cum_qty
                    )
                    if fill is not None:
                        self._fill_queue.put_nowait(fill)
            elif isinstance(data, dict):
                fill = parse_fill_from_order(
                    data, self._intent_to_local, self._order_cum_qty
                )
                if fill is not None:
                    self._fill_queue.put_nowait(fill)
        except Exception as e:
            logger.error("Error parsing order message: %s", e)

    # --- Broker protocol implementation ---------------------------------------

    async def place_order(self, order: Order) -> Order:
        """Submit an order to Bybit. Returns the order with broker_id filled."""
        http = self._ensure_http()
        base, quote = order.symbol.replace("-PERP", "").split("-")
        bybit_symbol, category = symbol_to_bybit(
            crypto_perp(base, quote, CryptoVenue.BYBIT)
        )

        # Check clock skew (non-blocking)
        await self._check_clock_skew(http)

        # Enforce Bybit lot-size / min-order rules so the request is not rejected.
        rounded_qty = _round_qty(order.symbol, order.qty)
        if rounded_qty <= 0:
            logger.warning(
                "BybitBroker.place_order rejected: qty %.6f rounds below min lot for %s",
                order.qty, order.symbol,
            )
            return order.model_copy(
                update={"status": OrderStatus.REJECTED, "ts": utc_now_iso()}
            )

        # Track the intent -> local order mapping before submission so that
        # WebSocket fills can resolve back to our local Order.id.
        self._intent_to_local[order.intent_id] = order.id

        params: dict[str, Any] = {
            "category": category,
            "symbol": bybit_symbol,
            "side": order_side_to_bybit(order.side),
            "orderType": order_type_to_bybit(order.order_type),
            "qty": str(rounded_qty),
            "orderLinkId": order.intent_id,
        }

        if order.price is not None:
            params["price"] = str(order.price)
        if order.sl is not None:
            params["stopLoss"] = str(_round_price(order.symbol, float(order.sl)))
        if order.tp is not None:
            params["takeProfit"] = str(_round_price(order.symbol, float(order.tp)))
        if order.reduce_only:
            # Marks the order as position-reducing only. Bybit rejects a
            # reduce-only order that would not reduce an open position, so this
            # must only be set on genuine close/reduce orders (software stops,
            # signal-flip flattens, session-end flatten).
            params["reduceOnly"] = "true"

        try:
            response = await asyncio.to_thread(http.place_order, **params)
            result = response.get("result", {})
            broker_id = result.get("orderId", "")

            # Status from the wire: Market orders usually fill quickly, but the
            # initial ack is "Created"/"New". The WS fill stream updates to
            # FILLED/PARTIAL; we report SUBMITTED here to stay honest.
            filled = order.model_copy(
                update={
                    "status": OrderStatus.SUBMITTED,
                    "broker_id": broker_id or f"bybit-{order.id[:8]}",
                    "qty": rounded_qty,
                    "ts": utc_now_iso(),
                }
            )
            return filled

        except Exception as e:
            msg = str(e)
            logger.error("BybitBroker.place_order failed: %s", e)
            if is_permission_error(None, msg):
                raise BybitPermissionError(
                    "Bybit rejected the order with error 10024 (regulatory / "
                    "product-eligibility restriction). This is an account-level "
                    "block, not a code bug. Common fixes: (1) complete KYC / "
                    "accept the derivatives-trading prompt in the Bybit app or "
                    "on testnet.bybit.kz, (2) switch to Bybit demo-trading API "
                    "keys, or (3) use production keys after regional verification."
                ) from e
            # Return order marked as rejected
            return order.model_copy(
                update={"status": OrderStatus.REJECTED, "ts": utc_now_iso()}
            )

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel a single order on Bybit."""
        http = self._ensure_http()
        bybit_symbol, category = symbol_str_to_bybit(symbol)

        try:
            await asyncio.to_thread(
                http.cancel_order,
                category=category,
                symbol=bybit_symbol,
                orderId=order_id,
            )
            # Return a cancelled version of the order
            return Order(
                id=order_id,
                intent_id="",
                trace_id="",
                symbol=symbol,
                side=OrderSide.BUY,  # Side doesn't matter for cancelled
                qty=tiny_positive_qty(),
                order_type=OrderType.MARKET,
                status=OrderStatus.CANCELLED,
                ts=utc_now_iso(),
            )
        except Exception as e:
            logger.error("BybitBroker.cancel_order failed: %s", e)
            raise

    async def cancel_all(self, symbol: str) -> list[Order]:
        """Cancel all open orders for a symbol on Bybit."""
        http = self._ensure_http()
        bybit_symbol, category = symbol_str_to_bybit(symbol)

        try:
            response = await asyncio.to_thread(
                http.cancel_all_orders,
                category=category,
                symbol=bybit_symbol,
            )
            result_list = response.get("result", {}).get("list", [])
            cancelled = []
            for item in result_list:
                cancelled.append(
                    Order(
                        id=item.get("orderId", ""),
                        intent_id="",
                        trace_id="",
                        symbol=symbol,
                        side=OrderSide.BUY,
                        qty=tiny_positive_qty(),
                        order_type=OrderType.MARKET,
                        status=OrderStatus.CANCELLED,
                        ts=utc_now_iso(),
                    )
                )
            return cancelled
        except Exception as e:
            logger.error("BybitBroker.cancel_all failed: %s", e)
            return []

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Fetch open positions from Bybit."""
        http = self._ensure_http()

        try:
            params: dict[str, Any] = {"category": "linear"}
            if symbol:
                bybit_symbol, _ = symbol_str_to_bybit(symbol)
                params["symbol"] = bybit_symbol
            else:
                # Unified-account linear positions require settleCoin when not filtering by symbol.
                params["settleCoin"] = "USDT"

            response = await asyncio.to_thread(http.get_positions, **params)
            return parse_positions(response)
        except Exception as e:
            logger.error("BybitBroker.get_positions failed: %s", e)
            return []

    async def get_balances(self) -> list[Balance]:
        """Fetch account balances from Bybit."""
        http = self._ensure_http()

        try:
            response = await asyncio.to_thread(
                http.get_wallet_balance,
                accountType="UNIFIED",
            )
            return parse_balances(response)
        except Exception as e:
            logger.error("BybitBroker.get_balances failed: %s", e)
            return [Balance(currency="USDT", available=0.0, total=0.0, ts=utc_now_iso())]

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Fetch open orders from Bybit."""
        http = self._ensure_http()

        try:
            params: dict[str, Any] = {"category": "linear"}
            if symbol:
                bybit_symbol, _ = symbol_str_to_bybit(symbol)
                params["symbol"] = bybit_symbol
            else:
                params["settleCoin"] = "USDT"

            response = await asyncio.to_thread(http.get_open_orders, **params)
            return parse_open_orders(response)
        except Exception as e:
            logger.error("BybitBroker.get_open_orders failed: %s", e)
            return []

    async def get_last_price(self, symbol: str) -> float | None:
        """Fetch the live last price for a symbol from Bybit.

        Used by the scalping orchestrator to re-anchor ATR-based TP/SL to the
        current market price at order time — the strategy computes stops
        relative to the 1m bar close, which can differ materially from the live
        price on a volatile book, and Bybit validates attached TP/SL against
        the live last price (not the bar close). Returns ``None`` on failure so
        callers can fall back to the bar-close-anchored stops.
        """
        http = self._ensure_http()
        try:
            bybit_symbol, category = symbol_str_to_bybit(symbol)
            tickers = await asyncio.to_thread(
                http.get_tickers, category=category, symbol=bybit_symbol
            )
            ticker_list = tickers.get("result", {}).get("list", [])
            if not ticker_list:
                return None
            # Prefer lastPrice (trade price); fall back to markPrice.
            price = ticker_list[0].get("lastPrice") or ticker_list[0].get("markPrice")
            return to_float(price) if price not in (None, "") else None
        except Exception as e:
            logger.debug("BybitBroker.get_last_price failed for %s: %s", symbol, e)
            return None


    async def get_orderbook(self, symbol: str, depth: int = 5) -> dict | None:
        """Fetch the live order book (top ``depth`` bid/ask levels) for a symbol.

        Returns ``{"bids": [[price, size], ...], "asks": [[price, size], ...]}``
        normalized from Bybit's ``get_orderbook`` response, or ``None`` on
        failure (thin testnet book, transient API error). Used by the Phase 4b
        order-flow poller to compute imbalance / spread / depth for entry-timing.
        """
        http = self._ensure_http()
        try:
            bybit_symbol, category = symbol_str_to_bybit(symbol)
            ob = await asyncio.to_thread(
                http.get_orderbook, category=category, symbol=bybit_symbol, limit=depth
            )
        except Exception as e:
            logger.debug("BybitBroker.get_orderbook failed for %s: %s", symbol, e)
            return None
        result = (ob or {}).get("result") or {}
        return {
            "bids": list(result.get("b") or []),
            "asks": list(result.get("a") or []),
        }


    async def get_closed_pnl(
        self, symbol: str, *, limit: int = 50, start_time_ms: int | None = None, end_time_ms: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch realized closed PnL for a symbol from Bybit.

        Wraps ``pybit.unified_trading.HTTP.get_closed_pnl``. Useful for
        post-session analytics because it reports the exchange's own
        realized PnL per closed trade.
        """
        http = self._ensure_http()
        bybit_symbol, category = symbol_str_to_bybit(symbol)
        params: dict[str, Any] = {
            "category": category,
            "symbol": bybit_symbol,
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        try:
            response = await asyncio.to_thread(http.get_closed_pnl, **params)
            return response.get("result", {}).get("list", [])
        except Exception as e:
            logger.error("BybitBroker.get_closed_pnl failed for %s: %s", symbol, e)
            return []

    async def poll_executions(
        self, symbol: str, *, limit: int = 50
    ) -> list[Fill]:
        """REST fallback that polls recent executions and yields incremental fills.

        Queries ``pybit.unified_trading.HTTP.get_executions`` for the given
        symbol, maps each execution back to the local ``Order.id`` via
        ``orderLinkId``, and returns only executions that represent new
        incremental quantity since the last poll.
        """
        http = self._ensure_http()
        bybit_symbol, category = symbol_str_to_bybit(symbol)
        params: dict[str, Any] = {
            "category": category,
            "symbol": bybit_symbol,
            "limit": limit,
        }
        try:
            response = await asyncio.to_thread(http.get_executions, **params)
            executions = response.get("result", {}).get("list", [])
        except Exception as e:
            logger.error("BybitBroker.poll_executions failed for %s: %s", symbol, e)
            return []

        fills: list[Fill] = []
        for exec_data in executions:
            fill = self.get_incremental_fill(exec_data)
            if fill is not None:
                fills.append(fill)
        return fills

    async def close_position(
        self, symbol: str, *, chunk_qty: float | None = None
    ) -> Order:
        """Close the entire open position for a symbol using reduce-only orders.

        Attempts to close in chunks with reduce-only market orders. If a market
        order fails because there is no immediate quantity to fill (common on
        illiquid testnet books), falls back to a limit order placed slightly
        inside the spread and retries.

        Parameters
        ----------
        symbol:
            Kairon canonical symbol, e.g. "ETH-USDT-PERP".
        chunk_qty:
            Maximum quantity per close order. If ``None``, a symbol-specific
            default is used (0.02 BTC, 0.2 ETH, 1.0 for others).

        Returns
        -------
        Order
            The last order submitted (status reflects the final attempt).
        """
        http = self._ensure_http()
        bybit_symbol, category = symbol_str_to_bybit(symbol)

        if chunk_qty is None:
            chunk_qty = {
                "BTC-USDT-PERP": 0.02,
                "ETH-USDT-PERP": 0.2,
            }.get(symbol, 1.0)

        positions = await self.get_positions(symbol)
        pos = next((p for p in positions if p.symbol == symbol), None)
        if pos is None or pos.qty <= 1e-9:
            return Order(
                id=f"close-none-{uuid7()[:8]}",
                intent_id=uuid7(),
                trace_id="",
                symbol=symbol,
                side=OrderSide.BUY,
                qty=tiny_positive_qty(),
                order_type=OrderType.MARKET,
                status=OrderStatus.FILLED,
                ts=utc_now_iso(),
            )

        close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        remaining = pos.qty
        last_order: Order | None = None
        max_retries = 3

        # Fetch mark price once for limit fallback pricing.
        mark_price = pos.avg_entry
        try:
            tickers = await asyncio.to_thread(
                http.get_tickers, category=category, symbol=bybit_symbol
            )
            ticker_list = tickers.get("result", {}).get("list", [])
            if ticker_list:
                mark_price = to_float(ticker_list[0].get("markPrice", mark_price))
        except Exception as e:
            logger.debug("Could not fetch mark price for %s close: %s", symbol, e)

        # Position-driven close loop. The thin testnet order book means a single
        # market reduce-only order often only partially fills (e.g. 0.001 of a
        # 0.015 chunk), so we must re-read the ACTUAL position after each order
        # and measure real progress, rather than assuming the requested chunk
        # filled. We chip away in lots up to ``chunk_qty``, falling back to a
        # crossing IOC limit at the best opposing-side price when market orders
        # stall, and give up only after several consecutive no-progress rounds.
        _, min_qty = _qty_step_and_min(symbol)
        no_progress = 0
        max_rounds = 100
        for _round in range(max_rounds):
            positions = await self.get_positions(symbol)
            pos = next((p for p in positions if p.symbol == symbol), None)
            remaining = pos.qty if pos else 0.0
            if remaining <= 1e-9:
                break

            this_chunk = min(chunk_qty, remaining)
            rounded_chunk = max(_round_qty(symbol, this_chunk), min_qty)
            if rounded_chunk <= 0:
                break

            order_id = f"close-{uuid7()[:8]}"
            intent_id = uuid7()
            self._intent_to_local[intent_id] = order_id

            market_params: dict[str, Any] = {
                "category": category,
                "symbol": bybit_symbol,
                "side": order_side_to_bybit(close_side),
                "orderType": "Market",
                "qty": str(rounded_chunk),
                "orderLinkId": intent_id,
                "reduceOnly": True,
            }

            placed = False
            for attempt in range(max_retries):
                try:
                    response = await asyncio.to_thread(http.place_order, **market_params)
                    result = response.get("result", {})
                    last_order = Order(
                        id=order_id,
                        intent_id=intent_id,
                        trace_id="",
                        symbol=symbol,
                        side=close_side,
                        qty=rounded_chunk,
                        order_type=OrderType.MARKET,
                        status=OrderStatus.SUBMITTED,
                        broker_id=result.get("orderId", "") or None,
                        ts=utc_now_iso(),
                    )
                    placed = True
                    break
                except Exception as e:
                    msg = str(e)
                    if "NoImmediateQtyToFill" in msg or "170135" in msg:
                        logger.warning(
                            "Market close failed for %s (attempt %d/%d): %s",
                            symbol, attempt + 1, max_retries, msg,
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0)
                        continue
                    logger.error("close_position market order failed for %s: %s", symbol, e)
                    break

            await asyncio.sleep(0.4)

            # Measure actual progress. If the market order did not reduce the
            # position (thin book / max-buy-price cap), fall back to a crossing
            # IOC limit at the best opposing-side price.
            positions2 = await self.get_positions(symbol)
            pos2 = next((p for p in positions2 if p.symbol == symbol), None)
            new_remaining = pos2.qty if pos2 else 0.0
            if new_remaining >= remaining - 1e-9 and new_remaining > 1e-9:
                cross_price = await self._best_crossing_price(
                    http, category, bybit_symbol, close_side, mark_price
                )
                if cross_price is not None:
                    limit_intent = uuid7()
                    self._intent_to_local[limit_intent] = order_id
                    limit_params: dict[str, Any] = {
                        "category": category,
                        "symbol": bybit_symbol,
                        "side": order_side_to_bybit(close_side),
                        "orderType": "Limit",
                        "qty": str(min(rounded_chunk, new_remaining)),
                        "price": str(cross_price),
                        "timeInForce": "IOC",
                        "orderLinkId": limit_intent,
                        "reduceOnly": True,
                    }
                    try:
                        response = await asyncio.to_thread(http.place_order, **limit_params)
                        result = response.get("result", {})
                        last_order = Order(
                            id=order_id,
                            intent_id=limit_intent,
                            trace_id="",
                            symbol=symbol,
                            side=close_side,
                            qty=rounded_chunk,
                            order_type=OrderType.LIMIT,
                            status=OrderStatus.SUBMITTED,
                            broker_id=result.get("orderId") or None,
                            ts=utc_now_iso(),
                        )
                    except Exception as e:
                        logger.debug("close_position crossing limit failed for %s: %s", symbol, e)
                    await asyncio.sleep(0.4)

                # Re-measure progress after the fallback.
                positions3 = await self.get_positions(symbol)
                pos3 = next((p for p in positions3 if p.symbol == symbol), None)
                new_remaining = pos3.qty if pos3 else 0.0
                if new_remaining >= remaining - 1e-9:
                    no_progress += 1
                    if no_progress >= 5:
                        logger.warning(
                            "close_position made no progress for %s after %d rounds; "
                            "residual=%.6f remains",
                            symbol, no_progress, new_remaining,
                        )
                        break
                else:
                    no_progress = 0
            else:
                no_progress = 0

        # Refresh position and, if residual remains, report the last order as partial.
        final_positions = await self.get_positions(symbol)
        final_pos = next((p for p in final_positions if p.symbol == symbol), None)
        if last_order is not None and final_pos is not None and final_pos.qty > 1e-9:
            last_order = last_order.model_copy(
                update={"status": OrderStatus.PARTIAL, "ts": utc_now_iso()}
            )

        if last_order is None:
            last_order = Order(
                id=f"close-empty-{uuid7()[:8]}",
                intent_id=uuid7(),
                trace_id="",
                symbol=symbol,
                side=OrderSide.BUY,
                qty=tiny_positive_qty(),
                order_type=OrderType.MARKET,
                status=OrderStatus.FILLED,
                ts=utc_now_iso(),
            )
        return last_order

    async def _best_crossing_price(
        self,
        http: Any,
        category: str,
        bybit_symbol: str,
        close_side: OrderSide,
        fallback_mark: float,
    ) -> float | None:
        """Best price to cross the book for a reduce-only close order.

        To close a SHORT (close_side=BUY) we lift the best ASK; to close a LONG
        (close_side=SELL) we hit the best BID. Returns None if the book is
        empty on the needed side.
        """
        try:
            ob = await asyncio.to_thread(
                http.get_orderbook, category=category, symbol=bybit_symbol, limit=5
            )
        except Exception as e:
            logger.debug("orderbook fetch failed for %s: %s", bybit_symbol, e)
            return None
        # BUY crosses asks; SELL crosses bids.
        side_key = "a" if close_side == OrderSide.BUY else "b"
        levels = (ob.get("result") or {}).get(side_key, []) or []
        for lvl in levels:
            price = to_float(lvl[0]) if len(lvl) > 0 else 0.0
            size = to_float(lvl[1]) if len(lvl) > 1 else 0.0
            if price > 0 and size > 0:
                return price
        return fallback_mark if fallback_mark > 0 else None

    def min_qty_for(self, symbol: str) -> float:
        """Return the minimum order quantity (lot size) for a symbol."""
        return _qty_step_and_min(symbol)[1]

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol on Bybit. Idempotent."""
        http = self._ensure_http()
        bybit_symbol, category = symbol_str_to_bybit(symbol)

        try:
            await asyncio.to_thread(
                http.set_leverage,
                category=category,
                symbol=bybit_symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            logger.info("Set leverage for %s to %dx", symbol, leverage)
        except Exception as e:
            msg = str(e)
            # Bybit error 110043 = leverage not modified (already at target).
            if "110043" in msg:
                logger.info("Leverage for %s is already %dx", symbol, leverage)
                return
            logger.error("BybitBroker.set_leverage failed: %s", e)
            raise

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
        """Place a conditional (stop/limit) order with a trigger price on Bybit."""
        http = self._ensure_http()
        bybit_symbol, category = symbol_str_to_bybit(symbol)

        rounded_qty = _round_qty(symbol, qty)
        if rounded_qty <= 0:
            logger.warning(
                "BybitBroker.place_conditional rejected: qty %.6f rounds below min lot for %s",
                qty, symbol,
            )
            return Order(
                id=f"bybit-cond-rejected-{uuid7()[:8]}",
                intent_id=uuid7(),
                trace_id="",
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=order_type,
                price=trigger_price,
                sl=sl,
                tp=tp,
                status=OrderStatus.REJECTED,
                ts=utc_now_iso(),
            )

        intent_id = uuid7()
        params: dict[str, Any] = {
            "category": category,
            "symbol": bybit_symbol,
            "side": order_side_to_bybit(side),
            "orderType": order_type_to_bybit(order_type),
            "qty": str(rounded_qty),
            "triggerPrice": str(trigger_price),
            "orderLinkId": intent_id,
        }

        if sl is not None:
            params["stopLoss"] = str(sl)
        if tp is not None:
            params["takeProfit"] = str(tp)

        try:
            response = await asyncio.to_thread(http.place_order, **params)
            result = response.get("result", {})
            broker_id = result.get("orderId", "")

            return Order(
                id=broker_id or f"bybit-cond-{intent_id[:8]}",
                intent_id=intent_id,
                trace_id="",
                symbol=symbol,
                side=side,
                qty=rounded_qty,
                order_type=order_type,
                price=trigger_price,
                sl=sl,
                tp=tp,
                status=OrderStatus.PENDING,
                broker_id=broker_id,
                ts=utc_now_iso(),
            )
        except Exception as e:
            msg = str(e)
            logger.error("BybitBroker.place_conditional failed: %s", e)
            if is_permission_error(None, msg):
                raise BybitPermissionError(
                    "Bybit rejected the conditional order with error 10024 "
                    "(regulatory / product-eligibility restriction). See the "
                    "diagnosis in BybitBroker.place_order for fixes."
                ) from e
            raise

    # --- WebSocket fill/position stream ------------------------------------

    async def stream_fills(self) -> asyncio.Queue[Fill]:
        """Return the queue that receives fill events from the WebSocket."""
        if not self._ws_connected:
            await self.connect_websocket()
        return self._fill_queue

    async def stream_positions(self) -> asyncio.Queue[Position]:
        """Return the queue that receives position events from the WebSocket."""
        if not self._ws_connected:
            await self.connect_websocket()
        return self._position_queue

    def get_incremental_fill(self, data: dict[str, Any]) -> Fill | None:
        """Parse a Bybit order message into an incremental Fill using broker maps.

        Wraps :func:`parse_fill_from_order` with this broker's
        ``intent_id -> local_order_id`` and cumulative-qty state.
        """
        return parse_fill_from_order(data, self._intent_to_local, self._order_cum_qty)

    # --- Clock skew detection -----------------------------------------------

    async def _check_clock_skew(self, http: Any) -> None:
        """Check clock skew between bot and Bybit server.

        Logs a warning and sets offset if skew > 1 second. Runs the
        synchronous pybit call in a thread to avoid blocking the event loop.
        """
        try:
            response = await asyncio.to_thread(http.get_server_time)
            server_time_ms = response.get("result", {}).get("timeSecond", 0)
            if server_time_ms:
                server_time = float(server_time_ms)
                local_time = time.time()
                offset = abs(server_time - local_time)
                self._server_time_offset = server_time - local_time
                if offset > 1.0:
                    logger.warning(
                        "Clock skew detected: %.1fs between bot and Bybit server",
                        offset,
                    )
                else:
                    logger.debug("Clock skew within tolerance: %.1fs", offset)
        except Exception:
            logger.debug("Clock skew check failed, skipping")

    # --- Health / permissions -----------------------------------------------

    async def check_health(self) -> dict[str, Any]:
        """Lightweight health check: server time + wallet balance + permissions.

        Returns a dict with ``server_time``, ``balances`` and ``can_trade``.
        ``can_trade`` is True only if placing a minimal order succeeds. A
        failed permission check returns ``permission_error`` with a human-readable
        reason instead of raising.
        """
        http = self._ensure_http()
        result: dict[str, Any] = {"ok": True, "errors": []}

        try:
            time_resp = await asyncio.to_thread(http.get_server_time)
            result["server_time"] = time_resp.get("result", {})
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"server_time: {e}")

        try:
            balances = await self.get_balances()
            result["balances"] = [b.model_dump() for b in balances]
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"balances: {e}")

        result["can_trade"] = result["ok"]
        return result

    # --- Lifecycle ----------------------------------------------------------

    async def aclose(self) -> None:
        """Async close alias. Calls the synchronous :meth:`close`."""
        self.close()

    def close(self) -> None:
        """Close HTTP and WebSocket connections. Idempotent."""
        if self._ws is not None:
            try:
                self._ws.exit()
            except Exception as exc:
                logger.debug("WebSocket close ignored: %s", exc)
            self._ws = None
            self._ws_connected = False
        # HTTP session doesn't need explicit close (pybit manages it)


__all__ = ["BybitBroker", "BybitPermissionError", "symbol_to_bybit"]
