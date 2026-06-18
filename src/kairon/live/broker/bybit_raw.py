"""BybitRawBroker: direct Bybit v5 REST/WebSocket implementation using httpx/websockets.

This broker bypasses ``pybit`` and signs requests with Bybit v5 HMAC-SHA256.
It is intended as a transparent alternative to :class:`BybitBroker` for
regions/accounts where explicit request logging helps diagnose restrictions.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Any

import httpx
from websockets.asyncio.client import connect as ws_connect

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
    BybitAPIError,
    BybitPermissionError,
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
    symbol_str_to_bybit,
    symbol_to_bybit,
    tiny_positive_qty,
    utc_now_iso,
    uuid7,
)

logger = logging.getLogger(__name__)


class BybitRawBroker:
    """Live broker implementation using direct Bybit v5 HTTP and WebSocket.

    Uses ``httpx.AsyncClient`` for async REST requests and ``websockets`` for
    the private WebSocket stream. All signing is performed in-process with
    HMAC-SHA256 per the Bybit v5 specification.

    Args:
        api_key: Bybit API key.
        api_secret: Bybit API secret.
        testnet: If True, connect to Bybit testnet instead of mainnet.
        tld: Bybit top-level domain suffix (com, kz, hk, eu, nl). Defaults to
            "com"; use "kz" for testnet.bybit.kz accounts.
        recv_window: Request recv window in milliseconds.
        max_reconnect_attempts: Maximum consecutive WebSocket reconnection attempts.
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
        recv_window: int = 5000,
        max_reconnect_attempts: int = 10,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._tld = tld
        self._recv_window = recv_window
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay

        # HTTP client is created lazily
        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock = asyncio.Lock()

        # WebSocket state
        self._ws_task: asyncio.Task[Any] | None = None
        self._ws_connected: bool = False
        self._ws_shutdown: bool = False
        self._ws_connect_lock: asyncio.Lock = asyncio.Lock()
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue()
        self._position_queue: asyncio.Queue[Position] = asyncio.Queue()

        # Clock skew
        self._server_time_offset_ms: float = 0.0
        self._last_clock_sync: float | None = None

        # Rate limiting
        self._last_get_ts: float = 0.0
        self._last_post_ts: float = 0.0
        self._rate_limit_lock: asyncio.Lock = asyncio.Lock()

        # Intent / cumulative-qty tracking for incremental fill capture
        self._intent_to_local: dict[str, str] = {}
        self._order_cum_qty: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # HTTP transport
    # -----------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(30.0, connect=10.0),
                        follow_redirects=False,
                    )
        return self._client

    def _base_url(self) -> str:
        if self._testnet:
            return f"https://api-testnet.bybit.{self._tld}"
        return f"https://api.bybit.{self._tld}"

    def _sign(self, timestamp: str, payload: str) -> str:
        param_str = f"{timestamp}{self._api_key}{self._recv_window}{payload}"
        return hmac.new(
            self._api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _utc_now_ms(self) -> int:
        return int(time.time() * 1000 + self._server_time_offset_ms)

    async def _sync_clock(self) -> None:
        """Sync local clock against Bybit server time."""
        response = await self._request(
            "GET", "/v5/market/time", signed=False
        )
        result = response.get("result", {})
        time_second = result.get("timeSecond")
        time_nano = result.get("timeNano")

        if time_nano:
            server_ms = int(time_nano) / 1_000_000
        elif time_second:
            server_ms = float(time_second) * 1000
        else:
            return

        local_ms = time.time() * 1000
        self._server_time_offset_ms = server_ms - local_ms
        self._last_clock_sync = time.monotonic()
        logger.debug(
            "Clock sync: server_ms=%.3f local_ms=%.3f offset_ms=%.3f",
            server_ms,
            local_ms,
            self._server_time_offset_ms,
        )

    async def _enforce_rate_limit(self, method: str) -> None:
        async with self._rate_limit_lock:
            now = time.monotonic()
            if method == "GET":
                elapsed = now - self._last_get_ts
                if elapsed < 0.1:
                    await asyncio.sleep(0.1 - elapsed)
                self._last_get_ts = time.monotonic()
            else:
                elapsed = now - self._last_post_ts
                if elapsed < 0.3:
                    await asyncio.sleep(0.3 - elapsed)
                self._last_post_ts = time.monotonic()

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        signed: bool = True,
    ) -> dict[str, Any]:
        """Send a raw Bybit v5 request and return the JSON response."""
        if signed:
            if self._last_clock_sync is None or (
                time.monotonic() - self._last_clock_sync > 300
            ):
                await self._sync_clock()

        await self._enforce_rate_limit(method)

        url = self._base_url() + endpoint
        timestamp = str(self._utc_now_ms())
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if method == "GET":
            query_params = {
                k: v for k, v in sorted((params or {}).items()) if v is not None
            }
            query_string = urllib.parse.urlencode(
                query_params, safe=",", quote_via=urllib.parse.quote
            )
            signed_payload = query_string
            if query_string:
                url = f"{url}?{query_string}"
            body = None
        else:
            body_payload = payload or {}
            try:
                body_bytes: bytes = json.dumps(body_payload, separators=(",", ":")).encode(
                    "utf-8"
                )
            except TypeError:
                body_bytes = json.dumps(body_payload).encode("utf-8")
            signed_payload = body_bytes.decode("utf-8")
            body = body_bytes

        if signed:
            headers.update({
                "X-BAPI-API-KEY": self._api_key,
                "X-BAPI-SIGN": self._sign(timestamp, signed_payload),
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": str(self._recv_window),
            })

        logger.debug(
            "BybitRawBroker %s %s ts=%s payload=%s",
            method,
            endpoint,
            timestamp if signed else "unsigned",
            signed_payload if signed else "unsigned",
        )

        client = await self._ensure_client()
        response = await client.request(method, url, headers=headers, content=body)

        try:
            data = response.json()
        except Exception as exc:
            raise BybitAPIError(
                f"Bybit returned non-JSON HTTP {response.status_code}: {response.text}"
            ) from exc

        ret_code = data.get("retCode")
        ret_msg = data.get("retMsg", "")

        logger.debug("BybitRawBroker %s %s retCode=%s", method, endpoint, ret_code)

        if ret_code != 0:
            if ret_code == 10002:
                await self._sync_clock()
                # Retry once with fresh timestamp
                return await self._request(
                    method,
                    endpoint,
                    payload=payload,
                    params=params,
                    signed=signed,
                )
            if is_permission_error(ret_code, ret_msg):
                raise BybitPermissionError(permission_error_message(), ret_code=ret_code)
            raise BybitAPIError(f"Bybit API error {ret_code}: {ret_msg}", ret_code=ret_code)

        return data

    # -----------------------------------------------------------------------
    # WebSocket transport
    # -----------------------------------------------------------------------

    async def connect_websocket(self) -> None:
        """Connect to Bybit private WebSocket and start background receive task."""
        async with self._ws_connect_lock:
            if self._ws_task is not None and not self._ws_task.done():
                return

            self._ws_shutdown = False
            self._ws_task = asyncio.create_task(
                self._ws_loop(),
                name="bybit-raw-ws",
            )

    async def _ws_loop(self) -> None:
        """WebSocket receive/reconnect loop."""
        attempt = 0
        while not self._ws_shutdown and attempt < self._max_reconnect_attempts:
            try:
                await self._ws_connect_once()
                # Successful run; reset attempt counter for future disconnects
                attempt = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                attempt += 1
                delay = min(
                    self._reconnect_base_delay * (2 ** (attempt - 1)),
                    self._reconnect_max_delay,
                )
                logger.warning(
                    "BybitRawBroker WebSocket error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self._max_reconnect_attempts,
                    e,
                    delay,
                )
                self._ws_connected = False
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break

        if attempt >= self._max_reconnect_attempts:
            logger.error(
                "BybitRawBroker WebSocket failed after %d attempts",
                self._max_reconnect_attempts,
            )
        self._ws_connected = False

    async def _ws_connect_once(self) -> None:
        if self._last_clock_sync is None or (
            time.monotonic() - self._last_clock_sync > 300
        ):
            await self._sync_clock()

        scheme = "stream-testnet" if self._testnet else "stream"
        uri = f"wss://{scheme}.bybit.{self._tld}/v5/private"

        async with ws_connect(uri) as websocket:
            # Auth
            expires = self._utc_now_ms() + 5000
            auth_val = f"GET/realtime{expires}"
            signature = hmac.new(
                self._api_secret.encode("utf-8"),
                auth_val.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            await websocket.send(
                json.dumps({"op": "auth", "args": [self._api_key, expires, signature]})
            )

            auth_confirmed = False
            while not auth_confirmed:
                raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                msg = json.loads(raw)
                op = msg.get("op", "")
                if op == "auth":
                    if msg.get("success"):
                        auth_confirmed = True
                    else:
                        raise BybitAPIError(
                            f"WebSocket auth failed: {msg.get('ret_msg', msg)}"
                        )
                elif op == "ping":
                    await websocket.send(json.dumps({"op": "pong"}))

            # Subscribe
            await websocket.send(
                json.dumps({"op": "subscribe", "args": ["order", "position"]})
            )
            sub_confirmed = False
            while not sub_confirmed:
                raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                msg = json.loads(raw)
                op = msg.get("op", "")
                if op == "subscribe":
                    if msg.get("success"):
                        sub_confirmed = True
                    else:
                        raise BybitAPIError(
                            f"WebSocket subscribe failed: {msg.get('ret_msg', msg)}"
                        )
                elif op == "ping":
                    await websocket.send(json.dumps({"op": "pong"}))

            self._ws_connected = True
            logger.info("BybitRawBroker WebSocket connected")

            # Receive loop
            while not self._ws_shutdown:
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                except TimeoutError:
                    # Send a ping to keep connection alive
                    try:
                        await websocket.send(json.dumps({"op": "ping"}))
                    except Exception:
                        break
                    continue

                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                op = msg.get("op", "")
                if op == "ping":
                    await websocket.send(json.dumps({"op": "pong"}))
                    continue
                if op:
                    # Ignore operation confirmations / heartbeats
                    continue

                topic = msg.get("topic", "")
                data_list = msg.get("data", [])
                if not isinstance(data_list, list):
                    data_list = [data_list]

                if topic == "order":
                    for item in data_list:
                        fill = parse_fill_from_order(
                            item, self._intent_to_local, self._order_cum_qty
                        )
                        if fill is not None:
                            self._fill_queue.put_nowait(fill)
                elif topic == "position":
                    for item in data_list:
                        try:
                            position = parse_position(item)
                            if position is not None:
                                self._position_queue.put_nowait(position)
                        except Exception as e:
                            logger.error("Error parsing position message: %s", e)

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

    # -----------------------------------------------------------------------
    # Broker protocol implementation
    # -----------------------------------------------------------------------

    async def place_order(self, order: Order) -> Order:
        """Submit an order to Bybit. Returns the order with broker_id filled."""
        base, quote = order.symbol.replace("-PERP", "").split("-")
        bybit_symbol, category = symbol_to_bybit(
            crypto_perp(base, quote, CryptoVenue.BYBIT)
        )

        # Track the intent -> local order mapping before submission so that
        # WebSocket fills can resolve back to our local Order.id.
        self._intent_to_local[order.intent_id] = order.id

        payload: dict[str, Any] = {
            "category": category,
            "symbol": bybit_symbol,
            "side": order_side_to_bybit(order.side),
            "orderType": order_type_to_bybit(order.order_type),
            "qty": str(order.qty),
            "orderLinkId": order.intent_id,
        }
        if order.price is not None:
            payload["price"] = str(order.price)
        if order.sl is not None:
            payload["stopLoss"] = str(order.sl)
        if order.tp is not None:
            payload["takeProfit"] = str(order.tp)

        try:
            response = await self._request("POST", "/v5/order/create", payload=payload)
        except BybitPermissionError:
            raise
        except Exception as exc:
            logger.error("BybitRawBroker.place_order failed: %s", exc)
            return order.model_copy(
                update={
                    "status": OrderStatus.REJECTED,
                    "broker_id": f"bybit-{order.id[:8]}",
                    "ts": utc_now_iso(),
                }
            )

        result = response.get("result", {})
        broker_id = result.get("orderId", "")

        return order.model_copy(
            update={
                "status": OrderStatus.SUBMITTED,
                "broker_id": broker_id or f"bybit-{order.id[:8]}",
                "ts": utc_now_iso(),
            }
        )

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel a single order on Bybit."""
        bybit_symbol, category = symbol_str_to_bybit(symbol)
        payload = {
            "category": category,
            "symbol": bybit_symbol,
            "orderId": order_id,
        }
        await self._request("POST", "/v5/order/cancel", payload=payload)

        return Order(
            id=order_id,
            intent_id="",
            trace_id="",
            symbol=symbol,
            side=OrderSide.BUY,
            qty=tiny_positive_qty(),
            order_type=OrderType.MARKET,
            status=OrderStatus.CANCELLED,
            ts=utc_now_iso(),
        )

    async def cancel_all(self, symbol: str) -> list[Order]:
        """Cancel all open orders for a symbol on Bybit."""
        bybit_symbol, category = symbol_str_to_bybit(symbol)
        payload = {
            "category": category,
            "symbol": bybit_symbol,
        }
        response = await self._request("POST", "/v5/order/cancel-all", payload=payload)
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

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Fetch open positions from Bybit."""
        params: dict[str, Any] = {"category": "linear"}
        if symbol:
            bybit_symbol, _ = symbol_str_to_bybit(symbol)
            params["symbol"] = bybit_symbol

        try:
            response = await self._request("GET", "/v5/position/list", params=params)
            return parse_positions(response)
        except Exception as e:
            logger.error("BybitRawBroker.get_positions failed: %s", e)
            return []

    async def get_balances(self) -> list[Balance]:
        """Fetch account balances from Bybit."""
        try:
            response = await self._request(
                "GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"}
            )
            return parse_balances(response)
        except Exception as e:
            logger.error("BybitRawBroker.get_balances failed: %s", e)
            return [Balance(currency="USDT", available=0.0, total=0.0, ts=utc_now_iso())]

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Fetch open orders from Bybit."""
        params: dict[str, Any] = {"category": "linear"}
        if symbol:
            bybit_symbol, _ = symbol_str_to_bybit(symbol)
            params["symbol"] = bybit_symbol

        try:
            response = await self._request("GET", "/v5/order/realtime", params=params)
            return parse_open_orders(response)
        except Exception as e:
            logger.error("BybitRawBroker.get_open_orders failed: %s", e)
            return []

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol on Bybit. Idempotent."""
        bybit_symbol, category = symbol_str_to_bybit(symbol)
        payload = {
            "category": category,
            "symbol": bybit_symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        await self._request("POST", "/v5/position/set-leverage", payload=payload)
        logger.info("Set leverage for %s to %dx", symbol, leverage)

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
        bybit_symbol, category = symbol_str_to_bybit(symbol)
        intent_id = uuid7()
        payload: dict[str, Any] = {
            "category": category,
            "symbol": bybit_symbol,
            "side": order_side_to_bybit(side),
            "orderType": order_type_to_bybit(order_type),
            "qty": str(qty),
            "triggerPrice": str(trigger_price),
            "orderLinkId": intent_id,
        }
        if sl is not None:
            payload["stopLoss"] = str(sl)
        if tp is not None:
            payload["takeProfit"] = str(tp)

        response = await self._request("POST", "/v5/order/create", payload=payload)
        result = response.get("result", {})
        broker_id = result.get("orderId", "")

        return Order(
            id=broker_id or f"bybit-cond-{intent_id[:8]}",
            intent_id=intent_id,
            trace_id="",
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=trigger_price,
            sl=sl,
            tp=tp,
            status=OrderStatus.PENDING,
            broker_id=broker_id,
            ts=utc_now_iso(),
        )

    # -----------------------------------------------------------------------
    # Health / permissions
    # -----------------------------------------------------------------------

    async def check_health(self) -> dict[str, Any]:
        """Lightweight health check: server time + wallet balance + permissions."""
        result: dict[str, Any] = {"ok": True, "errors": []}

        try:
            time_resp = await self._request("GET", "/v5/market/time")
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

        try:
            await self._request(
                "POST",
                "/v5/order/create",
                payload={
                    "category": "linear",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Limit",
                    "qty": "0.001",
                    "price": "0.001",
                },
            )
            result["can_trade_linear"] = True
        except Exception as e:
            ret_code = getattr(e, "ret_code", None)
            msg = str(e)
            if is_permission_error(ret_code, msg):
                result["can_trade_linear"] = False
                result["ok"] = False
                result["permission_error"] = permission_error_message()
            elif is_insufficient_balance(ret_code, msg):
                result["can_trade_linear"] = True
            else:
                result["can_trade_linear"] = "unknown"
                result["errors"].append(f"permission_probe: {e}")

        return result

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def aclose(self) -> None:
        """Async close of HTTP and WebSocket connections. Idempotent."""
        self._ws_shutdown = True
        if self._ws_task is not None and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        self._ws_connected = False

        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:
                logger.debug("HTTP client close ignored: %s", exc)
            self._client = None

    def close(self) -> None:
        """Sync close alias. Best effort if the event loop is not running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Called from within an async context: schedule async cleanup.
            loop.create_task(self.aclose())
            return

        try:
            asyncio.run(self.aclose())
        except Exception as exc:
            logger.debug("Close ignored: %s", exc)


__all__ = ["BybitRawBroker"]
