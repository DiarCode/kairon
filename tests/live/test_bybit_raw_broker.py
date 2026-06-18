"""Tests for BybitRawBroker — signing, request construction, parsing, and integration."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairon.live.broker.base import (
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from kairon.live.broker.bybit_raw import BybitRawBroker
from kairon.live.broker.bybit_shared import (
    BybitAPIError,
    BybitPermissionError,
)


def _broker(
    api_key: str = "key",
    api_secret: str = "secret",  # noqa: S107
    **kwargs: Any,
) -> BybitRawBroker:
    """Factory for test brokers with a placeholder secret."""
    broker = BybitRawBroker(api_key=api_key, api_secret=api_secret, **kwargs)
    broker._server_time_offset_ms = 0.0
    broker._last_clock_sync = time.monotonic()
    return broker


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


class TestBybitRawHMAC:
    """Verify _sign matches the Bybit v5 documented algorithm."""

    def test_sign_matches_hand_rolled(self) -> None:
        api_key = "xAK9wdZlV5UQZGVNyM"
        secret = "MVUWIqkRtDdQp8BH5FfoXTER0ER3ReVhSZ6j"  # noqa: S105
        recv_window = 5000
        timestamp = "1781509123000"
        payload = (
            '{"category":"linear","symbol":"BTCUSDT",'
            '"side":"Buy","orderType":"Limit",'
            '"qty":"0.001","price":"1.0"}'
        )

        broker = BybitRawBroker(api_key=api_key, api_secret=secret)
        expected = hmac.new(
            secret.encode("utf-8"),
            f"{timestamp}{api_key}{recv_window}{payload}".encode(),
            hashlib.sha256,
        ).hexdigest()
        assert broker._sign(timestamp, payload) == expected


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestBybitRawURL:
    """Verify base URL selection for testnet/mainnet and TLDs."""

    def test_testnet_com(self) -> None:
        broker = _broker(testnet=True, tld="com")
        assert broker._base_url() == "https://api-testnet.bybit.com"

    def test_testnet_kz(self) -> None:
        broker = _broker(testnet=True, tld="kz")
        assert broker._base_url() == "https://api-testnet.bybit.kz"

    def test_mainnet_kz(self) -> None:
        broker = _broker(testnet=False, tld="kz")
        assert broker._base_url() == "https://api.bybit.kz"


# ---------------------------------------------------------------------------
# Request headers / encoding
# ---------------------------------------------------------------------------


class TestBybitRawRequestHeaders:
    """Verify signed requests carry correct Bybit v5 headers."""

    @pytest.mark.asyncio
    async def test_post_headers(self) -> None:
        broker = _broker()
        broker._server_time_offset_ms = 0.0
        broker._last_clock_sync = time.monotonic()
        calls: list[dict[str, Any]] = []

        async def fake_request(method, url, *, headers, content):
            calls.append({"method": method, "headers": headers, "content": content})
            resp = MagicMock()
            resp.json.return_value = {"retCode": 0, "retMsg": "OK"}
            resp.status_code = 200
            return resp

        broker._client = MagicMock()
        broker._client.request = fake_request

        await broker._request(
            "POST",
            "/v5/order/create",
            payload={"category": "linear", "symbol": "BTCUSDT", "side": "Buy"},
        )

        call = calls[0]
        assert call["method"] == "POST"
        headers = call["headers"]
        assert headers["X-BAPI-API-KEY"] == "key"
        assert headers["X-BAPI-SIGN-TYPE"] == "2"
        assert "X-BAPI-SIGN" in headers
        assert "X-BAPI-TIMESTAMP" in headers
        assert headers["X-BAPI-RECV-WINDOW"] == "5000"
        body = json.loads(call["content"].decode("utf-8"))
        assert body["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_query_string_uses_percent_encoding(self) -> None:
        broker = _broker()
        broker._server_time_offset_ms = 0.0
        broker._last_clock_sync = time.monotonic()
        calls: list[dict[str, Any]] = []

        async def fake_request(method, url, *, headers, content):
            calls.append({"url": url, "headers": headers})
            resp = MagicMock()
            resp.json.return_value = {"retCode": 0, "retMsg": "OK"}
            resp.status_code = 200
            return resp

        broker._client = MagicMock()
        broker._client.request = fake_request

        await broker._request(
            "GET",
            "/v5/order/realtime",
            params={"category": "linear", "symbol": "BTC USDT"},
        )

        call = calls[0]
        assert "%20" in call["url"]
        assert "+" not in call["url"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestBybitRawRateLimit:
    """Verify POST requests are paced."""

    @pytest.mark.asyncio
    async def test_two_post_calls_are_paced(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(json=lambda: {"retCode": 0}, status_code=200)
        )

        start = time.monotonic()
        await broker._request("POST", "/v5/order/create", payload={"a": "1"})
        await broker._request("POST", "/v5/order/create", payload={"a": "2"})
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25  # 300 ms target, allow some slack


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestBybitRawResponseParsing:
    """Verify domain object parsing from sample Bybit responses."""

    @pytest.mark.asyncio
    async def test_get_balances(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(
                json=lambda: {
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "coin": [
                                    {
                                        "coin": "BTC",
                                        "availableToWithdraw": "0.5",
                                        "walletBalance": "1.0",
                                    }
                                ]
                            }
                        ]
                    },
                },
                status_code=200,
            )
        )

        balances = await broker.get_balances()
        assert len(balances) == 1
        assert balances[0].currency == "BTC"
        assert balances[0].available == 0.5
        assert balances[0].total == 1.0

    @pytest.mark.asyncio
    async def test_get_positions(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(
                json=lambda: {
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "symbol": "BTCUSDT",
                                "side": "Buy",
                                "size": "0.1",
                                "entryPrice": "50000",
                                "unrealisedPnl": "100",
                            }
                        ]
                    },
                },
                status_code=200,
            )
        )

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC-USDT-PERP"
        assert positions[0].side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_get_open_orders(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(
                json=lambda: {
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "orderId": "abc",
                                "symbol": "BTCUSDT",
                                "side": "Sell",
                                "orderType": "Limit",
                                "qty": "0.01",
                                "price": "60000",
                                "orderStatus": "New",
                            }
                        ]
                    },
                },
                status_code=200,
            )
        )

        orders = await broker.get_open_orders()
        assert len(orders) == 1
        assert orders[0].id == "abc"
        assert orders[0].side == OrderSide.SELL
        assert orders[0].status == OrderStatus.SUBMITTED


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestBybitRawPermissionError:
    """Verify retCode mapping to typed exceptions."""

    @pytest.mark.asyncio
    async def test_10024_raises_bybit_permission_error(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(
                json=lambda: {
                    "retCode": 10024,
                    "retMsg": "regulatory restrictions",
                },
                status_code=200,
            )
        )

        with pytest.raises(BybitPermissionError):
            await broker._request("POST", "/v5/order/create", payload={})

    @pytest.mark.asyncio
    async def test_other_retcode_raises_bybit_api_error(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(
                json=lambda: {"retCode": 170131, "retMsg": "Insufficient balance"},
                status_code=200,
            )
        )

        with pytest.raises(BybitAPIError):
            await broker._request("POST", "/v5/order/create", payload={})


class TestBybitRawReadFallback:
    """Verify read methods return empty lists on failure."""

    @pytest.mark.asyncio
    async def test_get_balances_returns_empty_on_error(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(side_effect=RuntimeError("network down"))

        balances = await broker.get_balances()
        assert len(balances) == 1
        assert balances[0].currency == "USDT"
        assert balances[0].total == 0.0


class TestBybitRawHealth170131:
    """Verify health check maps 170131 to can_trade_linear=True."""

    @pytest.mark.asyncio
    async def test_insufficient_balance_maps_to_permission_ok(self) -> None:
        broker = _broker()

        responses = [
            {"retCode": 0, "result": {"timeSecond": "1700000000"}},
            {
                "retCode": 0,
                "result": {"list": [{"coin": [{"coin": "BTC", "walletBalance": "1"}]}]},
            },
        ]

        async def fake_request(method, url, **kwargs):
            if url.endswith("/v5/market/time"):
                return MagicMock(json=lambda: responses.pop(0), status_code=200)
            if url.endswith("/v5/account/wallet-balance"):
                return MagicMock(json=lambda: responses.pop(0), status_code=200)
            if url.endswith("/v5/order/create"):
                return MagicMock(
                    json=lambda: {"retCode": 170131, "retMsg": "Insufficient balance"},
                    status_code=200,
                )
            raise AssertionError(f"unexpected endpoint: {url}")

        broker._client = MagicMock()
        broker._client.request = fake_request

        health = await broker.check_health()
        assert health["can_trade_linear"] is True


class TestBybitRawHealth10024:
    """Verify health check maps 10024 to can_trade_linear=False."""

    @pytest.mark.asyncio
    async def test_permission_restriction_maps_to_can_trade_false(self) -> None:
        broker = _broker()

        responses = [
            {"retCode": 0, "result": {"timeSecond": "1700000000"}},
            {
                "retCode": 0,
                "result": {"list": [{"coin": [{"coin": "BTC", "walletBalance": "1"}]}]},
            },
        ]

        async def fake_request(method, url, **kwargs):
            if url.endswith("/v5/market/time"):
                return MagicMock(json=lambda: responses.pop(0), status_code=200)
            if url.endswith("/v5/account/wallet-balance"):
                return MagicMock(json=lambda: responses.pop(0), status_code=200)
            if url.endswith("/v5/order/create"):
                return MagicMock(
                    json=lambda: {"retCode": 10024, "retMsg": "regulatory restriction"},
                    status_code=200,
                )
            raise AssertionError(f"unexpected endpoint: {url}")

        broker._client = MagicMock()
        broker._client.request = fake_request

        health = await broker.check_health()
        assert health["can_trade_linear"] is False
        assert health["ok"] is False
        assert "permission_error" in health


# ---------------------------------------------------------------------------
# Clock skew
# ---------------------------------------------------------------------------


class TestBybitRawClockSkew:
    """Verify _utc_now_ms applies server offset."""

    @pytest.mark.asyncio
    async def test_offset_applied(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(
                json=lambda: {"retCode": 0, "result": {"timeSecond": "2000000000"}},
                status_code=200,
            )
        )

        await broker._sync_clock()
        # Server claims year 2033, local is 2026 -> offset ~7 years
        assert abs(broker._server_time_offset_ms) > 1000
        now = broker._utc_now_ms()
        assert now > int(time.time() * 1000) + 1_000_000


class TestBybitRawRequestRetry10002:
    """Verify retCode=10002 triggers a retry after clock sync."""

    @pytest.mark.asyncio
    async def test_retry_once_on_request_expired(self) -> None:
        broker = _broker()
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            side_effect=[
                MagicMock(
                    json=lambda: {"retCode": 10002, "retMsg": "request expired"},
                    status_code=200,
                ),
                MagicMock(
                    json=lambda: {"retCode": 0, "result": {"timeSecond": "1700000000"}},
                    status_code=200,
                ),
                MagicMock(json=lambda: {"retCode": 0, "retMsg": "OK"}, status_code=200),
            ]
        )

        response = await broker._request("GET", "/v5/market/time", signed=False)
        assert response["retCode"] == 0
        assert broker._client.request.call_count == 3


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestBybitRawProtocol:
    """Verify BybitRawBroker satisfies the Broker protocol."""

    def test_isinstance_broker(self) -> None:
        broker = _broker(api_key="k")
        assert isinstance(broker, Broker)

    @pytest.mark.asyncio
    async def test_cancel_returns_positive_qty(self) -> None:
        broker = _broker(api_key="k")
        broker._client = MagicMock()
        broker._client.request = AsyncMock(
            return_value=MagicMock(json=lambda: {"retCode": 0}, status_code=200)
        )

        order = await broker.cancel_order("BTC-USDT-PERP", "order-123")
        assert order.status == OrderStatus.CANCELLED
        assert order.qty > 0


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


class TestBybitRawWebSocketAuth:
    """Verify WebSocket auth message construction."""

    @pytest.mark.asyncio
    async def test_auth_signature(self) -> None:
        broker = _broker()
        broker._server_time_offset_ms = 0.0
        broker._last_clock_sync = time.monotonic()

        sent_messages = []
        received_messages: list[str | BaseException] = [
            json.dumps({"op": "auth", "success": True}),
            json.dumps({"op": "subscribe", "success": True}),
            asyncio.CancelledError(),
        ]

        async def fake_send(message: str | bytes) -> None:
            sent_messages.append(message.decode() if isinstance(message, bytes) else message)

        fake_ws = MagicMock()
        fake_ws.send = fake_send
        fake_ws.recv = AsyncMock(side_effect=received_messages)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=fake_ws)
        cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kairon.live.broker.bybit_raw.ws_connect", return_value=cm),
            contextlib.suppress(asyncio.CancelledError),
        ):
            await broker._ws_connect_once()

        auth_msg = json.loads(sent_messages[0])
        assert auth_msg["op"] == "auth"
        key, expires, signature = auth_msg["args"]
        assert key == "key"
        expected = hmac.new(
            b"secret",
            f"GET/realtime{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()
        assert signature == expected
        assert broker._ws_connected is True


class TestBybitRawWebSocketPingPong:
    """Verify ping/pong handling."""

    @pytest.mark.asyncio
    async def test_ping_replies_pong(self) -> None:
        broker = _broker()
        broker._server_time_offset_ms = 0.0
        broker._last_clock_sync = time.monotonic()

        sent_messages: list[str] = []
        received_messages: list[str | Exception] = [
            json.dumps({"op": "auth", "success": True, "conn_id": "c"}),
            json.dumps({"op": "subscribe", "success": True, "conn_id": "c"}),
            json.dumps({"op": "ping", "args": ["ping"]}),
        ]

        async def fake_send(message: str | bytes) -> None:
            sent_messages.append(message.decode() if isinstance(message, bytes) else message)

        async def fake_recv() -> str:
            msg = received_messages.pop(0)
            if isinstance(msg, Exception):
                raise msg
            try:
                data = json.loads(msg)
            except Exception:
                return msg
            if data.get("op") == "ping":
                broker._ws_shutdown = True
            return msg

        fake_ws = MagicMock()
        fake_ws.send = fake_send
        fake_ws.recv = fake_recv
        fake_ws.close = AsyncMock()

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=fake_ws)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("kairon.live.broker.bybit_raw.ws_connect", return_value=cm):
            await broker._ws_connect_once()

        pong = next(
            (json.loads(m) for m in sent_messages if json.loads(m).get("op") == "pong"),
            None,
        )
        assert pong == {"op": "pong"}


# ---------------------------------------------------------------------------
# Integration tests (gated by env var)
# ---------------------------------------------------------------------------

bybit_api_key = os.environ.get("KAIRON_BYBIT_API_KEY", "")
bybit_api_secret = os.environ.get("KAIRON_BYBIT_API_SECRET", "")
bybit_tld = os.environ.get("KAIRON_BYBIT_TLD", "com")

if not bybit_api_key or not bybit_api_secret:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"')
            if key == "KAIRON_BYBIT_API_KEY":
                bybit_api_key = value
            elif key == "KAIRON_BYBIT_API_SECRET":
                bybit_api_secret = value
            elif key == "KAIRON_BYBIT_TLD":
                bybit_tld = value


@pytest.mark.skipif(
    not bybit_api_key or not bybit_api_secret,
    reason="KAIRON_BYBIT_API_KEY/SECRET not set",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestBybitRawBrokerIntegration:
    """Integration tests against Bybit testnet."""

    async def test_get_wallet_balance(self) -> None:
        broker = BybitRawBroker(
            api_key=bybit_api_key,
            api_secret=bybit_api_secret,
            testnet=True,
            tld=bybit_tld,
        )
        balances = await broker.get_balances()
        assert len(balances) >= 1
        assert balances[0].total >= 0
        await broker.aclose()

    async def test_get_positions_empty(self) -> None:
        broker = BybitRawBroker(
            api_key=bybit_api_key,
            api_secret=bybit_api_secret,
            testnet=True,
            tld=bybit_tld,
        )
        positions = await broker.get_positions()
        assert isinstance(positions, list)
        await broker.aclose()

    async def test_symbol_round_trip_on_testnet(self) -> None:
        broker = BybitRawBroker(
            api_key=bybit_api_key,
            api_secret=bybit_api_secret,
            testnet=True,
            tld=bybit_tld,
        )
        order = Order(
            id="test-001",
            intent_id="intent-001",
            trace_id="trace-001",
            symbol="BTC-USDT-PERP",
            side=OrderSide.BUY,
            qty=0.001,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            ts="2026-06-13T12:00:00+00:00",
        )
        try:
            filled = await broker.place_order(order)
        except BybitPermissionError as e:
            pytest.skip(f"Testnet account restricted from linear perps: {e}")

        assert filled.broker_id is not None
        assert filled.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED)
        await broker.aclose()
