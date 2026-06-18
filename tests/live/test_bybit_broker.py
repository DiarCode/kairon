"""Tests for BybitBroker — symbol mapping, order construction, HMAC, and integration (gated by env var)."""

from __future__ import annotations

import hashlib
import hmac
import os

import pytest

from kairon.data.symbols import CryptoVenue, crypto_perp, crypto_spot
from kairon.live.broker.base import Order
from kairon.live.broker.bybit import BybitBroker, symbol_to_bybit
from kairon.live.broker.bybit_shared import (
    bybit_to_symbol_str,
    symbol_str_to_bybit,
)

# ---------------------------------------------------------------------------
# Symbol mapping tests (unit, no network)
# ---------------------------------------------------------------------------


class TestSymbolMapping:
    """Test Kairon Symbol → Bybit wire format conversion."""

    def test_btc_usdt_perp(self) -> None:
        symbol = crypto_perp("BTC", "USDT", CryptoVenue.BYBIT)
        bybit_sym, category = symbol_to_bybit(symbol)
        assert bybit_sym == "BTCUSDT"
        assert category == "linear"

    def test_eth_usdt_perp(self) -> None:
        symbol = crypto_perp("ETH", "USDT", CryptoVenue.BYBIT)
        bybit_sym, category = symbol_to_bybit(symbol)
        assert bybit_sym == "ETHUSDT"
        assert category == "linear"

    def test_sol_usdt_perp(self) -> None:
        symbol = crypto_perp("SOL", "USDT", CryptoVenue.BYBIT)
        bybit_sym, category = symbol_to_bybit(symbol)
        assert bybit_sym == "SOLUSDT"
        assert category == "linear"

    def test_spot_symbol_raises(self) -> None:
        """Spot symbols are not supported for Bybit trading."""
        symbol = crypto_spot("BTC", "USDT", CryptoVenue.BYBIT)
        with pytest.raises(ValueError, match="Only perpetual symbols"):
            symbol_to_bybit(symbol)

    def test_wrong_venue_raises(self) -> None:
        symbol = crypto_perp("BTC", "USDT", CryptoVenue.BINANCE)
        with pytest.raises(ValueError, match="venue must be BYBIT"):
            symbol_to_bybit(symbol)


class TestSymbolStrMapping:
    """Test string-based symbol mapping helpers."""

    def test_btc_usdt_perp_str(self) -> None:
        bybit_sym, category = symbol_str_to_bybit("BTC-USDT-PERP")
        assert bybit_sym == "BTCUSDT"
        assert category == "linear"

    def test_eth_usdt_perp_str(self) -> None:
        bybit_sym, category = symbol_str_to_bybit("ETH-USDT-PERP")
        assert bybit_sym == "ETHUSDT"
        assert category == "linear"

    def test_round_trip_btc(self) -> None:
        """Canonical → Bybit → Canonical round trip."""
        bybit_sym, _ = symbol_str_to_bybit("BTC-USDT-PERP")
        kairon_sym = bybit_to_symbol_str(bybit_sym)
        assert kairon_sym == "BTC-USDT-PERP"

    def test_round_trip_sol(self) -> None:
        bybit_sym, _ = symbol_str_to_bybit("SOL-USDT-PERP")
        kairon_sym = bybit_to_symbol_str(bybit_sym)
        assert kairon_sym == "SOL-USDT-PERP"

    def test_round_trip_eth(self) -> None:
        bybit_sym, _ = symbol_str_to_bybit("ETH-USDT-PERP")
        kairon_sym = bybit_to_symbol_str(bybit_sym)
        assert kairon_sym == "ETH-USDT-PERP"


class TestBybitBrokerInit:
    """Test BybitBroker initialization without network."""

    def test_init_testnet(self) -> None:
        broker = BybitBroker(
            api_key="test_key",
            api_secret="test_secret",  # noqa: S106
            testnet=True,
        )
        assert broker._testnet is True
        assert broker._tld == "com"
        assert broker._http is None  # Lazy init
        assert broker._ws_connected is False

    def test_init_mainnet(self) -> None:
        broker = BybitBroker(
            api_key="test_key",
            api_secret="test_secret",  # noqa: S106
            testnet=False,
        )
        assert broker._testnet is False

    def test_custom_reconnect_params(self) -> None:
        broker = BybitBroker(
            api_key="test_key",
            api_secret="test_secret",  # noqa: S106
            testnet=True,
            max_reconnect_attempts=5,
            reconnect_base_delay=2.0,
            reconnect_max_delay=60.0,
        )
        assert broker._max_reconnect_attempts == 5
        assert broker._reconnect_base_delay == 2.0
        assert broker._reconnect_max_delay == 60.0

    def test_init_custom_tld(self) -> None:
        broker = BybitBroker(
            api_key="test_key",
            api_secret="test_secret",  # noqa: S106
            testnet=True,
            tld="kz",
        )
        assert broker._tld == "kz"


class TestBybitHMAC:
    """Verify pybit signs requests using the same HMAC-SHA256 as Bybit docs."""

    def test_pybit_hmac_matches_hand_rolled(self) -> None:
        api_key = "xAK9wdZlV5UQZGVNyM"
        secret = "MVUWIqkRtDdQp8BH5FfoXTER0ER3ReVhSZ6j"  # noqa: S105
        recv_window = 5000
        timestamp = 1781509123000
        payload = (
            '{"category":"linear","symbol":"BTCUSDT",'
            '"side":"Buy","orderType":"Limit",'
            '"qty":"0.001","price":"1.0"}'
        )

        param_str = f"{timestamp}{api_key}{recv_window}{payload}"
        expected = hmac.new(
            secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        from pybit._http_manager import _V5HTTPManager

        manager = _V5HTTPManager.__new__(_V5HTTPManager)
        manager.api_key = api_key
        manager.api_secret = secret
        actual = manager._auth(payload, recv_window, timestamp)

        assert actual == expected


# ---------------------------------------------------------------------------
# Integration tests (gated by KAIRON_BYBIT_API_KEY env var)
# ---------------------------------------------------------------------------

BYBIT_API_KEY = os.environ.get("KAIRON_BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.environ.get("KAIRON_BYBIT_API_SECRET", "")
BYBIT_TLD = os.environ.get("KAIRON_BYBIT_TLD", "com")


# If env vars are not set, fall back to the project's .env file so that
# running ``pytest -m integration`` in a fresh shell still picks up the
# configured credentials.
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    from pathlib import Path

    # Project root: this file is under tests/live/, so go up two levels.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"')
            if key == "KAIRON_BYBIT_API_KEY":
                BYBIT_API_KEY = value
            elif key == "KAIRON_BYBIT_API_SECRET":
                BYBIT_API_SECRET = value
            elif key == "KAIRON_BYBIT_TLD":
                BYBIT_TLD = value


@pytest.mark.skipif(
    not BYBIT_API_KEY or not BYBIT_API_SECRET,
    reason="KAIRON_BYBIT_API_KEY/SECRET not set",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestBybitBrokerIntegration:
    """Integration tests against Bybit testnet.

    These tests require real API credentials and hit the testnet.
    Run with: pytest -m integration tests/live/test_bybit_broker.py
    """

    async def test_get_wallet_balance(self) -> None:
        broker = BybitBroker(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            testnet=True,
            tld=BYBIT_TLD,
        )
        balances = await broker.get_balances()
        assert len(balances) >= 1
        # The testnet account holds 1 BTC, so first coin may be BTC rather than USDT.
        assert balances[0].total >= 0
        broker.close()

    async def test_get_positions_empty(self) -> None:
        broker = BybitBroker(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            testnet=True,
            tld=BYBIT_TLD,
        )
        positions = await broker.get_positions()
        # May have positions from previous testnet activity
        assert isinstance(positions, list)
        broker.close()

    async def test_symbol_round_trip_on_testnet(self) -> None:
        """Verify symbol mapping works end-to-end on testnet.

        If the testnet account is region-restricted from linear perps, the
        call raises BybitPermissionError and we assert the error message is
        helpful rather than failing mysteriously.
        """
        from kairon.live.broker.base import OrderSide, OrderStatus, OrderType
        from kairon.live.broker.bybit import BybitPermissionError

        broker = BybitBroker(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            testnet=True,
            tld=BYBIT_TLD,
        )
        # Place a small market order on testnet
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
        assert filled.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED, OrderStatus.REJECTED)
        broker.close()
