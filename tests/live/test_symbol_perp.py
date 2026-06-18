"""Tests for crypto_perp symbol factory — verifying perp support."""

from __future__ import annotations

from kairon.data.symbols import CryptoVenue, crypto_perp


class TestCryptoPerpSymbol:
    """Verify that the crypto_perp factory produces valid perpetual symbols."""

    def test_btc_usdt_perp_bybit(self) -> None:
        symbol = crypto_perp("BTC", "USDT", CryptoVenue.BYBIT)
        assert symbol.is_perp is True
        assert symbol.canonical == "BTC-USDT-PERP"
        assert symbol.asset_class.value == "crypto_perp"

    def test_eth_usdt_perp_bybit(self) -> None:
        symbol = crypto_perp("ETH", "USDT", CryptoVenue.BYBIT)
        assert symbol.is_perp is True
        assert symbol.canonical == "ETH-USDT-PERP"

    def test_sol_usdt_perp_bybit(self) -> None:
        symbol = crypto_perp("SOL", "USDT", CryptoVenue.BYBIT)
        assert symbol.is_perp is True
        assert symbol.canonical == "SOL-USDT-PERP"