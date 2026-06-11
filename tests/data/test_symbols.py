"""Tests for the typed symbol model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kairon.data.symbols import (
    AssetClass,
    CryptoVenue,
    StockVenue,
    Symbol,
    crypto_perp,
    crypto_spot,
    etf,
    index_,
    stock,
)


def test_crypto_spot_factory() -> None:
    s = crypto_spot("btc", "usdt", CryptoVenue.BINANCE)
    assert s.canonical == "BTC-USDT"
    assert s.base == "BTC"
    assert s.quote == "USDT"
    assert s.asset_class == AssetClass.CRYPTO_SPOT
    assert s.venue == CryptoVenue.BINANCE
    assert s.is_crypto is True
    assert s.is_perp is False
    assert s.display == "BTC/USDT"


def test_crypto_perp_factory() -> None:
    s = crypto_perp("eth", "usdt", CryptoVenue.BYBIT)
    assert s.canonical == "ETH-USDT-PERP"
    assert s.is_crypto is True
    assert s.is_perp is True


def test_stock_factory() -> None:
    s = stock("aapl", StockVenue.POLYGON)
    assert s.canonical == "AAPL"
    assert s.asset_class == AssetClass.STOCK
    assert s.is_crypto is False
    assert s.display == "AAPL"


def test_etf_factory() -> None:
    s = etf("SPY", StockVenue.TIINGO)
    assert s.canonical == "SPY"
    assert s.asset_class == AssetClass.ETF


def test_index_factory() -> None:
    s = index_("VIX")
    assert s.canonical == "VIX"
    assert s.asset_class == AssetClass.INDEX


def test_symbol_is_frozen() -> None:
    s = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    with pytest.raises(ValidationError):
        s.canonical = "ETH-USDT"  # type: ignore[misc]


def test_symbol_rejects_lowercase_or_invalid() -> None:
    # Lowercase is upper-cased to "BTCUSDT", which is a valid TICKER form.
    # We require explicit hyphenation for crypto. Use a non-crypto stock
    # form to verify strict ticker validation.
    with pytest.raises(ValidationError):
        Symbol(canonical="BTC USDT", asset_class=AssetClass.CRYPTO_SPOT)
    with pytest.raises(ValidationError):
        Symbol(canonical="!@#", asset_class=AssetClass.CRYPTO_SPOT)
    with pytest.raises(ValidationError):
        Symbol(canonical="A" * 50, asset_class=AssetClass.STOCK)


def test_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        Symbol(
            canonical="BTC-USDT",
            asset_class=AssetClass.CRYPTO_SPOT,
            bogus="field",  # type: ignore[call-arg]
        )


def test_str_returns_canonical() -> None:
    s = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    assert str(s) == "BTC-USDT"


def test_equality() -> None:
    s1 = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    s2 = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    assert s1 == s2
    s3 = crypto_spot("ETH", "USDT", CryptoVenue.BINANCE)
    assert s1 != s3
