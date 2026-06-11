"""Typed canonical symbols, venues, and asset-class enums.

We adopt a single internal symbol convention so that adapters, features, and
the backtest engine all speak the same language:

- Crypto spot:    ``BASE-QUOTE`` (e.g., ``BTC-USDT``)
- Crypto perp:    ``BASE-QUOTE-PERP`` (e.g., ``BTC-USDT-PERP``)
- Stocks:         ``TICKER`` (e.g., ``AAPL``)
- ETF:            ``TICKER`` (e.g., ``SPY``)
- Indices:        ``INDEX-NAME`` (e.g., ``SP500``, ``VIX``)
- FX:             ``BASE-QUOTE`` (e.g., ``EUR-USD``)
- Commodities:    ``CODE`` (e.g., ``GC=F``)

This module is the single source of truth; adapters must convert to/from
the canonical form. The typed schema enforces the rules so that
``BTCUSDT`` and ``BTC/USDT`` and ``BTC-USDT`` cannot silently mean
different things inside the system.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetClass(str, Enum):
    """Top-level asset class for a symbol."""

    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    FX = "fx"
    COMMODITY = "commodity"
    BOND = "bond"


class CryptoVenue(str, Enum):
    """Crypto exchanges supported by CCXT in v1."""

    BINANCE = "binance"
    BYBIT = "bybit"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    OKX = "okx"


class StockVenue(str, Enum):
    """Stock data providers in v1."""

    POLYGON = "polygon"
    TIINGO = "tiingo"
    ALPHA_VANTAGE = "alpha_vantage"
    YFINANCE = "yfinance"
    TWRR = "twrr"


# ---------------------------------------------------------------------------
# Symbol model
# ---------------------------------------------------------------------------
_BASE_QUOTE_RE: Final[str] = r"^[A-Z0-9]{1,16}-[A-Z0-9]{1,16}$"
_TICKER_RE: Final[str] = r"^[A-Z0-9.\-]{1,16}$"
_INDEX_RE: Final[str] = r"^[A-Z0-9]{1,16}$"


class Symbol(BaseModel):
    """A canonical instrument identifier.

    The same string ``BTC-USDT`` means the same thing across all data
    sources, all features, and all backtests. Adapters translate their
    native representation into this canonical form on ingress and out
    on egress.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    canonical: str = Field(
        min_length=3,
        max_length=40,
        description="Canonical ticker, e.g. BTC-USDT, AAPL, SP500, GC=F",
    )
    asset_class: AssetClass
    venue: CryptoVenue | StockVenue | str | None = Field(
        default=None, description="Data venue (exchange or provider)"
    )
    base: str | None = Field(default=None, description="Base currency, if applicable")
    quote: str | None = Field(default=None, description="Quote currency, if applicable")

    @field_validator("canonical")
    @classmethod
    def _validate_canonical(cls, v: str) -> str:
        upper = v.upper()
        if not (
            re.match(_BASE_QUOTE_RE, upper)
            or re.match(_TICKER_RE, upper)
            or re.match(_INDEX_RE, upper)
        ):
            raise ValueError(
                f"invalid canonical symbol {v!r}; expected BASE-QUOTE, TICKER, or INDEX"
            )
        return upper

    @property
    def is_crypto(self) -> bool:
        return self.asset_class in (AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_PERP)

    @property
    def is_perp(self) -> bool:
        return self.asset_class == AssetClass.CRYPTO_PERP

    @property
    def display(self) -> str:
        """A human-readable label, e.g. ``BTC/USDT`` or ``AAPL``."""
        if self.base and self.quote:
            sep = "/" if self.is_crypto else ""
            return f"{self.base}{sep}{self.quote}"
        return self.canonical

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.canonical


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def crypto_spot(base: str, quote: str, venue: CryptoVenue) -> Symbol:
    """Build a spot crypto symbol, e.g. ``crypto_spot("BTC", "USDT", BINANCE)``."""
    return Symbol(
        canonical=f"{base.upper()}-{quote.upper()}",
        asset_class=AssetClass.CRYPTO_SPOT,
        venue=venue,
        base=base.upper(),
        quote=quote.upper(),
    )


def crypto_perp(base: str, quote: str, venue: CryptoVenue) -> Symbol:
    """Build a perpetual crypto symbol, e.g. ``crypto_perp("BTC", "USDT", BYBIT)``."""
    return Symbol(
        canonical=f"{base.upper()}-{quote.upper()}-PERP",
        asset_class=AssetClass.CRYPTO_PERP,
        venue=venue,
        base=base.upper(),
        quote=quote.upper(),
    )


def stock(ticker: str, venue: StockVenue) -> Symbol:
    """Build a stock/ETF symbol, e.g. ``stock("AAPL", POLYGON)``."""
    return Symbol(canonical=ticker.upper(), asset_class=AssetClass.STOCK, venue=venue)


def etf(ticker: str, venue: StockVenue) -> Symbol:
    """Build an ETF symbol, e.g. ``etf("SPY", POLYGON)``."""
    return Symbol(canonical=ticker.upper(), asset_class=AssetClass.ETF, venue=venue)


def index_(name: str, venue: StockVenue | None = None) -> Symbol:
    """Build an index symbol, e.g. ``index_("SP500")`` or ``index_("VIX")``."""
    return Symbol(canonical=name.upper(), asset_class=AssetClass.INDEX, venue=venue)
