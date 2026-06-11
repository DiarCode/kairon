"""Tests for the CCXT adapter — fully mocked, no real exchange call."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pyarrow as pa
import pytest

from kairon.data.adapters import AdapterError
from kairon.data.adapters.ccxt_adapter import CCXTAdapter
from kairon.data.io import OHLCV_SCHEMA
from kairon.data.symbols import (
    CryptoVenue,
    StockVenue,
    crypto_perp,
    crypto_spot,
    stock,
)


def _ccxt_rows() -> list[list[Any]]:
    base = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    return [
        [base, 100.0, 101.0, 99.0, 100.5, 10.0],
        [base + 5 * 60_000, 100.5, 101.5, 100.0, 101.0, 20.0],
    ]


@pytest.mark.asyncio
async def test_afetch_returns_ohlcv_schema() -> None:
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE)
    client = MagicMock()
    client.close = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=_ccxt_rows())
    with patch.object(adapter, "_get_client", return_value=client):
        sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
        t = await adapter.afetch(
            sym,
            "5m",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 1, tzinfo=UTC),
        )
    assert t.schema == OHLCV_SCHEMA
    assert t.num_rows == 2
    await adapter.aclose()


@pytest.mark.asyncio
async def test_afetch_rejects_stock_symbol() -> None:
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE)
    sym = stock("AAPL", StockVenue.POLYGON)
    with pytest.raises(AdapterError, match="crypto symbol"):
        await adapter.afetch(
            sym,
            "5m",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 1, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_afetch_rejects_perp_on_coinbase() -> None:
    adapter = CCXTAdapter(venue=CryptoVenue.COINBASE)
    sym = crypto_perp("BTC", "USDT", CryptoVenue.COINBASE)
    with pytest.raises(AdapterError, match="does not support perps"):
        await adapter.afetch(
            sym,
            "5m",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 1, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_afetch_retries_on_transient_error() -> None:
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE, max_retries=2)
    client = MagicMock()
    client.close = AsyncMock()
    client.fetch_ohlcv = AsyncMock(
        side_effect=[Exception("net blip"), _ccxt_rows()]
    )
    with patch.object(adapter, "_get_client", return_value=client):
        sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
        t = await adapter.afetch(
            sym,
            "5m",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 1, tzinfo=UTC),
        )
    assert t.num_rows == 2
    assert client.fetch_ohlcv.await_count == 2
    await adapter.aclose()


@pytest.mark.asyncio
async def test_afetch_raises_after_max_retries() -> None:
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE, max_retries=1)
    client = MagicMock()
    client.close = AsyncMock()
    client.fetch_ohlcv = AsyncMock(side_effect=Exception("boom"))
    with patch.object(adapter, "_get_client", return_value=client):
        sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
        with pytest.raises(AdapterError, match="after .* retries"):
            await adapter.afetch(
                sym,
                "5m",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 1, tzinfo=UTC),
            )
    await adapter.aclose()


@pytest.mark.asyncio
async def test_afetch_rejects_unsupported_timeframe() -> None:
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE)
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    with pytest.raises(AdapterError, match="unsupported timeframe"):
        await adapter.afetch(
            sym,
            "3m",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 1, tzinfo=UTC),
        )


def test_fetch_sync_wrapper() -> None:
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE)
    client = MagicMock()
    client.close = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=_ccxt_rows())
    with patch.object(adapter, "_get_client", return_value=client):
        sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
        t = adapter.fetch(
            sym,
            "5m",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 1, tzinfo=UTC),
        )
    assert isinstance(t, pa.Table)
    assert t.schema == OHLCV_SCHEMA


def test_adapter_name_is_ccxt() -> None:
    assert CCXTAdapter().name == "ccxt"
