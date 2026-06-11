"""Tests for :func:`kairon.live.feed.fetch_current_price` (US-004).

The function is the verification thread's seam to live ccxt. We mock
``_ccxt_client.make_client`` so no network touches happen in CI.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from kairon.live.feed import fetch_current_price


def _make_mock_client(ticker_price: float) -> Any:
    """Return a coroutine double that mimics the ccxt async client's API."""

    class _MockClient:
        async def fetch_ticker(self, market: str) -> dict[str, Any]:
            assert "/" in market, f"market should be base/quote, got {market!r}"
            return {"last": ticker_price, "symbol": market}

        async def close(self) -> None:
            return None

    return _MockClient()


def test_fetch_current_price_parses_btcusdt() -> None:
    with patch("kairon.data.adapters._ccxt_client.make_client") as mc:
        mc.return_value = _make_mock_client(68234.10)
        price = fetch_current_price("BTCUSDT", "binance")
    assert price == 68234.10
    # The market passed to fetch_ticker is base/quote.
    mc.assert_called_once_with("binance")


def test_fetch_current_price_handles_btcusd() -> None:
    with patch("kairon.data.adapters._ccxt_client.make_client") as mc:
        mc.return_value = _make_mock_client(68000.0)
        price = fetch_current_price("BTCUSD", "bybit")
    assert price == 68000.0
    mc.assert_called_once_with("bybit")


def test_fetch_current_price_lowercases_venue_to_string() -> None:
    with patch("kairon.data.adapters._ccxt_client.make_client") as mc:
        mc.return_value = _make_mock_client(1.0)
        fetch_current_price("ETHUSDT", "binance")
    mc.assert_called_once_with("binance")


def test_fetch_current_price_raises_on_missing_last() -> None:
    class _NoLastClient:
        async def fetch_ticker(self, market: str) -> dict[str, Any]:
            return {"symbol": market}  # no 'last'

        async def close(self) -> None:
            return None

    with patch("kairon.data.adapters._ccxt_client.make_client") as mc:
        mc.return_value = _NoLastClient()
        with pytest.raises(ValueError, match="no 'last' price"):
            fetch_current_price("BTCUSDT", "binance")


def test_fetch_current_price_closes_client_on_exit() -> None:
    """The mock client's close() should be called even if fetch_ticker raises."""
    closed = {"count": 0}

    class _Boom:
        async def fetch_ticker(self, market: str) -> dict[str, Any]:
            raise RuntimeError("boom")

        async def close(self) -> None:
            closed["count"] += 1

    with patch("kairon.data.adapters._ccxt_client.make_client") as mc:
        mc.return_value = _Boom()
        with pytest.raises(RuntimeError):
            fetch_current_price("BTCUSDT", "binance")
    assert closed["count"] == 1
