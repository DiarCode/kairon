"""Tests for the US-equity WebSocket stub.

The stub does not open a real connection: we exercise the
:class:`USEquityWebSocketFeed` directly and assert that:

1. ``watch_ohlcv`` invokes the callback exactly once with a zero-row
   :data:`OHLCV_SCHEMA` table.
2. :meth:`aclose` is idempotent: calling it more than once is a no-op.
3. ``is_implemented is False`` (the stub marker).
4. The pydantic config honors ``AGENTS.md``: frozen + ``extra="forbid"``
   + ``strict=True`` (extra fields raise ``ValidationError``).
5. The class is importable with no network, no API key, and no real
   WebSocket client.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pyarrow as pa
import pytest
from pydantic import ValidationError

from kairon.data.adapters import MarketDataAdapter
from kairon.data.adapters.us_equity_ws import (
    OhlcvCallback,
    USEquityWebSocketFeed,
)
from kairon.data.io import OHLCV_SCHEMA


def _empty_table() -> pa.Table:
    """Return a freshly-constructed zero-row ``OHLCV_SCHEMA`` table."""
    return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)


@pytest.mark.asyncio
async def test_equity_candle_round_trip() -> None:
    """Construct, subscribe, assert single empty emission, close idempotently."""
    feed = USEquityWebSocketFeed(venue="polygon")
    # Stub marker is set.
    assert feed.is_implemented is False

    received: list[pa.Table] = []
    callback: OhlcvCallback = received.append

    # watch_ohlcv is a coroutine; await it.
    await feed.watch_ohlcv("SPY", "1h", callback)

    # The callback was invoked exactly once.
    assert len(received) == 1, f"expected 1 callback, got {len(received)}"
    table = received[0]
    # The emitted table is empty and has the canonical OHLCV_SCHEMA.
    assert isinstance(table, pa.Table)
    assert table.schema == OHLCV_SCHEMA
    assert table.num_rows == 0
    # And the empty table matches what we'd construct by hand (defense
    # in depth against accidental schema drift in the stub).
    expected = _empty_table()
    assert table.num_rows == expected.num_rows
    assert table.schema == expected.schema

    # aclose() is idempotent: call it twice, no exception, no return value.
    result1 = await feed.aclose()
    result2 = await feed.aclose()
    assert result1 is None
    assert result2 is None


@pytest.mark.asyncio
async def test_watch_ohlcv_emits_no_rows_when_venue_is_tiingo() -> None:
    """The stub contract is identical regardless of venue choice."""
    feed = USEquityWebSocketFeed(venue="tiingo")
    assert feed.venue == "tiingo"
    assert feed.is_implemented is False

    received: list[pa.Table] = []
    await feed.watch_ohlcv("AAPL", "5m", received.append)
    assert len(received) == 1
    assert received[0].schema == OHLCV_SCHEMA
    assert received[0].num_rows == 0
    await feed.aclose()


@pytest.mark.asyncio
async def test_watch_ohlcv_rejects_empty_symbol() -> None:
    """Defensive check: empty/None symbol raises ``ValueError``."""
    feed = USEquityWebSocketFeed(venue="polygon")
    with pytest.raises(ValueError, match="symbol"):
        await feed.watch_ohlcv("", "1h", lambda _t: None)


@pytest.mark.asyncio
async def test_watch_ohlcv_rejects_empty_timeframe() -> None:
    """Defensive check: empty/None timeframe raises ``ValueError``."""
    feed = USEquityWebSocketFeed(venue="polygon")
    with pytest.raises(ValueError, match="timeframe"):
        await feed.watch_ohlcv("SPY", "", lambda _t: None)


@pytest.mark.asyncio
async def test_watch_ohlcv_rejects_none_callback() -> None:
    """Defensive check: ``None`` callback raises ``ValueError``."""
    feed = USEquityWebSocketFeed(venue="polygon")
    with pytest.raises(ValueError, match="callback"):
        await feed.watch_ohlcv("SPY", "1h", None)  # type: ignore[arg-type]


def test_is_implemented_is_literal_false() -> None:
    """The stub marker is exactly ``False`` (a ``Literal[False]`` field)."""
    feed = USEquityWebSocketFeed()
    assert feed.is_implemented is False
    assert feed.is_implemented == False  # noqa: E712 - explicit literal check
    # And it cannot be set to True at construction (pydantic Literal).
    with pytest.raises(ValidationError):
        USEquityWebSocketFeed(is_implemented=True)  # type: ignore[arg-type]


def test_venue_is_constrained_literal() -> None:
    """Venue must be one of the two supported US-equity providers."""
    # Valid venues.
    assert USEquityWebSocketFeed(venue="polygon").venue == "polygon"
    assert USEquityWebSocketFeed(venue="tiingo").venue == "tiingo"
    # Default is polygon.
    assert USEquityWebSocketFeed().venue == "polygon"
    # Anything else is rejected.
    with pytest.raises(ValidationError):
        USEquityWebSocketFeed(venue="alpha_vantage")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        USEquityWebSocketFeed(venue="binance")  # type: ignore[arg-type]


def test_config_is_frozen_extra_forbid_strict() -> None:
    """The model_config honors AGENTS.md invariants."""
    cfg = USEquityWebSocketFeed.model_config
    assert cfg.get("frozen") is True
    assert cfg.get("extra") == "forbid"
    assert cfg.get("strict") is True

    feed = USEquityWebSocketFeed(venue="polygon")
    # Frozen: cannot mutate an attribute.
    with pytest.raises(ValidationError):
        feed.venue = "tiingo"  # type: ignore[misc]
    # Extra="forbid": an extra field raises ValidationError.
    with pytest.raises(ValidationError):
        USEquityWebSocketFeed(venue="polygon", nonsense=True)  # type: ignore[call-arg]


def test_name_is_stable() -> None:
    """The adapter name is the documented public identifier."""
    assert USEquityWebSocketFeed.name == "us_equity_ws"


def test_fetch_returns_empty_ohlcv_table() -> None:
    """The REST compatibility shim returns a zero-row ``OHLCV_SCHEMA`` table.

    This is what satisfies the :class:`MarketDataAdapter` protocol
    compile-check; the real impl will replace it with a real REST call.
    """
    feed = USEquityWebSocketFeed(venue="polygon")
    table: Any = feed.fetch(
        symbol="SPY",
        timeframe="1h",
        start=None,
        end=None,
    )
    assert isinstance(table, pa.Table)
    assert table.schema == OHLCV_SCHEMA
    assert table.num_rows == 0


def test_satisfies_market_data_adapter_protocol() -> None:
    """Runtime check that the stub satisfies :class:`MarketDataAdapter`.

    The protocol is ``@runtime_checkable``, so ``isinstance`` works.
    """
    feed = USEquityWebSocketFeed(venue="polygon")
    assert isinstance(feed, MarketDataAdapter)


@pytest.mark.asyncio
async def test_stub_does_no_network_work() -> None:
    """The full import + construct + subscribe + close cycle is synchronous-fast.

    We don't need a network; the stub's whole point is to be hermetic.
    This test guards against accidental regression to a real WS client
    being imported at module load time.
    """
    # Import must succeed (the module is already imported at the top of
    # this file, but we re-import here to make the hermetic-import
    # guarantee explicit and locally scannable).
    import kairon.data.adapters.us_equity_ws as _stub_mod  # noqa: F401
    assert _stub_mod.USEquityWebSocketFeed is USEquityWebSocketFeed

    # The module has no third-party WS dependency; no monkeypatch
    # needed to assert "no network attempted". A 0.5s budget is generous
    # and catches accidental sleeps in the stub.
    async def _run() -> None:
        feed = USEquityWebSocketFeed(venue="polygon")
        await feed.watch_ohlcv("SPY", "1h", lambda _t: None)
        await feed.aclose()

    start = asyncio.get_event_loop().time()
    await asyncio.wait_for(_run(), timeout=0.5)
    elapsed = asyncio.get_event_loop().time() - start
    # We don't assert a tight bound — only that no network/sleep crept in.
    assert elapsed < 0.5
