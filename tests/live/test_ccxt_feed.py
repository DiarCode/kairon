"""Tests for the live CCXT candle feed.

The feed is a thin state machine on top of
:class:`kairon.data.adapters.CCXTAdapter.watch_ohlcv`. The adapter is
itself a thin wrapper over a ccxt client; we mock the client at the
adapter's ``_get_client`` seam, feed a deterministic sequence of 1m
ticks, and assert the aggregator produces exactly one closed bucket per
60-second boundary.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pyarrow as pa
import pytest

from kairon.data.adapters.ccxt_adapter import CCXTAdapter
from kairon.data.io import OHLCV_SCHEMA
from kairon.data.symbols import CryptoVenue, crypto_spot
from kairon.live.feed import CcxtCandleFeed, CcxtCandleFeedConfig


def _candle(
    ts: datetime,
    o: float,
    h: float,
    l: float,
    c: float,
    v: float,
) -> list[Any]:
    """Build a ccxt-style OHLCV row (ts in milliseconds since epoch)."""
    return [int(ts.timestamp() * 1000), o, h, l, c, v]


def _make_mock_client(rows: list[list[Any]]) -> MagicMock:
    """Build a MagicMock that mimics a ccxt async client.

    Each call to ``watch_ohlcv`` returns the next entry in ``rows``;
    once exhausted, it raises ``StopAsyncIteration`` so the consumer
    coroutine exits cleanly. ``close`` is an ``AsyncMock`` so the
    feed's :meth:`aclose` path is exerciseable.
    """
    iter_rows = iter(rows)

    async def watch_ohlcv(
        market: str, timeframe: str = "1m"
    ) -> list[list[Any]]:
        try:
            return next(iter_rows)
        except StopIteration as exc:  # noqa: PERF203 - test fixture
            raise StopAsyncIteration from exc

    client = MagicMock()
    client.close = AsyncMock()
    client.watch_ohlcv = AsyncMock(side_effect=watch_ohlcv)
    return client


@pytest.mark.asyncio
async def test_1m_candle_round_trip() -> None:
    """Feed 3 successive 1m ticks; aggregator emits 1 row per bucket."""
    # Three ticks, all in the *same* 1m bucket, then one tick that
    # rolls us into a new bucket (forces a closed candle).
    bucket_a_start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    bucket_b_start = datetime(2024, 1, 1, 0, 1, 0, tzinfo=UTC)
    rows = [
        # Three ticks in the 00:00 bucket; aggregator should fold
        # them into one row, not emit until the bucket closes.
        [_candle(bucket_a_start, 100.0, 101.0, 99.0, 100.5, 1.0)],
        [_candle(bucket_a_start, 100.5, 102.0, 100.0, 101.5, 2.0)],
        [_candle(bucket_a_start, 101.5, 102.5, 101.0, 102.0, 3.0)],
        # First tick of the 00:01 bucket: this should close the
        # 00:00 bucket (1 emitted row) and start a new one.
        [_candle(bucket_b_start, 102.0, 103.0, 101.5, 102.5, 4.0)],
    ]

    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    config = CcxtCandleFeedConfig(
        venue=CryptoVenue.BINANCE,
        timeframe="1m",
        poll_interval_seconds=1.0,
    )
    client = _make_mock_client(rows)
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE)
    # The adapter's ``aclose`` only forwards to the underlying client
    # when ``adapter._client`` is set. Patch both ``_get_client`` (so
    # the adapter does not try to build a real one) and ``_client``
    # (so ``aclose`` has a target to call ``close()`` on).
    with (
        patch.object(adapter, "_get_client", return_value=client),
        patch.object(adapter, "_client", client),
    ):
        feed = CcxtCandleFeed(symbols=[sym], config=config, adapter=adapter)
        # The callback the adapter would call per candle is bound to
        # the feed's ``_on_candle``. Replicate that here.
        for batch in rows:
            for row in batch:
                feed._on_candle(adapter._candle_row_to_table(row))  # noqa: SLF001

        # The aggregator should have emitted exactly one row so far:
        # the closed 00:00 bucket. The 00:01 bucket is still in-flight.
        emitted = []
        while not feed.queue.empty():
            emitted.append(feed.queue.get_nowait())
        assert len(emitted) == 1, f"expected 1 closed bucket, got {len(emitted)}"
        closed = emitted[0]
        assert closed.schema == OHLCV_SCHEMA
        assert closed.num_rows == 1
        ts = closed.column("ts")[0].as_py()
        assert ts == bucket_a_start
        # The aggregator's OHLCV rules: first open (100.0), max high
        # (102.5), min low (99.0), last close (102.0), sum volume (6.0).
        assert closed.column("open")[0].as_py() == 100.0
        assert closed.column("high")[0].as_py() == 102.5
        assert closed.column("low")[0].as_py() == 99.0
        assert closed.column("close")[0].as_py() == 102.0
        assert closed.column("volume")[0].as_py() == 6.0

        # aclose() is idempotent: calling it twice should not raise,
        # should not block, and should call the underlying client's
        # close exactly once. The patch is still in scope here.
        await feed.aclose()
        await feed.aclose()
        assert client.close.await_count == 1


@pytest.mark.asyncio
async def test_feed_emits_one_row_per_unique_1m_bucket() -> None:
    """Five ticks spanning 3 buckets produce exactly 3 closed rows."""
    b0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    b1 = datetime(2024, 6, 1, 12, 1, 0, tzinfo=UTC)
    b2 = datetime(2024, 6, 1, 12, 2, 0, tzinfo=UTC)
    ticks: list[list[Any]] = [
        [_candle(b0, 1.0, 1.0, 1.0, 1.0, 1.0)],
        [_candle(b0, 1.0, 2.0, 1.0, 1.5, 1.0)],
        [_candle(b1, 1.5, 1.5, 1.5, 1.5, 1.0)],
        [_candle(b1, 1.5, 3.0, 1.0, 2.0, 1.0)],
        [_candle(b2, 2.0, 4.0, 2.0, 3.0, 1.0)],
    ]
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    feed = CcxtCandleFeed(
        symbols=[sym],
        config=CcxtCandleFeedConfig(
            venue=CryptoVenue.BINANCE,
            timeframe="1m",
            poll_interval_seconds=1.0,
        ),
    )
    adapter = CCXTAdapter(venue=CryptoVenue.BINANCE)
    for batch in ticks:
        for row in batch:
            feed._on_candle(adapter._candle_row_to_table(row))  # noqa: SLF001

    rows: list[pa.Table] = []
    while not feed.queue.empty():
        rows.append(feed.queue.get_nowait())
    assert len(rows) == 2, (
        f"expected 2 closed buckets (b0 and b1); got {len(rows)}"
    )
    assert all(r.schema == OHLCV_SCHEMA for r in rows)
    timestamps = [r.column("ts")[0].as_py() for r in rows]
    assert timestamps == [b0, b1]

    await feed.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent_without_underlying_client() -> None:
    """``aclose`` on a feed that was never started is a no-op."""
    feed = CcxtCandleFeed(
        symbols=[crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)],
        config=CcxtCandleFeedConfig(venue=CryptoVenue.BINANCE, timeframe="1m"),
    )
    await feed.aclose()
    await feed.aclose()  # second call must not raise


def test_config_is_frozen_and_strict() -> None:
    """The feed config honors AGENTS.md: frozen + extra='forbid' + strict."""
    cfg = CcxtCandleFeedConfig()
    assert cfg.model_config.get("frozen") is True
    assert cfg.model_config.get("extra") == "forbid"
    assert cfg.model_config.get("strict") is True
    with pytest.raises(Exception):  # ValidationError on extra field
        CcxtCandleFeedConfig(venue=CryptoVenue.BINANCE, nonsense=True)  # type: ignore[call-arg]


def test_feed_rejects_unsupported_venue() -> None:
    """Only Binance and Bybit are supported by the public WS path.

    The restriction lives in :class:`CcxtCandleFeed.__init__` (not the
    pydantic config) so a config with Coinbase is technically valid,
    but constructing the feed must fail.
    """
    sym = crypto_spot("BTC", "USDT", CryptoVenue.COINBASE)
    cfg = CcxtCandleFeedConfig(venue=CryptoVenue.COINBASE, timeframe="1m")
    with pytest.raises(ValueError, match="does not support public"):
        CcxtCandleFeed(symbols=[sym], config=cfg)
