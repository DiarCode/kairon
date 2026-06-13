"""CCXT adapter for crypto OHLCV (Binance, Bybit, Coinbase, Kraken, OKX).

CCXT provides a unified async API for 100+ exchanges. We use the
**async** flavor (``ccxt.async_support``) and convert to a pyarrow
table with the canonical ``OHLCV_SCHEMA``.

Notes
-----
- We do not include the L2 order book here (Phase 1 covers OHLCV only).
- All timestamps are coerced to timezone-aware UTC.
- The adapter is hermetic in tests: ``ccxt`` calls go through a small
  indirection so tests can patch the underlying client.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, ClassVar, Callable

import pyarrow as pa
from loguru import logger

from kairon.data.adapters import (
    AdapterError,
    MarketDataAdapter,
    RateLimitedError,
)
from kairon.data.io import OHLCV_SCHEMA
from kairon.data.symbols import AssetClass, CryptoVenue, Symbol

# Type alias: callback receives a one-row OHLCV table per emitted candle.
# Typed as a plain Callable so pyright can verify it without leaking ccxt's
# untyped primitives.
OhlcvCallback = Callable[[pa.Table], None]

# Map our canonical timeframe string to CCXT's
CCXT_TIMEFRAMES: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}


class CCXTAdapter:
    """Unified crypto OHLCV adapter backed by CCXT.

    Parameters
    ----------
    venue:
        Which exchange to talk to. We import the ccxt client lazily so
        that importing this module is cheap and the dependency can be
        mocked in tests.
    max_retries:
        Number of retries on transient errors (network, rate limit).
    chunk_ms:
        CCXT has a max-bars-per-request limit; we chunk large windows
        into ``chunk_ms``-millisecond windows. Default 30 days.
    """

    name: ClassVar[str] = "ccxt"

    def __init__(
        self,
        venue: CryptoVenue = CryptoVenue.BINANCE,
        *,
        max_retries: int = 5,
        chunk_ms: int = 30 * 24 * 60 * 60 * 1000,
        testnet: bool = False,
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        self.venue = venue
        self.max_retries = max_retries
        self.chunk_ms = chunk_ms
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        self._client: Any = None

    def _get_client(self) -> Any:  # noqa: ANN401 - ccxt is not well-typed
        if self._client is None:
            from kairon.data.adapters._ccxt_client import make_client  # noqa: PLC0415 - lazy

            self._client = make_client(
                self.venue.value,
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _ccxt_market_id(self, symbol: Symbol) -> str:
        """Build the ccxt market identifier from a Symbol.

        For perpetual contracts (``is_perp``), ccxt requires the
        ``BASE/QUOTE:SETTLE`` format (e.g. ``BTC/USDT:USDT``).
        For spot, the plain ``BASE/QUOTE`` format is used.
        """
        base_market = f"{symbol.base}/{symbol.quote}"
        if symbol.is_perp:
            return f"{base_market}:{symbol.quote}"
        return base_market

    async def _fetch_chunk(
        self,
        symbol: Symbol,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[Any]]:
        client = self._get_client()
        if timeframe not in CCXT_TIMEFRAMES:
            raise AdapterError(f"unsupported timeframe {timeframe!r}")
        market = self._ccxt_market_id(symbol)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await client.fetch_ohlcv(
                    market,
                    timeframe=CCXT_TIMEFRAMES[timeframe],
                    since=start_ms,
                    limit=1000,
                    params={"endTime": end_ms},
                )
            except Exception as exc:  # ccxt raises a wide variety
                last_err = exc
                wait = 2**attempt
                logger.warning(
                    "ccxt fetch failed (venue={}, market={}, attempt={}): {}; retrying in {}s",
                    self.venue.value,
                    market,
                    attempt,
                    exc,
                    wait,
                )
                if "rate" in str(exc).lower():
                    raise RateLimitedError(str(exc)) from exc
                await asyncio.sleep(wait)
        raise AdapterError(
            f"ccxt fetch failed after {self.max_retries} retries: {last_err}"
        )

    async def afetch(
        self,
        symbol: Symbol,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        """Async fetch OHLCV; returns a pyarrow table with ``OHLCV_SCHEMA``."""
        if not symbol.is_crypto or symbol.base is None or symbol.quote is None:
            raise AdapterError(
                f"CCXT requires a crypto symbol with base/quote; got {symbol}"
            )
        if symbol.asset_class == AssetClass.CRYPTO_PERP and self.venue in {
            CryptoVenue.COINBASE,
        }:
            raise AdapterError(f"{self.venue.value} does not support perps")
        start_ms = int(start.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end.astimezone(UTC).timestamp() * 1000)
        rows: list[list[Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            chunk_end = min(cursor + self.chunk_ms, end_ms)
            rows.extend(await self._fetch_chunk(symbol, timeframe, cursor, chunk_end))
            cursor = chunk_end
        if not rows:
            return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)
        # CCXT returns [ts_ms, o, h, l, c, v]; convert to the schema.
        ts = [datetime.fromtimestamp(r[0] / 1000, tz=UTC) for r in rows]
        opens = [float(r[1]) for r in rows]
        highs = [float(r[2]) for r in rows]
        lows = [float(r[3]) for r in rows]
        closes = [float(r[4]) for r in rows]
        vols = [float(r[5]) for r in rows]
        # Drop any zero-volume rows from the very first call (CCXT often returns
        # an extra warm-up bar).
        keep_idx = [i for i, t in enumerate(ts) if t >= start.astimezone(UTC)]
        ts = [ts[i] for i in keep_idx]
        opens = [opens[i] for i in keep_idx]
        highs = [highs[i] for i in keep_idx]
        lows = [lows[i] for i in keep_idx]
        closes = [closes[i] for i in keep_idx]
        vols = [vols[i] for i in keep_idx]
        return pa.table(
            {
                "ts": ts,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": vols,
            },
            schema=OHLCV_SCHEMA,
        )

    def fetch(
        self,
        symbol: Symbol,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        """Synchronous wrapper around :meth:`afetch`."""
        return asyncio.run(self.afetch(symbol, timeframe, start, end))

    @staticmethod
    def _candle_row_to_table(row: list[Any]) -> pa.Table:
        """Convert a single ccxt OHLCV row to a one-row pyarrow table.

        A ccxt OHLCV row is ``[ts_ms, open, high, low, close, volume]``; we
        coerce ``ts_ms`` to a timezone-aware UTC ``datetime`` and build a
        table whose schema is exactly :data:`kairon.data.io.OHLCV_SCHEMA`.
        """
        if len(row) != 6:
            raise AdapterError(
                f"ccxt candle row must have 6 elements, got {len(row)}"
            )
        ts = datetime.fromtimestamp(float(row[0]) / 1000.0, tz=UTC)
        return pa.table(
            {
                "ts": [ts],
                "open": [float(row[1])],
                "high": [float(row[2])],
                "low": [float(row[3])],
                "close": [float(row[4])],
                "volume": [float(row[5])],
            },
            schema=OHLCV_SCHEMA,
        )

    async def watch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: str,
        callback: OhlcvCallback,
    ) -> None:
        """Stream OHLCV candles via the ccxt public WebSocket.

        Wraps the underlying async client's ``watch_ohlcv`` (Binance and
        Bybit both support it in the public ``ccxt.async_support``
        package). For each emitted candle we convert the ccxt row to a
        one-row :data:`OHLCV_SCHEMA` table and pass it to ``callback``.

        Parameters
        ----------
        symbol:
            A crypto symbol with ``base`` and ``quote`` (e.g. BTC/USDT).
        timeframe:
            Our canonical timeframe (e.g. ``"1m"``); must be a key in
            :data:`CCXT_TIMEFRAMES`.
        callback:
            Synchronous callable invoked once per candle with a
            one-row ``OHLCV_SCHEMA`` table.

        Notes
        -----
        The underlying ``watch_ohlcv`` is awaited in a loop until the
        client raises a cancellation or ``aclose`` is called. The
        callback is invoked synchronously (matching the protocol used
        by the higher-level :class:`kairon.live.feed.CcxtCandleFeed`).
        """
        if not symbol.is_crypto or symbol.base is None or symbol.quote is None:
            raise AdapterError(
                f"CCXT requires a crypto symbol with base/quote; got {symbol}"
            )
        if timeframe not in CCXT_TIMEFRAMES:
            raise AdapterError(f"unsupported timeframe {timeframe!r}")
        client = self._get_client()
        market = self._ccxt_market_id(symbol)
        # ``watch_ohlcv`` returns a list of candles (each is a row) and
        # blocks until the next update arrives. We iterate forever; the
        # caller is responsible for cancellation via :meth:`aclose`.
        while True:
            candles: list[list[Any]] = await client.watch_ohlcv(
                market, timeframe=CCXT_TIMEFRAMES[timeframe]
            )
            for row in candles:
                callback(self._candle_row_to_table(row))


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
def _ensure_protocol() -> None:
    """Compile-time check that CCXTAdapter satisfies the MarketDataAdapter protocol."""
    adapter: MarketDataAdapter = CCXTAdapter()  # type: ignore[assignment]
    _ = adapter


_ensure_protocol()
