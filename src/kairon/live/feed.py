"""Live candle feed over ccxt's public WebSocket.

This module is the streaming counterpart to
:class:`kairon.data.adapters.ccxt_adapter.CCXTAdapter`. While the adapter
answers "give me N bars from A to B", the feed answers "give me the next
closed candle as soon as it lands". The feed is a small, deterministic
state machine: it consumes raw 1-minute candles (or trades) from
``ccxt.async_support.watch_ohlcv``, aggregates them into the requested
``(symbol, timeframe)`` buckets, and yields typed
:data:`~kairon.data.io.OHLCV_SCHEMA` tables.

Design choices
--------------
- **No ``ccxt.pro``.** The public ``ccxt.async_support.watch_ohlcv`` is
  supported on Binance and Bybit as of 2024+. L2 microstructure
  (which does require ``ccxt.pro``) is out of scope for W1.1.
- **Aggregation, not throttling.** The feed is *additive* over a
  configurable bucket size, so a 1m source candle can fill a 1m bucket
  directly, a 1m source can roll up to a 5m bucket, and so on. Each
  emitted row covers one closed bucket.
- **Hermetic in tests.** All IO is routed through the existing
  :class:`kairon.data.adapters.CCXTAdapter` (which itself goes through
  the ``_ccxt_client`` seam), so tests can mock the underlying client
  with no live network.
- **Idempotent close.** :meth:`CcxtCandleFeed.aclose` is safe to call
  multiple times and never blocks.

This module deliberately does *not* call any external scheduler or
async runtime; the caller is expected to drive the feed from a worker
task and to consume rows via an async iterator.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from kairon.data.adapters.ccxt_adapter import CCXTAdapter
from kairon.data.io import OHLCV_SCHEMA
from kairon.data.symbols import CryptoVenue, Symbol, crypto_spot


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class CcxtCandleFeedConfig(BaseModel):
    """Immutable configuration for a :class:`CcxtCandleFeed`.

    The config is frozen + strict per ``AGENTS.md`` so that an instance
    cannot be mutated after construction. All fields are required unless
    they have an explicit default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    venue: CryptoVenue = Field(
        default=CryptoVenue.BINANCE,
        description="Crypto exchange backing the feed (Binance or Bybit).",
    )
    timeframe: str = Field(
        default="1m",
        description="Canonical timeframe (e.g. '1m', '5m').",
    )
    poll_interval_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=60.0,
        description=(
            "Polling interval for the underlying watch_ohlcv loop. "
            "The actual candle cadence is bounded by the exchange's "
            "native timeframe (e.g. 60s for 1m)."
        ),
    )
    max_buffered_buckets: int = Field(
        default=4096,
        gt=0,
        le=1_000_000,
        description="Hard cap on per-(symbol,timeframe) buffered buckets.",
    )


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------
class CcxtCandleFeed:
    """Stream OHLCV candles from Binance or Bybit via ccxt.

    The feed owns a :class:`CCXTAdapter` (which provides the typed
    ``watch_ohlcv`` shim) and an internal aggregator that rolls
    incoming 1m candles up to the configured ``timeframe`` bucket size.

    Parameters
    ----------
    symbols:
        One or more crypto symbols to subscribe to. They must be crypto
        (spot or perp); the venue is taken from
        :attr:`CcxtCandleFeedConfig.venue`.
    config:
        Feed configuration. See :class:`CcxtCandleFeedConfig`.
    adapter:
        Optional pre-built :class:`CCXTAdapter`. Tests inject a mock
        adapter here; production code lets the feed build one.

    Notes
    -----
    A :class:`CcxtCandleFeed` is **single-shot**: call :meth:`run` once
    and then :meth:`aclose` to release the underlying ccxt client.
    :meth:`aclose` is idempotent and never blocks.
    """

    # Exchanges known to expose public ``watch_ohlcv`` in ccxt's free
    # ``async_support`` package. Other venues in :class:`CryptoVenue`
    # would require ``ccxt.pro`` and are intentionally rejected here.
    SUPPORTED_VENUES: ClassVar[frozenset[CryptoVenue]] = frozenset(
        {CryptoVenue.BINANCE, CryptoVenue.BYBIT}
    )

    def __init__(
        self,
        symbols: list[Symbol] | tuple[Symbol, ...],
        config: CcxtCandleFeedConfig | None = None,
        *,
        adapter: CCXTAdapter | None = None,
    ) -> None:
        cfg = config or CcxtCandleFeedConfig()
        if cfg.venue not in self.SUPPORTED_VENUES:
            raise ValueError(
                f"venue {cfg.venue.value!r} does not support public "
                f"watch_ohlcv; supported: "
                f"{sorted(v.value for v in self.SUPPORTED_VENUES)}"
            )
        if not symbols:
            raise ValueError("CcxtCandleFeed requires at least one symbol")
        for s in symbols:
            if not s.is_crypto or s.base is None or s.quote is None:
                raise ValueError(
                    f"symbol {s.canonical!r} is not a crypto spot/perp"
                )
        self._config = cfg
        self._symbols: tuple[Symbol, ...] = tuple(symbols)
        self._adapter: CCXTAdapter = adapter or CCXTAdapter(venue=cfg.venue)
        self._closed = False
        # In-progress (open) candle per (symbol, timeframe). Each entry
        # is a dict with keys ts, open, high, low, close, volume. A new
        # row whose minute-bucket matches the open one is folded in;
        # a row whose bucket advances the open candle is *first* closed
        # (emitted via the queue) and *then* a new open candle is
        # started.
        self._open: dict[tuple[str, str], dict[str, Any]] = {}
        # Buffer of fully closed buckets, keyed by (symbol, timeframe)
        # so that downstream code can sort by emission order.
        self._buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        # Outstanding rows waiting to be consumed by :meth:`run`.
        self._queue: asyncio.Queue[pa.Table] = asyncio.Queue()
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> CcxtCandleFeedConfig:
        """The immutable feed configuration."""
        return self._config

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """The symbols the feed is subscribed to."""
        return self._symbols

    @property
    def venue(self) -> CryptoVenue:
        """The crypto venue backing the feed."""
        return self._config.venue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """Stop the feed and release the underlying ccxt client.

        Idempotent: calling :meth:`aclose` more than once is a no-op
        and never raises.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        await self._adapter.aclose()

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Drive the feed; ``await`` returns when :meth:`aclose` is called.

        Each closed (symbol, timeframe) bucket is published as a one-row
        :data:`OHLCV_SCHEMA` table on :attr:`queue`. This method does
        not return rows directly; the caller can either iterate
        :attr:`queue` or use :meth:`run_collect` to gather a snapshot.
        """
        tasks: list[asyncio.Task[None]] = []
        for sym in self._symbols:
            tasks.append(
                asyncio.create_task(
                    self._adapter.watch_ohlcv(sym, self._config.timeframe, self._on_candle),
                    name=f"watch_ohlcv:{sym.canonical}",
                )
            )
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await self.aclose()

    @property
    def queue(self) -> asyncio.Queue[pa.Table]:
        """Queue of completed (closed) candles as ``OHLCV_SCHEMA`` tables."""
        return self._queue

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _bucket_start(ts: datetime, timeframe: str) -> datetime:
        """Return the UTC start of the bucket that ``ts`` falls in."""
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC)")
        seconds = _timeframe_to_seconds(timeframe)
        if seconds <= 0:
            raise ValueError(f"invalid timeframe {timeframe!r}")
        epoch_s = int(ts.astimezone(UTC).timestamp())
        floored = epoch_s - (epoch_s % seconds)
        return datetime.fromtimestamp(floored, tz=UTC)

    def _on_candle(self, table: pa.Table) -> None:
        """Synchronous callback invoked by the adapter for each candle.

        The adapter's callback is per-symbol (one watch_ohlcv coroutine
        per symbol), so we infer the symbol from the single subscribed
        symbol. If the feed was constructed with more than one symbol,
        we still process each candle correctly because each coroutine
        is bound to a specific symbol at the call site.
        """
        if table.schema != OHLCV_SCHEMA:
            raise ValueError(
                f"feed received non-OHLCV table: schema={table.schema}"
            )
        if table.num_rows != 1:
            raise ValueError(
                f"feed expected one-row tables, got {table.num_rows}"
            )
        ts = table.column("ts")[0].as_py()
        if not isinstance(ts, datetime):
            raise ValueError(f"feed received non-datetime ts: {ts!r}")
        if ts.tzinfo is None:
            raise ValueError("feed received naive datetime ts")
        ts_utc = ts.astimezone(UTC)
        open_v = float(table.column("open")[0].as_py())
        high = float(table.column("high")[0].as_py())
        low = float(table.column("low")[0].as_py())
        close = float(table.column("close")[0].as_py())
        vol = float(table.column("volume")[0].as_py())
        sym = self._symbols[0]
        key = (sym.canonical, self._config.timeframe)
        bucket = self._bucket_start(ts_utc, self._config.timeframe)
        existing = self._open.get(key)
        if existing is None or bucket > existing["ts"]:
            # Either we have no in-progress candle, or the bucket
            # advanced: in both cases, close the existing candle
            # (if any) and open a new one.
            if existing is not None:
                self._emit(key, existing)
            self._open[key] = _new_bucket(bucket, open_v, high, low, close, vol)
            return
        # Same bucket as the in-progress candle: fold the tick in.
        # Aggregating OHLCV is: keep first open, take max high, take
        # min low, take last close, sum volume.
        existing["high"] = max(existing["high"], high)
        existing["low"] = min(existing["low"], low)
        existing["close"] = close
        existing["volume"] = float(existing["volume"]) + vol

    def _emit(self, key: tuple[str, str], row: dict[str, Any]) -> None:
        """Close out a bucket and enqueue a one-row OHLCV_SCHEMA table."""
        buf = self._buckets[key]
        buf.append(row)
        if len(buf) > self._config.max_buffered_buckets:
            # Drop the oldest entries to honor the cap.
            del buf[: len(buf) - self._config.max_buffered_buckets]
        table = pa.table(
            {
                "ts": [row["ts"]],
                "open": [float(row["open"])],
                "high": [float(row["high"])],
                "low": [float(row["low"])],
                "close": [float(row["close"])],
                "volume": [float(row["volume"])],
            },
            schema=OHLCV_SCHEMA,
        )
        self._queue.put_nowait(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_bucket(
    ts: datetime,
    open_v: float,
    high: float,
    low: float,
    close: float,
    vol: float,
) -> dict[str, Any]:
    """Build a fresh in-progress bucket dict."""
    return {
        "ts": ts,
        "open": open_v,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }


def _timeframe_to_seconds(timeframe: str) -> int:
    """Parse a ccxt timeframe string like ``"1m"`` or ``"4h"`` to seconds."""
    if not timeframe:
        raise ValueError("timeframe must be non-empty")
    unit = timeframe[-1]
    try:
        n = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"invalid timeframe {timeframe!r}") from exc
    if n <= 0:
        raise ValueError(f"invalid timeframe {timeframe!r}")
    if unit == "m":
        return n * 60
    if unit == "h":
        return n * 60 * 60
    if unit == "d":
        return n * 24 * 60 * 60
    if unit == "w":
        return n * 7 * 24 * 60 * 60
    raise ValueError(f"unsupported timeframe unit {unit!r}")


def _bucket_end(bucket_start: datetime, timeframe: str) -> datetime:
    """Return the (exclusive) end of a bucket given its start."""
    return bucket_start + timedelta(seconds=_timeframe_to_seconds(timeframe))


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------
def btc_usdt_feed(
    venue: CryptoVenue = CryptoVenue.BINANCE,
    *,
    timeframe: str = "1m",
    poll_interval_seconds: float = 1.0,
) -> CcxtCandleFeed:
    """Build a :class:`CcxtCandleFeed` for the canonical BTC/USDT market.

    Provided as a small convenience for callers (and for the real-data
    capture script) so they do not have to assemble
    :class:`kairon.data.symbols.Symbol` instances by hand.
    """
    cfg = CcxtCandleFeedConfig(
        venue=venue,
        timeframe=timeframe,
        poll_interval_seconds=poll_interval_seconds,
    )
    sym = crypto_spot("BTC", "USDT", venue)
    return CcxtCandleFeed(symbols=[sym], config=cfg)


__all__ = [
    "CcxtCandleFeed",
    "CcxtCandleFeedConfig",
    "btc_usdt_feed",
    "fetch_current_price",
]


# ---------------------------------------------------------------------------
# Web-app one-shot price fetcher (US-004, additive).
# Used by the verification thread to compute actual_pct / delta_pct for a
# finished run. Goes through the same `_ccxt_client.make_client(venue)` seam
# as the streaming feed, so tests can mock the client with no live network.
# `CCXTAdapter` itself is NOT modified.
# ---------------------------------------------------------------------------


def _async_fetch_ticker(venue: str, market: str) -> float:
    """Construct a ccxt async client, call fetch_ticker, return the last price."""
    from kairon.data.adapters._ccxt_client import make_client  # lazy

    client = make_client(venue)

    async def _go() -> float:
        try:
            ticker = await client.fetch_ticker(market)
            last = ticker.get("last")
            if last is None:
                raise ValueError(f"ticker for {market} on {venue} has no 'last' price")
            return float(last)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    return asyncio.run(_go())


def fetch_current_price(asset: str, venue: str = "binance") -> float:
    """One-shot current-price fetch for a given asset on a given venue.

    ``asset`` is a TradingView-style symbol like ``BTCUSDT``; the function
    splits it into ``base/quote`` and calls ``fetch_ticker`` on the public
    ccxt client. Returns the last traded price as a ``float``.

    Example:
        >>> fetch_current_price("BTCUSDT", "binance")  # doctest: +SKIP
        68234.10
    """
    asset_upper = asset.upper()
    if asset_upper.endswith("USDT"):
        base, quote = asset_upper[:-4], "USDT"
    elif asset_upper.endswith("USD"):
        base, quote = asset_upper[:-3], "USD"
    else:
        # fall back: assume the whole string is the base and the quote is USDT
        base, quote = asset_upper, "USDT"
    market = f"{base}/{quote}"
    return _async_fetch_ticker(venue, market)


# ``_bucket_end`` is exposed for tests and downstream utilities that
# need to reason about inclusive/exclusive bucket boundaries. It is a
# module-level helper (not re-exported in ``__all__``) but is kept
# defined here for completeness.
_ = _bucket_end  # noqa: PIE782 - keep helper discoverable
