"""US-equity WebSocket feed stub (Polygon / Tiingo).

This module ships a typed stub of the US-equity streaming adapter. The
full implementation (Polygon ``wss://socket.polygon.io/stocks`` and
Tiingo ``wss://api.tiingo.com/iex``) is deferred to a future 1-PR change
because the engineer executing W1.2 does not have a Polygon or Tiingo
API key (see ``reports/w0_fallback.md``).

What the stub does
------------------
- Compiles. The class is a ``pydantic v2`` model with
  ``frozen=True, extra="forbid", strict=True`` per ``AGENTS.md`` and a
  ``Literal[False]`` ``is_implemented`` field that marks it as a stub.
- Satisfies the :class:`~kairon.data.adapters.MarketDataAdapter`
  protocol (a no-op ``fetch`` and a stable ``name``).
- Exposes a streaming-shaped :meth:`watch_ohlcv` that immediately calls
  the caller-supplied ``callback`` once with an empty
  :data:`~kairon.data.io.OHLCV_SCHEMA` table and returns. The callback
  contract matches the live feed's aggregator wiring so callers can
  plumb the interface end-to-end before the real WebSocket lands.
- Has an idempotent :meth:`aclose` that is a no-op for the stub.

What the stub does NOT do
-------------------------
- It does not open a network connection.
- It does not require an API key.
- It does not emit any real candle rows. The empty-table emission is a
  signal to the caller that the channel is "connected" but no data has
  arrived yet (matching the W0 fallback contract).

What the full implementation will require
-----------------------------------------
- An API key in ``KAIRON_POLYGON_API_KEY`` (or ``KAIRON_TIINGO_API_KEY``).
- A WebSocket client (``websockets`` or ``httpx-ws``); the W1.4 settings
  story adds the env var.
- A per-venue message parser (Polygon ``T.*`` and ``AM.*`` schemas;
  Tiingo ``tiingo`` and ``iex`` channels) that normalizes into
  :data:`~kairon.data.io.OHLCV_SCHEMA`.
- A rate-limit / reconnection policy (Polygon free tier: 100 msg/min
  per socket; Tiingo IEX: 50 symbols/socket). The full impl will mirror
  the retry/backoff pattern used in :class:`CCXTAdapter`.
"""

from __future__ import annotations

from typing import Callable, ClassVar, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from kairon.data.adapters import MarketDataAdapter
from kairon.data.io import OHLCV_SCHEMA

# Callback contract: the feed invokes the callback once per emitted
# candle with a one-row (or zero-row, for the stub) ``OHLCV_SCHEMA``
# pyarrow table. We type it as a plain ``Callable`` to keep pyright
# strict-mode happy without leaking third-party types, exactly like
# :data:`kairon.data.adapters.ccxt_adapter.OhlcvCallback`.
OhlcvCallback = Callable[[pa.Table], None]


def _empty_ohlcv_table() -> pa.Table:
    """Build a zero-row pyarrow table with the canonical ``OHLCV_SCHEMA``."""
    return pa.table({f.name: [] for f in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA)


class USEquityWebSocketFeed(BaseModel):
    """US-equity streaming feed (Polygon or Tiingo) — STUB.

    Parameters
    ----------
    venue:
        Which US-equity provider this feed would talk to. The stub does
        not actually open a connection; it only records the choice so
        downstream code (e.g. a future ``resolve_ws_url(venue)`` helper)
        can pick the right endpoint once the API key is available.
    is_implemented:
        Literal ``False`` for the stub. The real implementation will
        override this to ``True`` and replace the no-op ``watch_ohlcv``
        with a real WebSocket loop.

    Notes
    -----
    - The class is frozen + strict per ``AGENTS.md``: callers cannot
      mutate ``venue`` after construction, and any extra field raises
      ``ValidationError`` (caught in :file:`tests/data/test_us_equity_ws.py`).
    - The stub is importable in tests with no network, no API key, and
      no scheduler; it is fully hermetic.
    - The streaming contract is intentionally synchronous (callback
      returns ``None``), matching :meth:`CCXTAdapter.watch_ohlcv` so the
      upstream :class:`kairon.live.feed.CcxtCandleFeed`-style aggregator
      can consume it unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    # Public marker: ``Literal[False]`` so the type system *and* the
    # runtime both know this is a stub. The real implementation will
    # narrow this to ``bool`` and default to ``True``.
    is_implemented: Literal[False] = Field(
        default=False,
        description=(
            "Stub marker. Real WebSocket impl will set this to True "
            "and replace watch_ohlcv with a live loop."
        ),
    )

    venue: Literal["polygon", "tiingo"] = Field(
        default="polygon",
        description="US-equity data provider. Stub records the choice; no network is opened.",
    )

    # The MarketDataAdapter protocol requires a stable ``name``. The
    # stub uses ``"us_equity_ws"`` (the convention from ccxt_adapter's
    # ``name = "ccxt"``); the real impl may keep this or specialize per
    # venue (``"polygon_ws"`` / ``"tiingo_ws"``) once a single choice
    # is locked in.
    name: ClassVar[str] = "us_equity_ws"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """Release resources held by the feed.

        For the stub, this is a no-op. The real implementation will
        close the underlying WebSocket and cancel any in-flight tasks.
        ``aclose`` is idempotent: calling it more than once is safe.
        """
        # No-op by design. The stub has no client to close; we keep the
        # method present so callers can wire ``try/finally feed.aclose()``
        # patterns today and have them Just Work once the real impl lands.
        return None

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------
    async def watch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        callback: OhlcvCallback,
    ) -> None:
        """Stream OHLCV candles for a US equity. STUB: emits empty table once.

        Parameters
        ----------
        symbol:
            The equity ticker (e.g. ``"SPY"``). The stub does not
            validate it; the real implementation will forward it to the
            venue's subscribe message.
        timeframe:
            Our canonical timeframe (e.g. ``"1m"``, ``"1h"``). The stub
            does not validate it; the real implementation will translate
            it to the venue-native aggregation.
        callback:
            Synchronous callable invoked with a one-row (or zero-row,
            for the stub) :data:`OHLCV_SCHEMA` pyarrow table per emitted
            candle.

        Notes
        -----
        The stub's behavior is: call ``callback`` exactly once with an
        empty :data:`OHLCV_SCHEMA` table and return. This lets the
        caller wire the full interface (register the feed, attach the
        aggregator, schedule ``aclose`` on shutdown) without needing
        real market data, a network connection, or an API key.

        The real implementation will replace this with a
        ``while not self._stop_event.is_set()`` loop over the
        venue-native WebSocket, converting each tick to a one-row
        :data:`OHLCV_SCHEMA` table and forwarding it to ``callback``.
        """
        # Defensive shape checks. The real implementation will use the
        # same checks before forwarding to the parser.
        if not symbol:
            raise ValueError(f"symbol must be a non-empty string, got {symbol!r}")
        if not timeframe:
            raise ValueError(f"timeframe must be a non-empty string, got {timeframe!r}")
        if callback is None:
            raise ValueError("callback is required")
        # Single emission: empty OHLCV table, exactly as the W0
        # fallback contract specifies. Return immediately; there is no
        # background loop in the stub.
        callback(_empty_ohlcv_table())
        return None

    # ------------------------------------------------------------------
    # REST compatibility (MarketDataAdapter protocol)
    # ------------------------------------------------------------------
    def fetch(
        self,
        symbol: object,
        timeframe: str,
        start: object,
        end: object,
    ) -> pa.Table:
        """Return an empty OHLCV table.

        Satisfies :class:`~kairon.data.adapters.MarketDataAdapter` so
        the stub compile-checks against the protocol. The real
        implementation will translate ``(symbol, timeframe, start, end)``
        to a Polygon/Tiingo REST request and return real rows.
        """
        # No input validation here: the protocol's signature is
        # ``fetch(symbol, timeframe, start, end)`` and pyright's
        # structural check only requires the method to exist with the
        # right shape. The stub's job is to type-check, not to enforce
        # runtime semantics.
        _ = (symbol, timeframe, start, end)
        return _empty_ohlcv_table()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
def _ensure_protocol() -> None:
    """Compile-time check that ``USEquityWebSocketFeed`` satisfies the protocol.

    Mirrors :func:`kairon.data.adapters.ccxt_adapter._ensure_protocol` and
    :func:`kairon.data.adapters.fred._ensure_protocol`. The
    ``# type: ignore[assignment]`` is justified: ``USEquityWebSocketFeed``
    is a ``pydantic`` ``BaseModel`` with extra fields (``is_implemented``)
    that the narrow :class:`MarketDataAdapter` protocol does not declare.
    Once the real implementation lands and the pydantic field is removed
    (or defaulted to ``True``), this ignore can be revisited.
    """
    adapter: MarketDataAdapter = USEquityWebSocketFeed(venue="polygon")  # type: ignore[assignment]
    _ = adapter


_ensure_protocol()


__all__ = ["USEquityWebSocketFeed", "OhlcvCallback"]
