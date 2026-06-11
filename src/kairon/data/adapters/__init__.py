"""Adapter base classes.

All data adapters (CCXT, Polygon, Tiingo, FRED, Glassnode, ...) implement
the ``MarketDataAdapter`` protocol. The protocol is intentionally narrow:
fetch a typed frame, return its hash. Anything else (caching, retries,
parallelism) lives in the orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pyarrow as pa

from kairon.data.symbols import Symbol


@runtime_checkable
class MarketDataAdapter(Protocol):
    """Minimal contract for any market data adapter.

    Implementations fetch OHLCV bars for a single (symbol, timeframe)
    pair over a closed-open time interval and return a pyarrow table
    with the canonical ``OHLCV_SCHEMA``. Adapters must be:

    - **Idempotent.** Calling ``fetch`` twice with the same args
      should return the same data.
    - **Typed.** Every input is typed; every output is a pyarrow table
      with ``OHLCV_SCHEMA``.
    - **Bounded.** Implementations are responsible for chunking large
      ranges into API-supported windows; callers never see partial
      windows.
    """

    name: str

    def fetch(
        self,
        symbol: Symbol,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pa.Table: ...


class AdapterError(RuntimeError):
    """Raised when an adapter cannot complete a request."""


class RateLimitedError(AdapterError):
    """Raised when an adapter is being rate-limited by its upstream."""
