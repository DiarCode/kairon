"""FRED adapter for US macro series.

The Federal Reserve Economic Data (FRED) API provides free, reliable
access to 800k+ macro time series (CPI, rates, M2, GDP, ...). We use
``httpx`` for the HTTP call and parse the JSON response into a pyarrow
table with a (ts, value) schema. The adapter does not aggregate or
transform; that lives in the features layer.

FRED requires an API key in the ``FRED_API_KEY`` env var (or a Bearer
token via ``FRED_API_KEY``). The adapter raises if it is missing.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx
import pyarrow as pa

from kairon.data.adapters import AdapterError, MarketDataAdapter
from kairon.data.symbols import Symbol

FRED_SCHEMA: pa.Schema = pa.schema(
    [("ts", pa.timestamp("us", tz="UTC")), ("value", pa.float64())]
)

FRED_BASE_URL = "https://api.stlouisfed.org"


class FredAdapter:
    """FRED macro series adapter.

    The "symbol" is interpreted as the FRED series ID (e.g., ``CPIAUCSL``
    for CPI, ``DGS10`` for 10y Treasury). The ``canonical`` field of the
    symbol must equal the series ID (uppercase).
    """

    name: str = "fred"

    def __init__(self, *, api_key: str | None = None, timeout_s: float = 30.0) -> None:
        key: str | None = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise AdapterError(
                "FRED adapter requires FRED_API_KEY env var or explicit api_key"
            )
        # After the guard, key is truthy, but pyright doesn't track that,
        # so we assert via a typed local.
        self._api_key: str = key
        self._timeout_s = timeout_s

    @property
    def api_key(self) -> str:
        """The FRED API key (public read-only view, for diagnostics/tests)."""
        return self._api_key

    def _to_arrow(self, observations: list[dict[str, str]]) -> pa.Table:
        ts: list[datetime] = []
        vals: list[float] = []
        for o in observations:
            date_str = o.get("date", "")
            value_str = o.get("value", ".")
            if not date_str or value_str == ".":
                continue
            try:
                year, month, day = (int(p) for p in date_str.split("-"))
            except ValueError:
                continue
            try:
                v = float(value_str)
            except ValueError:
                continue
            ts.append(datetime(year, month, day, tzinfo=UTC))
            vals.append(v)
        return pa.table({"ts": ts, "value": vals}, schema=FRED_SCHEMA)

    def fetch(
        self,
        symbol: Symbol,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        if start.tzinfo is None or end.tzinfo is None:
            raise AdapterError("FRED fetch requires timezone-aware UTC start/end")
        start_s = start.astimezone(UTC).strftime("%Y-%m-%d")
        end_s = end.astimezone(UTC).strftime("%Y-%m-%d")
        params = {
            "series_id": symbol.canonical,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start_s,
            "observation_end": end_s,
        }
        url = f"{FRED_BASE_URL}/fred/series/observations"
        try:
            resp = httpx.get(url, params=params, timeout=self._timeout_s)
        except httpx.HTTPError as exc:
            raise AdapterError(f"FRED HTTP error: {exc}") from exc
        if resp.status_code == 429:
            raise AdapterError("FRED rate limited (HTTP 429)")
        if resp.status_code != 200:
            raise AdapterError(
                f"FRED returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        body = resp.json()
        if "observations" not in body:
            raise AdapterError(f"FRED unexpected response: {body}")
        return self._to_arrow(body["observations"])


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
def _ensure_protocol() -> None:
    adapter: MarketDataAdapter = FredAdapter(api_key="dummy")  # type: ignore[assignment]
    _ = adapter


_ensure_protocol()
