"""Tests for the ingestion orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pytest

from kairon.data.adapters import AdapterError
from kairon.data.diagnostics import Severity
from kairon.data.ingestion import IngestionResult, ingest
from kairon.data.io import OHLCV_SCHEMA
from kairon.data.symbols import CryptoVenue, crypto_spot


class FakeAdapter:
    name = "fake"

    def __init__(self, table: pa.Table | None = None, raise_exc: bool = False) -> None:
        self.table = table or pa.table(
            {
                "ts": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
                ],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [10.0, 20.0],
            },
            schema=OHLCV_SCHEMA,
        )
        self.raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    def fetch(
        self,
        symbol: object,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pa.Table:
        self.calls.append(
            {
                "symbol": str(symbol),
                "timeframe": timeframe,
                "start": start,
                "end": end,
            }
        )
        if self.raise_exc:
            raise AdapterError("simulated")
        return self.table


def test_ingest_happy_path(tmp_path_factory: pytest.TempPathFactory) -> None:
    from kairon.data.io import DataPaths

    paths = DataPaths(root=tmp_path_factory.mktemp("data"))
    adapter = FakeAdapter()
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    res = ingest(
        adapter,
        symbol=sym,
        timeframe="5m",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, 1, tzinfo=UTC),
        paths=paths,
    )
    assert isinstance(res, IngestionResult)
    assert res.symbol == "BTC-USDT"
    assert res.n_rows == 2
    assert len(res.written_paths) >= 1
    assert res.table_hash


def test_ingest_rejects_inverted_range() -> None:
    adapter = FakeAdapter()
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    with pytest.raises(ValueError, match="must be <"):
        ingest(
            adapter,
            symbol=sym,
            timeframe="5m",
            start=datetime(2024, 1, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_ingest_wraps_unexpected_adapter_error() -> None:
    class Boom:
        name = "boom"

        def fetch(
            self,
            symbol: object,
            timeframe: str,
            start: datetime,
            end: datetime,
        ) -> pa.Table:
            raise RuntimeError("nope")

    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    with pytest.raises(AdapterError, match="adapter.fetch failed"):
        ingest(
            Boom(),  # type: ignore[arg-type]
            symbol=sym,
            timeframe="5m",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 1, tzinfo=UTC),
        )


def test_ingest_raises_on_quality_error() -> None:
    # Inject a negative volume -> diagnostic error
    bad = pa.table(
        {
            "ts": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [-1.0],
        },
        schema=OHLCV_SCHEMA,
    )
    adapter = FakeAdapter(table=bad)
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    with pytest.raises(ValueError, match="data quality errors"):
        ingest(
            adapter,
            symbol=sym,
            timeframe="5m",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 1, tzinfo=UTC),
        )


def test_ingest_does_not_raise_when_raise_on_error_false(tmp_path_factory: pytest.TempPathFactory) -> None:
    from kairon.data.io import DataPaths

    paths = DataPaths(root=tmp_path_factory.mktemp("data"))
    bad = pa.table(
        {
            "ts": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [-1.0],
        },
        schema=OHLCV_SCHEMA,
    )
    adapter = FakeAdapter(table=bad)
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    res = ingest(
        adapter,
        symbol=sym,
        timeframe="5m",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, 1, tzinfo=UTC),
        paths=paths,
        raise_on_error=False,
    )
    assert any(r.severity == Severity.ERROR for r in res.report.results)
