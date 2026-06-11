"""Tests for parquet IO round-trips and content hashes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from kairon.data.io import (
    OHLCV_SCHEMA,
    DataPaths,
    read_ohlcv,
    table_hash,
    write_ohlcv,
)
from kairon.data.symbols import CryptoVenue, crypto_spot


@pytest.fixture
def sample_table() -> pa.Table:
    ts = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        datetime(2024, 2, 1, 12, 0, tzinfo=UTC),
    ]
    return pa.table(
        {
            "ts": ts,
            "open": [100.0, 101.0, 110.0],
            "high": [101.0, 102.0, 112.0],
            "low": [99.0, 100.0, 108.0],
            "close": [100.5, 101.5, 111.5],
            "volume": [10.0, 20.0, 30.0],
        },
        schema=OHLCV_SCHEMA,
    )


def test_write_creates_per_month_partitions(
    sample_table: pa.Table, tmp_path: Path
) -> None:
    paths = DataPaths(root=tmp_path)
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    written = write_ohlcv(
        sample_table, symbol=sym, venue="ccxt", timeframe="5m", paths=paths
    )
    assert len(written) == 2
    rel = [p.relative_to(tmp_path).as_posix() for p in written]
    assert any("2024/01.parquet" in r for r in rel)
    assert any("2024/02.parquet" in r for r in rel)


def test_write_appends_to_existing(sample_table: pa.Table, tmp_path: Path) -> None:
    paths = DataPaths(root=tmp_path)
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    write_ohlcv(sample_table, symbol=sym, venue="ccxt", timeframe="5m", paths=paths)
    write_ohlcv(sample_table, symbol=sym, venue="ccxt", timeframe="5m", paths=paths)
    rt = read_ohlcv(symbol=sym, venue="ccxt", timeframe="5m", paths=paths)
    assert rt.num_rows == 6


def test_read_empty(tmp_path: Path) -> None:
    paths = DataPaths(root=tmp_path)
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    rt = read_ohlcv(symbol=sym, venue="ccxt", timeframe="5m", paths=paths)
    assert rt.num_rows == 0
    assert rt.schema == OHLCV_SCHEMA


def test_table_hash_is_stable(sample_table: pa.Table) -> None:
    h1 = table_hash(sample_table)
    h2 = table_hash(sample_table)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_table_hash_changes_with_data(sample_table: pa.Table) -> None:
    h1 = table_hash(sample_table)
    alt = pa.table(
        {
            "ts": sample_table.column("ts").to_pylist(),
            "open": [200.0, 201.0, 210.0],
            "high": [201.0, 202.0, 212.0],
            "low": [199.0, 200.0, 208.0],
            "close": [200.5, 201.5, 211.5],
            "volume": [10.0, 20.0, 30.0],
        },
        schema=OHLCV_SCHEMA,
    )
    assert table_hash(alt) != h1


def test_write_rejects_wrong_schema() -> None:
    bad = pa.table({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="schema mismatch"):
        write_ohlcv(bad, symbol=crypto_spot("BTC", "USDT", CryptoVenue.BINANCE), venue="x", timeframe="1m")  # type: ignore[arg-type]


def test_round_trip_preserves_schema(sample_table: pa.Table, tmp_path: Path) -> None:
    paths = DataPaths(root=tmp_path)
    sym = crypto_spot("BTC", "USDT", CryptoVenue.BINANCE)
    write_ohlcv(sample_table, symbol=sym, venue="ccxt", timeframe="5m", paths=paths)
    rt = read_ohlcv(symbol=sym, venue="ccxt", timeframe="5m", paths=paths)
    assert rt.schema == OHLCV_SCHEMA
    assert sorted(rt.column("ts").to_pylist()) == sorted(
        sample_table.column("ts").to_pylist()
    )
