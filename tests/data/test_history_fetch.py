"""Tests for the history fetcher's pagination + incremental sync (no network).

A fake adapter replaces ``CCXTAdapter`` so no testnet connection is made. The
tests verify the 1000-bar-per-window pagination and the incremental
``sync_history`` resume-from-last-stored-ts behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from kairon.data import history_fetch
from kairon.data.io import OHLCV_SCHEMA
from kairon.data.symbols import CryptoVenue, crypto_perp

SYM = crypto_perp("ETH", "USDT", CryptoVenue.BYBIT)


class _FakeAdapter:
    """Records the (start, end) windows requested and synthesizes 1 bar/minute."""

    def __init__(self, *, venue: CryptoVenue, testnet: bool, fail_windows: int = 0) -> None:
        self.venue = venue
        self.testnet = testnet
        self.calls: list[tuple[datetime, datetime]] = []
        self._fail_windows = fail_windows
        self.closed = False

    async def afetch(self, symbol, timeframe, start, end) -> pa.Table:
        self.calls.append((start, end))
        if self._fail_windows > 0:
            self._fail_windows -= 1
            raise RuntimeError("simulated exchange hiccup")
        step = 60  # 1m
        ts: list[datetime] = []
        t = start.astimezone(UTC)
        end = end.astimezone(UTC)
        while t < end:
            ts.append(t)
            t = t + timedelta(seconds=step)
        n = len(ts)
        return pa.table(
            {"ts": ts, "open": [100.0] * n, "high": [100.0] * n,
             "low": [100.0] * n, "close": [100.0] * n, "volume": [1.0] * n},
            schema=OHLCV_SCHEMA,
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_fetch_history_paginates_at_chunk_bars(monkeypatch) -> None:
    # 2000 minutes of 1m data, chunk_bars=1000 -> exactly 2 windows.
    fake_cls = _FakeAdapter
    monkeypatch.setattr(history_fetch, "CCXTAdapter", fake_cls)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=2000)
    table = await history_fetch.fetch_history(
        SYM, "1m", start=start, end=end, testnet=True, chunk_bars=1000,
    )
    assert table.num_rows == 2000
    # The fetcher paginates the 2000-minute range into two 1000-bar windows and
    # concatenates them in order, deduplicated.
    ts = table.column("ts").to_pylist()
    assert ts == sorted(ts)
    assert len(set(ts)) == 2000
    assert ts[0] == start
    assert ts[-1] == end - timedelta(minutes=1)


@pytest.mark.asyncio
async def test_fetch_history_resumes_after_failed_window(monkeypatch) -> None:
    # The first window fails; the fetcher must advance and still return the rest.
    class _Flaky(_FakeAdapter):
        def __init__(self, *, venue, testnet) -> None:
            super().__init__(venue=venue, testnet=testnet, fail_windows=1)

    monkeypatch.setattr(history_fetch, "CCXTAdapter", _Flaky)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=2000)
    table = await history_fetch.fetch_history(
        SYM, "1m", start=start, end=end, testnet=True, chunk_bars=1000,
    )
    # First 1000-bar window dropped (failed), second 1000 returned.
    assert table.num_rows == 1000


@pytest.mark.asyncio
async def test_sync_history_incremental_resume(monkeypatch, tmp_path: Path) -> None:
    # Seed the store with 5 bars, then sync a 1-week horizon; the fetcher should
    # resume AFTER the last stored bar (not re-fetch the stored prefix).
    from kairon.data.history_store import merge_history
    from kairon.data.io import OHLCV_SCHEMA

    start = datetime(2026, 1, 1, tzinfo=UTC)
    seed = pa.table(
        {"ts": [start + timedelta(minutes=i) for i in range(5)],
         "open": [100.0] * 5, "high": [100.0] * 5, "low": [100.0] * 5,
         "close": [100.0] * 5, "volume": [1.0] * 5},
        schema=OHLCV_SCHEMA,
    )
    merge_history(seed, tmp_path, SYM.canonical, "1m")

    captured: list[datetime] = []

    class _ResumeAdapter(_FakeAdapter):
        async def afetch(self, symbol, timeframe, s, e) -> pa.Table:
            captured.append(s.astimezone(UTC))
            return await super().afetch(symbol, timeframe, s, e)

    monkeypatch.setattr(history_fetch, "CCXTAdapter", _ResumeAdapter)
    # Use a `now` only 10 minutes after the seed start so the horizon is tiny
    # and the fetch window is governed by the last stored ts, not the horizon.
    now = start + timedelta(minutes=10)
    table = await history_fetch.sync_history(
        SYM, "1m", root=tmp_path, weeks=1, testnet=True, chunk_bars=1000, now=now,
    )
    # The merged store has the 5 seed bars + the bars fetched after minute 5.
    assert table.num_rows >= 5
    # The fetcher started AFTER the last stored bar (minute 4 + 1 bar = minute 5).
    assert captured, "fetcher should have made at least one afetch call"
    assert captured[0] >= start + timedelta(minutes=5)


@pytest.mark.asyncio
async def test_sync_history_skips_when_up_to_date(monkeypatch, tmp_path: Path) -> None:
    # When the last stored bar is within the horizon and at/after `now - 1 bar`,
    # no fetch should occur and the existing table is returned unchanged.
    from kairon.data.history_store import merge_history
    from kairon.data.io import OHLCV_SCHEMA

    start = datetime(2026, 1, 1, tzinfo=UTC)
    seed = pa.table(
        {"ts": [start + timedelta(minutes=i) for i in range(5)],
         "open": [100.0] * 5, "high": [100.0] * 5, "low": [100.0] * 5,
         "close": [100.0] * 5, "volume": [1.0] * 5},
        schema=OHLCV_SCHEMA,
    )
    merge_history(seed, tmp_path, SYM.canonical, "1m")

    class _ExplodingAdapter(_FakeAdapter):
        async def afetch(self, *a, **k) -> pa.Table:
            raise AssertionError("should not fetch when already up to date")

    monkeypatch.setattr(history_fetch, "CCXTAdapter", _ExplodingAdapter)
    # `now` is one bar after the last stored bar -> start >= now -> no fetch.
    now = start + timedelta(minutes=5)
    table = await history_fetch.sync_history(
        SYM, "1m", root=tmp_path, weeks=1, testnet=True, chunk_bars=1000, now=now,
    )
    assert table.num_rows == 5
