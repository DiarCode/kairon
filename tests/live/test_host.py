"""Tests for the in-process SessionHost (start/stop/halt/snapshot/shutdown)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest

from kairon.live.config import LiveConfig
from kairon.live.host import SessionError, SessionHost

# ---------------------------------------------------------------------------
# Fakes — replace the heavy network/loop deps the host wires up.
# ---------------------------------------------------------------------------


class _FakeLoop:
    """Stand-in for CooledTradingLoop: tracks running state, no ticking."""

    def __init__(self, **_kwargs: object) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def finalize_open_positions(self) -> None:
        return None


class _FakeFeed:
    """Stand-in for CcxtCandleFeed: run blocks until aclose() sets the event."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        await self._stop.wait()

    async def aclose(self) -> None:
        self._stop.set()


def _settings_stub() -> SimpleNamespace:
    return SimpleNamespace(bybit_api_key="", bybit_api_secret="", bybit_tld="com")


def _paper_config(**overrides: object) -> LiveConfig:
    defaults: dict[str, object] = {
        "symbols": ("BTC-USDT-PERP",),
        "timeframe": "1m",
        "cadence_seconds": 10,
        "max_daily_loss_pct": 0.03,
        "max_open_positions": 5,
        "warmup_bars": 2,
        "reconcile_interval_seconds": 30,
        "reconcile_grace_seconds": 120,
        "dry_run": True,
        "bybit_testnet": False,
    }
    defaults.update(overrides)
    return LiveConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def patched_host(monkeypatch: pytest.MonkeyPatch, tmp_path) -> SessionHost:
    """A SessionHost whose feed + loop are fakes (no network, no ticking)."""
    monkeypatch.setattr("kairon.live.feed.CcxtCandleFeed", _FakeFeed)
    monkeypatch.setattr("kairon.live.cooldown.CooledTradingLoop", _FakeLoop)
    return SessionHost(data_dir=tmp_path)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestSessionHostLifecycle:
    async def test_start_returns_session_metadata(self, patched_host: SessionHost) -> None:
        result = await patched_host.start_session(_paper_config(), _settings_stub())
        assert result["mode"] == "Paper"
        assert result["initial_equity"] == 10_000.0
        assert result["session_id"]
        assert patched_host.is_running is False  # FakeLoop.start task hasn't run yet
        await asyncio.sleep(0)  # let the start task set is_running
        assert patched_host.is_running is True
        await patched_host.stop_session()

    async def test_double_start_guard(self, patched_host: SessionHost) -> None:
        await patched_host.start_session(_paper_config(), _settings_stub())
        await asyncio.sleep(0)  # FakeLoop.start sets is_running
        with pytest.raises(SessionError, match="already running"):
            await patched_host.start_session(_paper_config(), _settings_stub())
        await patched_host.stop_session()

    async def test_stop_is_idempotent(self, patched_host: SessionHost) -> None:
        await patched_host.start_session(_paper_config(), _settings_stub())
        await asyncio.sleep(0)
        stopped = await patched_host.stop_session()
        assert stopped["status"] == "stopped"
        # Second stop with no active session.
        again = await patched_host.stop_session()
        assert again["status"] == "not_running"
        assert patched_host.active is None

    async def test_shutdown_all_stops_sessions(self, patched_host: SessionHost) -> None:
        await patched_host.start_session(_paper_config(), _settings_stub())
        await asyncio.sleep(0)
        assert patched_host.active is not None
        await patched_host.shutdown_all()
        assert patched_host.active is None
        assert patched_host.is_running is False


# ---------------------------------------------------------------------------
# Halt / unhalt
# ---------------------------------------------------------------------------


class TestSessionHostHalt:
    async def test_halt_then_unhalt(self, patched_host: SessionHost) -> None:
        await patched_host.start_session(_paper_config(), _settings_stub())
        await asyncio.sleep(0)
        halted = await patched_host.halt()
        assert halted["status"] == "halted"
        snap = patched_host.snapshot()
        assert snap["halted"] is True
        assert snap["status"] == "halted"
        resumed = await patched_host.unhalt()
        assert resumed["status"] == "running"
        assert patched_host.snapshot()["halted"] is False
        await patched_host.stop_session()

    async def test_halt_without_session_raises(self, patched_host: SessionHost) -> None:
        with pytest.raises(SessionError, match="No active session"):
            await patched_host.halt()


# ---------------------------------------------------------------------------
# Snapshot shape
# ---------------------------------------------------------------------------


class TestSessionHostSnapshot:
    STOPPED_KEYS: ClassVar[frozenset[str]] = frozenset({
        "status", "is_running", "session_id", "mode", "equity", "initial_equity",
        "session_pnl", "realized_pnl", "unrealized_pnl", "n_positions", "halted",
        "halt_reason", "last_heartbeat", "last_signal_ts", "symbols", "positions",
        "decisions", "closed_trades", "orders", "events", "win_rate", "equity_series",
    })

    def test_snapshot_when_stopped(self, patched_host: SessionHost) -> None:
        snap = patched_host.snapshot()
        assert frozenset(snap.keys()) == self.STOPPED_KEYS
        assert snap["status"] == "stopped"
        assert snap["is_running"] is False
        assert snap["session_id"] is None
        assert snap["positions"] == []
        assert snap["equity_series"] == []

    async def test_snapshot_when_running(self, patched_host: SessionHost) -> None:
        await patched_host.start_session(_paper_config(), _settings_stub())
        await asyncio.sleep(0)
        try:
            snap = patched_host.snapshot()
            assert snap["status"] == "running"
            assert snap["is_running"] is True
            assert snap["mode"] == "Paper"
            assert snap["session_id"]
            assert snap["n_positions"] == 0
            assert snap["initial_equity"] == 10_000.0
        finally:
            await patched_host.stop_session()
