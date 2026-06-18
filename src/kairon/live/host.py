"""In-process host for live trading sessions.

The FastAPI dashboard server *is* the session host: ``start_session`` wires
broker + feed + Guardian + Reconciler + strategy + :class:`TradingLoop`
together and runs them as asyncio tasks inside the server process; the SSE
snapshot stream and the control endpoints read live state from the active
session's :class:`LiveStore`. This mirrors the wiring in
``scripts/run_testnet_symbol.py`` but as a library so the web layer can drive
it without spawning a subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kairon.config import KaironSettings
from kairon.live.config import LiveConfig
from kairon.live.store import LiveStore

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Raised by :class:`SessionHost` for actionable start/stop/control failures."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class Session:
    """A hosted live trading session."""

    id: str
    config: LiveConfig
    store: LiveStore
    broker: Any
    feed: Any
    loop: Any
    started_at: float
    initial_equity: float
    _tasks: set[asyncio.Task[None]] = field(default_factory=set)


def _mode_label(config: LiveConfig) -> str:
    if config.dry_run:
        return "Paper"
    return "MAINNET" if not config.bybit_testnet else "Bybit Testnet"


def _win_rate(closed: list[dict[str, Any]]) -> float:
    decided = [c for c in closed if c.get("realized_pnl") is not None]
    if not decided:
        return 0.0
    wins = sum(1 for c in decided if float(c["realized_pnl"] or 0.0) > 0)
    return wins / len(decided)


def _pos_dict(p: Any) -> dict[str, Any]:
    return {
        "symbol": p.symbol,
        "side": p.side.value,
        "qty": p.qty,
        "avg_entry": p.avg_entry,
        "unrealized_pnl": float(p.unrealized_pnl or 0.0),
        "ts": p.ts,
    }


def _decision_dict(d: Any) -> dict[str, Any]:
    return {
        "order_id": d.order_id,
        "symbol": d.symbol,
        "timestamp": d.timestamp,
        "direction": d.direction,
        "confidence": d.confidence,
        "magnitude": d.magnitude,
        "horizon": d.horizon,
        "trend_score": d.trend_score,
        "momentum_score": d.momentum_score,
        "structure_score": d.structure_score,
        "volume_score": d.volume_score,
        "justifications": list(d.justifications)[:4],
        "outcome": d.outcome,
        "outcome_pnl": d.outcome_pnl,
    }


def _event_dict(e: dict[str, Any]) -> dict[str, Any]:
    payload = e.get("payload_json")
    parsed: Any = payload
    if isinstance(payload, str):
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(payload)
    return {"ts": e["ts"], "kind": e["kind"], "severity": e["severity"], "payload": parsed}


def _to_symbol_tuple(symbols: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize dashboard symbol strings to canonical form (strip blanks)."""
    return tuple(s for s in symbols if s)


class SessionHost:
    """Owns hosted trading sessions; lives on ``app.state.session_host``.

    The UI drives a single active session at a time; ``start_session`` refuses
    a second concurrent start. A registry keyed by ``session_id`` keeps the
    door open for multi-session later.
    """

    def __init__(self, *, data_dir: Path = Path("data")) -> None:
        self._data_dir = Path(data_dir)
        self._sessions: dict[str, Session] = {}
        self._active_id: str | None = None
        self._lock = asyncio.Lock()

    # --- accessors ---------------------------------------------------------
    @property
    def active(self) -> Session | None:
        return self._sessions.get(self._active_id) if self._active_id else None

    @property
    def is_running(self) -> bool:
        loop_obj = self.active.loop if self.active else None
        return bool(loop_obj and loop_obj.is_running)

    # --- lifecycle ---------------------------------------------------------
    async def start_session(
        self,
        config: LiveConfig,
        settings: KaironSettings,
        *,
        cooldown_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Build broker+store+feed+guardian+reconciler+strategy+loop, preflight, spawn."""
        async with self._lock:
            if self.active and self.active.loop.is_running:
                raise SessionError("already_running", "A session is already running. Stop it first.")

            # Lazy imports — heavy broker/feed/strategy deps may be optional.
            from kairon.data.symbols import CryptoVenue, crypto_perp
            from kairon.live.broker import BybitBroker, PaperBroker
            from kairon.live.cooldown import (
                CooldownBrokerWrapper,
                CooledTradingLoop,
            )
            from kairon.live.feed import CcxtCandleFeed, CcxtCandleFeedConfig
            from kairon.live.guardian import Guardian
            from kairon.live.reconciler import Reconciler
            from kairon.live.strategy import ComprehensiveStrategy

            symbols = _to_symbol_tuple(config.symbols)
            session_id = uuid.uuid4().hex[:12]
            self._data_dir.mkdir(parents=True, exist_ok=True)
            db_path = self._data_dir / f"live_{session_id}.db"
            store = LiveStore(db_path)
            store.unhalt()

            # Broker selection driven by config.dry_run (mode toggle in UI).
            if config.dry_run:
                broker: Any = PaperBroker(initial_balance=10_000.0, currency="USDT")
                initial_equity = 10_000.0
                store.write_heartbeat(mode="paper", equity=initial_equity, n_positions=0)
            else:
                broker = BybitBroker(
                    api_key=settings.bybit_api_key,
                    api_secret=settings.bybit_api_secret,
                    testnet=config.bybit_testnet,
                    tld=config.bybit_tld,
                )
                try:
                    initial_equity = await self._preflight(
                        broker, symbols, store, mode="testnet" if config.bybit_testnet else "live"
                    )
                except Exception as e:
                    with contextlib.suppress(Exception):
                        store.close()
                    raise SessionError("preflight_failed", f"Broker preflight failed: {e}") from e

            wrapped = CooldownBrokerWrapper(broker, cooldown_seconds=cooldown_seconds)

            # Feed — multi-symbol via CcxtCandleFeed (already a library class).
            venue = CryptoVenue.BYBIT
            feed_symbols = [crypto_perp(*s.split("-")[:2], venue) for s in symbols]
            feed_config = CcxtCandleFeedConfig(
                venue=venue,
                timeframe=config.timeframe,
                testnet=config.bybit_testnet or config.dry_run,
                api_key=settings.bybit_api_key if not config.dry_run else "",
                api_secret=settings.bybit_api_secret if not config.dry_run else "",
            )
            feed = CcxtCandleFeed(feed_symbols, config=feed_config)

            guardian = Guardian(
                max_position_equity_fraction=0.20,
                max_total_leverage=1.0,
                max_open_positions=config.max_open_positions,
                max_daily_loss_pct=config.max_daily_loss_pct,
                store=store,
            )
            reconciler = Reconciler(
                drift_tolerance_pct=0.05,
                grace_seconds=config.reconcile_grace_seconds,
                reconcile_interval_seconds=config.reconcile_interval_seconds,
                symbols=symbols,
                store=store,
                broker=wrapped,
            )
            strategy = ComprehensiveStrategy()
            loop = CooledTradingLoop(
                config=config,
                broker=wrapped,
                strategy=strategy,
                guardian=guardian,
                reconciler=reconciler,
                store=store,
                feed=feed,
            )

            sess = Session(
                id=session_id,
                config=config,
                store=store,
                broker=broker,
                feed=feed,
                loop=loop,
                started_at=time.time(),
                initial_equity=initial_equity,
            )

            feed_task = asyncio.create_task(feed.run(), name=f"feed-{session_id}")
            loop_task = asyncio.create_task(loop.start(), name=f"loop-{session_id}")
            sess._tasks = {feed_task, loop_task}
            for t in sess._tasks:
                t.add_done_callback(sess._tasks.discard)

            self._sessions[session_id] = sess
            self._active_id = session_id
            logger.info("Started session %s (%s) symbols=%s", session_id, _mode_label(config), list(symbols))
            return {
                "session_id": session_id,
                "mode": _mode_label(config),
                "initial_equity": initial_equity,
                "symbols": list(symbols),
            }

    async def _preflight(
        self, broker: Any, symbols: tuple[str, ...], store: LiveStore, *, mode: str
    ) -> float:
        """Health + balance + leverage. Returns initial USDT equity."""
        health = await broker.check_health()
        if not health.get("ok"):
            raise RuntimeError(f"Health check failed: {health.get('errors')}")
        balances = await broker.get_balances()
        usdt = next((b for b in balances if b.currency == "USDT"), None)
        if usdt is None or usdt.total <= 0:
            raise RuntimeError("No USDT balance found on account")
        for sym in symbols:
            with contextlib.suppress(Exception):
                await broker.set_leverage(sym, 3)
        store.write_heartbeat(mode=mode, equity=usdt.total, n_positions=0)
        return float(usdt.total)

    async def stop_session(self, session_id: str | None = None) -> dict[str, Any]:
        """stop → finalize_open_positions → flatten → aclose. Idempotent."""
        async with self._lock:
            sess = self._resolve(session_id)
            if sess is None:
                return {"status": "not_running"}
            try:
                await sess.loop.stop()
                with contextlib.suppress(Exception):
                    await sess.loop.finalize_open_positions()
                await self._flatten(sess)
                with contextlib.suppress(Exception):
                    await sess.feed.aclose()
                aclose = getattr(sess.broker, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()
            finally:
                with contextlib.suppress(Exception):
                    sess.store.write_heartbeat(mode="shutdown", equity=0.0, n_positions=0)
                with contextlib.suppress(Exception):
                    sess.store.close()
                self._sessions.pop(sess.id, None)
                if self._active_id == sess.id:
                    self._active_id = None
            logger.info("Stopped session %s", sess.id)
            return {"status": "stopped", "session_id": sess.id}

    async def _flatten(self, sess: Session) -> None:
        """Close any open position per symbol (reduce-only). Best-effort."""
        for sym in _to_symbol_tuple(sess.config.symbols):
            close = getattr(sess.broker, "close_position", None)
            if close is None:
                break
            with contextlib.suppress(Exception):
                await close(sym)

    async def halt(self, reason: str = "manual_halt_via_dashboard") -> dict[str, Any]:
        sess = self.active
        if sess is None:
            raise SessionError("not_running", "No active session to halt.")
        sess.store.halt(reason=reason)
        sess.store.write_event(kind="halt", severity="critical", payload_json=json.dumps({"reason": reason}))
        return {"status": "halted", "reason": reason}

    async def unhalt(self) -> dict[str, Any]:
        sess = self.active
        if sess is None:
            raise SessionError("not_running", "No active session.")
        sess.store.unhalt()
        sess.store.write_event(kind="unhalt", severity="info", payload_json='{"reason":"manual_resume"}')
        return {"status": "running"}

    async def shutdown_all(self) -> None:
        """FastAPI shutdown: stop every hosted session."""
        for sid in list(self._sessions):
            with contextlib.suppress(Exception):
                await self.stop_session(sid)

    # --- read side ---------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Synchronous snapshot read from the active session's store (SSE source)."""
        sess = self.active
        if sess is None:
            return {
                "status": "stopped",
                "is_running": False,
                "session_id": None,
                "mode": None,
                "equity": None,
                "initial_equity": 0.0,
                "session_pnl": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "n_positions": 0,
                "halted": False,
                "halt_reason": None,
                "last_heartbeat": None,
                "last_signal_ts": None,
                "symbols": [],
                "positions": [],
                "decisions": [],
                "closed_trades": [],
                "orders": [],
                "events": [],
                "win_rate": 0.0,
                "equity_series": [],
            }
        store = sess.store
        hb = store.get_recent_heartbeat()
        positions = store.get_positions()
        decisions = store.get_decisions(limit=50)
        closed = store.get_closed_trades()
        orders = store.get_recent_orders(limit=50)
        events = store.get_recent_events(limit=50)
        halted = store.is_halted()
        halt_reason = store.get_runtime_state("halted") if halted else None
        realized = sum(float(c.get("realized_pnl") or 0.0) for c in closed)
        unrealized = sum(float(p.unrealized_pnl or 0.0) for p in positions)
        return {
            "status": "halted" if halted else ("running" if sess.loop.is_running else "stopped"),
            "is_running": bool(sess.loop.is_running),
            "session_id": sess.id,
            "mode": _mode_label(sess.config),
            "equity": hb["equity"] if hb else sess.initial_equity,
            "initial_equity": sess.initial_equity,
            "session_pnl": round(realized + unrealized, 8),
            "realized_pnl": round(realized, 8),
            "unrealized_pnl": round(unrealized, 8),
            "n_positions": len(positions),
            "halted": halted,
            "halt_reason": halt_reason,
            "last_heartbeat": hb["ts"] if hb else None,
            "last_signal_ts": hb["last_signal_ts"] if hb else None,
            "symbols": list(_to_symbol_tuple(sess.config.symbols)),
            "positions": [_pos_dict(p) for p in positions],
            "decisions": [_decision_dict(d) for d in decisions],
            "closed_trades": closed,
            "orders": orders,
            "events": [_event_dict(e) for e in events],
            "win_rate": _win_rate(closed),
            "equity_series": store.get_heartbeat_history(limit=120),
        }

    def _resolve(self, session_id: str | None) -> Session | None:
        if session_id is None:
            return self.active
        return self._sessions.get(session_id)


__all__ = ["Session", "SessionError", "SessionHost"]
