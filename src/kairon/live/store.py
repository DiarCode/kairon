"""LiveStore: persist orders, fills, positions, and runtime state to sqlite.

Co-located in the same ``runs.db`` file as :class:`RunStore`. Tables are
namespaced with a ``live_`` prefix to avoid collisions. Uses the same
stdlib sqlite3 + threading.Lock pattern as RunStore.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timezone
from pathlib import Path

from kairon.live.broker.base import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.journal import TradeDecision, decision_to_row, row_to_decision

_LIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_orders (
    id          TEXT PRIMARY KEY,
    intent_id   TEXT NOT NULL,
    trace_id    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         REAL NOT NULL,
    order_type  TEXT NOT NULL,
    price       REAL,
    sl          REAL,
    tp          REAL,
    status      TEXT NOT NULL DEFAULT 'pending',
    broker_id   TEXT,
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_orders_symbol ON live_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_live_orders_status ON live_orders(status);
CREATE INDEX IF NOT EXISTS idx_live_orders_ts ON live_orders(ts);

CREATE TABLE IF NOT EXISTS live_fills (
    id          TEXT PRIMARY KEY,
    order_id    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    fee         REAL NOT NULL DEFAULT 0.0,
    fee_ccy     TEXT NOT NULL DEFAULT 'USDT',
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_fills_order_id ON live_fills(order_id);
CREATE INDEX IF NOT EXISTS idx_live_fills_ts ON live_fills(ts);

CREATE TABLE IF NOT EXISTS live_positions (
    symbol          TEXT PRIMARY KEY,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    avg_entry       REAL NOT NULL,
    unrealized_pnl  REAL NOT NULL DEFAULT 0.0,
    ts              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_heartbeat (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    mode            TEXT NOT NULL,
    equity          REAL,
    n_positions    INTEGER,
    last_signal_ts TEXT
);

CREATE TABLE IF NOT EXISTS live_runtime_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'info',
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_live_events_kind ON live_events(kind);
CREATE INDEX IF NOT EXISTS idx_live_events_ts ON live_events(ts);

CREATE TABLE IF NOT EXISTS live_closed_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_qty       REAL NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    realized_pnl   REAL NOT NULL,
    fee             REAL NOT NULL DEFAULT 0.0,
    entry_ts        TEXT NOT NULL,
    exit_ts         TEXT NOT NULL,
    duration_seconds REAL
);
CREATE INDEX IF NOT EXISTS idx_live_closed_trades_symbol ON live_closed_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_live_closed_trades_ts ON live_closed_trades(exit_ts);

CREATE TABLE IF NOT EXISTS live_decisions (
    order_id            TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    strategy_name       TEXT NOT NULL,
    direction           REAL NOT NULL,
    confidence          REAL NOT NULL,
    magnitude           REAL,
    volatility          REAL,
    horizon             TEXT,
    trend_score         REAL,
    momentum_score      REAL,
    structure_score     REAL,
    volume_score        REAL,
    indicators_json     TEXT NOT NULL DEFAULT '{}',
    risk_json           TEXT NOT NULL DEFAULT '{}',
    justifications_json TEXT NOT NULL DEFAULT '[]',
    outcome             TEXT,
    outcome_pnl         REAL,
    outcome_ts          TEXT
);
CREATE INDEX IF NOT EXISTS idx_live_decisions_symbol ON live_decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_live_decisions_ts ON live_decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_live_decisions_outcome ON live_decisions(outcome);
"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class LiveStore:
    """SQLite-backed store for live trading state.

    Co-located in the same database file as :class:`RunStore`. Tables are
    prefixed with ``live_`` to avoid namespace collisions. Uses the same
    stdlib sqlite3 + threading.Lock pattern.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.executescript(_LIVE_SCHEMA)

    # --- Orders -----------------------------------------------------------

    def write_order(self, order: Order) -> None:
        """Persist an order intent. Overwrites if ``order.id`` already exists."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO live_orders "
                "(id, intent_id, trace_id, symbol, side, qty, order_type, "
                "price, sl, tp, status, broker_id, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order.id,
                    order.intent_id,
                    order.trace_id,
                    order.symbol,
                    order.side.value,
                    order.qty,
                    order.order_type.value,
                    order.price,
                    order.sl,
                    order.tp,
                    order.status.value,
                    order.broker_id,
                    order.ts,
                ),
            )

    def update_order_status(self, order_id: str, status: OrderStatus, broker_id: str | None = None) -> None:
        """Update the status of an existing order."""
        with self._lock:
            if broker_id is not None:
                self._conn.execute(
                    "UPDATE live_orders SET status = ?, broker_id = ? WHERE id = ?",
                    (status.value, broker_id, order_id),
                )
            else:
                self._conn.execute(
                    "UPDATE live_orders SET status = ? WHERE id = ?",
                    (status.value, order_id),
                )

    def get_order(self, order_id: str) -> Order | None:
        """Return an order by ID, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, intent_id, trace_id, symbol, side, qty, order_type, "
                "price, sl, tp, status, broker_id, ts FROM live_orders WHERE id = ?",
                (order_id,),
            ).fetchone()
        if row is None:
            return None
        return Order(
            id=row[0],
            intent_id=row[1],
            trace_id=row[2],
            symbol=row[3],
            side=OrderSide(row[4]),
            qty=row[5],
            order_type=OrderType(row[6]),
            price=row[7],
            sl=row[8],
            tp=row[9],
            status=OrderStatus(row[10]),
            broker_id=row[11],
            ts=row[12],
        )

    # --- Fills -------------------------------------------------------------

    def write_fill(self, fill: Fill) -> None:
        """Persist a fill event."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO live_fills "
                "(id, order_id, symbol, side, qty, price, fee, fee_ccy, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fill.id,
                    fill.order_id,
                    fill.symbol,
                    fill.side.value,
                    fill.qty,
                    fill.price,
                    fill.fee,
                    fill.fee_ccy,
                    fill.ts,
                ),
            )

    def get_fills_for_order(self, order_id: str) -> list[Fill]:
        """Return all fills for an order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, order_id, symbol, side, qty, price, fee, fee_ccy, ts "
                "FROM live_fills WHERE order_id = ? ORDER BY ts",
                (order_id,),
            ).fetchall()
        return [
            Fill(
                id=r[0],
                order_id=r[1],
                symbol=r[2],
                side=OrderSide(r[3]),
                qty=r[4],
                price=r[5],
                fee=r[6],
                fee_ccy=r[7],
                ts=r[8],
            )
            for r in rows
        ]

    # --- Positions ---------------------------------------------------------

    def write_position(self, position: Position) -> None:
        """Upsert a position snapshot (overwrites previous)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO live_positions "
                "(symbol, side, qty, avg_entry, unrealized_pnl, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    position.symbol,
                    position.side.value,
                    position.qty,
                    position.avg_entry,
                    position.unrealized_pnl,
                    position.ts,
                ),
            )

    def get_positions(self) -> list[Position]:
        """Return all positions."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, side, qty, avg_entry, unrealized_pnl, ts FROM live_positions"
            ).fetchall()
        return [
            Position(
                symbol=r[0],
                side=OrderSide(r[1]),
                qty=r[2],
                avg_entry=r[3],
                unrealized_pnl=r[4],
                ts=r[5],
            )
            for r in rows
        ]

    def delete_position(self, symbol: str) -> None:
        """Remove a position (e.g. after close)."""
        with self._lock:
            self._conn.execute("DELETE FROM live_positions WHERE symbol = ?", (symbol,))

    # --- Closed trades (realized PnL) ----------------------------------------

    def write_closed_trade(
        self,
        symbol: str,
        side: str,
        entry_qty: float,
        entry_price: float,
        exit_price: float,
        realized_pnl: float,
        fee: float = 0.0,
        entry_ts: str = "",
        exit_ts: str = "",
        duration_seconds: float | None = None,
    ) -> None:
        """Record a closed trade with realized PnL."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO live_closed_trades "
                "(symbol, side, entry_qty, entry_price, exit_price, realized_pnl, fee, entry_ts, exit_ts, duration_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, side, entry_qty, entry_price, exit_price, realized_pnl, fee, entry_ts, exit_ts, duration_seconds),
            )

    def get_closed_trades(self, symbol: str | None = None) -> list[dict]:
        """Return closed trades ordered by exit time (oldest first)."""
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    "SELECT id, symbol, side, entry_qty, entry_price, exit_price, "
                    "realized_pnl, fee, entry_ts, exit_ts, duration_seconds "
                    "FROM live_closed_trades WHERE symbol = ? ORDER BY exit_ts ASC",
                    (symbol,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, symbol, side, entry_qty, entry_price, exit_price, "
                    "realized_pnl, fee, entry_ts, exit_ts, duration_seconds "
                    "FROM live_closed_trades ORDER BY exit_ts ASC"
                ).fetchall()
        return [
            {
                "id": r[0],
                "symbol": r[1],
                "side": r[2],
                "entry_qty": r[3],
                "entry_price": r[4],
                "exit_price": r[5],
                "realized_pnl": r[6],
                "fee": r[7],
                "entry_ts": r[8],
                "exit_ts": r[9],
                "duration_seconds": r[10],
            }
            for r in rows
        ]

    # --- Heartbeat ---------------------------------------------------------

    def write_heartbeat(
        self,
        mode: str,
        equity: float | None = None,
        n_positions: int | None = None,
        last_signal_ts: str | None = None,
    ) -> None:
        """Write a heartbeat row."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO live_heartbeat (ts, mode, equity, n_positions, last_signal_ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (_utc_now_iso(), mode, equity, n_positions, last_signal_ts),
            )

    # --- Runtime state (kill switch, mode, etc.) --------------------------

    def halt(self, reason: str) -> None:
        """Set the halt flag. The TradingLoop reads this at the top of every tick."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO live_runtime_state (key, value, updated_at) "
                "VALUES ('halted', ?, ?)",
                (reason, _utc_now_iso()),
            )

    def is_halted(self) -> bool:
        """Check if the halt flag is set."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM live_runtime_state WHERE key = 'halted'"
            ).fetchone()
        return row is not None

    def unhalt(self) -> None:
        """Clear the halt flag."""
        with self._lock:
            self._conn.execute("DELETE FROM live_runtime_state WHERE key = 'halted'")

    def get_runtime_state(self, key: str) -> str | None:
        """Get a runtime state value."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM live_runtime_state WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    # --- Events (audit trail) ---------------------------------------------

    def write_event(self, kind: str, severity: str = "info", payload_json: str = "{}") -> None:
        """Persist an audit event."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO live_events (ts, kind, severity, payload_json) VALUES (?, ?, ?, ?)",
                (_utc_now_iso(), kind, severity, payload_json),
            )

    # --- Dashboard queries --------------------------------------------------

    def get_recent_heartbeat(self) -> dict[str, object] | None:
        """Return the most recent heartbeat row, or None if no heartbeats exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT ts, mode, equity, n_positions, last_signal_ts "
                "FROM live_heartbeat ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "ts": row[0],
            "mode": row[1],
            "equity": row[2],
            "n_positions": row[3],
            "last_signal_ts": row[4],
        }

    def get_heartbeat_history(self, limit: int = 120) -> list[dict[str, object]]:
        """Return the last ``limit`` heartbeat rows, oldest-first.

        Feeds the equity sparkline on the live dashboard. Skips rows whose
        ``equity`` is NULL so the chart only plots real equity samples.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, equity FROM live_heartbeat "
                "WHERE equity IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        rows = list(reversed(rows))
        return [{"ts": r[0], "equity": r[1]} for r in rows]

    def get_recent_orders(self, limit: int = 50) -> list[dict[str, object]]:
        """Return recent orders ordered by timestamp descending."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, symbol, side, qty, order_type, price, status, broker_id, ts "
                "FROM live_orders ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "symbol": r[1],
                "side": r[2],
                "qty": r[3],
                "order_type": r[4],
                "price": r[5],
                "status": r[6],
                "broker_id": r[7],
                "ts": r[8],
            }
            for r in rows
        ]

    def get_recent_events(self, limit: int = 50) -> list[dict[str, object]]:
        """Return recent audit events ordered by ID descending."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, kind, severity, payload_json "
                "FROM live_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "ts": r[0],
                "kind": r[1],
                "severity": r[2],
                "payload_json": r[3],
            }
            for r in rows
        ]

    # --- Analytics queries -------------------------------------------------

    def get_all_fills(self) -> list[dict]:
        """Return all fills ordered by timestamp (oldest first)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, order_id, symbol, side, qty, price, fee, fee_ccy, ts "
                "FROM live_fills ORDER BY ts ASC"
            ).fetchall()
        return [
            {
                "id": r[0],
                "order_id": r[1],
                "symbol": r[2],
                "side": r[3],
                "qty": r[4],
                "price": r[5],
                "fee": r[6],
                "fee_ccy": r[7],
                "ts": r[8],
            }
            for r in rows
        ]

    def get_all_orders(self) -> list[dict]:
        """Return all orders ordered by timestamp (oldest first)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, intent_id, trace_id, symbol, side, qty, order_type, "
                "price, sl, tp, status, broker_id, ts "
                "FROM live_orders ORDER BY ts ASC"
            ).fetchall()
        return [
            {
                "id": r[0],
                "intent_id": r[1],
                "trace_id": r[2],
                "symbol": r[3],
                "side": r[4],
                "qty": r[5],
                "order_type": r[6],
                "price": r[7],
                "sl": r[8],
                "tp": r[9],
                "status": r[10],
                "broker_id": r[11],
                "ts": r[12],
            }
            for r in rows
        ]

    def get_all_heartbeats(self) -> list[dict]:
        """Return all heartbeat rows ordered by id (oldest first)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, mode, equity, n_positions, last_signal_ts "
                "FROM live_heartbeat ORDER BY id ASC"
            ).fetchall()
        return [
            {
                "id": r[0],
                "ts": r[1],
                "mode": r[2],
                "equity": r[3],
                "n_positions": r[4],
                "last_signal_ts": r[5],
            }
            for r in rows
        ]

    def get_all_events(self) -> list[dict]:
        """Return all events ordered by id (oldest first)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, kind, severity, payload_json "
                "FROM live_events ORDER BY id ASC"
            ).fetchall()
        return [
            {
                "id": r[0],
                "ts": r[1],
                "kind": r[2],
                "severity": r[3],
                "payload_json": r[4],
            }
            for r in rows
        ]

    # --- Decisions (trade journal) ------------------------------------------

    def write_decision(self, decision: TradeDecision) -> None:
        """Persist a trade decision with full indicator snapshot."""
        row = decision_to_row(decision)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO live_decisions "
                "(order_id, symbol, timestamp, strategy_name, direction, confidence, "
                "magnitude, volatility, horizon, trend_score, momentum_score, "
                "structure_score, volume_score, indicators_json, risk_json, "
                "justifications_json, outcome, outcome_pnl, outcome_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["order_id"],
                    row["symbol"],
                    row["timestamp"],
                    row["strategy_name"],
                    row["direction"],
                    row["confidence"],
                    row["magnitude"],
                    row["volatility"],
                    row["horizon"],
                    row["trend_score"],
                    row["momentum_score"],
                    row["structure_score"],
                    row["volume_score"],
                    row["indicators_json"],
                    row["risk_json"],
                    row["justifications_json"],
                    row["outcome"],
                    row["outcome_pnl"],
                    row["outcome_ts"],
                ),
            )

    def update_decision_outcome(
        self,
        order_id: str,
        outcome: str,
        pnl: float | None = None,
        ts: str | None = None,
    ) -> None:
        """Update the outcome of a trade decision after the trade closes."""
        with self._lock:
            self._conn.execute(
                "UPDATE live_decisions SET outcome = ?, outcome_pnl = ?, outcome_ts = ? "
                "WHERE order_id = ?",
                (outcome, pnl, ts or _utc_now_iso(), order_id),
            )

    def get_decisions(
        self,
        symbol: str | None = None,
        outcome: str | None = None,
        limit: int = 200,
    ) -> list[TradeDecision]:
        """Return trade decisions, optionally filtered by symbol and/or outcome."""
        with self._lock:
            query = (
                "SELECT order_id, symbol, timestamp, strategy_name, direction, confidence, "
                "magnitude, volatility, horizon, trend_score, momentum_score, "
                "structure_score, volume_score, indicators_json, risk_json, "
                "justifications_json, outcome, outcome_pnl, outcome_ts "
                "FROM live_decisions"
            )
            conditions: list[str] = []
            params: list[object] = []
            if symbol is not None:
                conditions.append("symbol = ?")
                params.append(symbol)
            if outcome is not None:
                conditions.append("outcome = ?")
                params.append(outcome)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(query, tuple(params)).fetchall()

        columns = [
            "order_id", "symbol", "timestamp", "strategy_name", "direction",
            "confidence", "magnitude", "volatility", "horizon", "trend_score",
            "momentum_score", "structure_score", "volume_score", "indicators_json",
            "risk_json", "justifications_json", "outcome", "outcome_pnl", "outcome_ts",
        ]
        return [row_to_decision(dict(zip(columns, r, strict=True))) for r in rows]

    # --- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying sqlite connection. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()


__all__ = ["LiveStore"]
