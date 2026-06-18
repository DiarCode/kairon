"""Reconciler: diffs local position state against broker truth.

The :class:`Reconciler` compares the local ``live_positions`` table against
the broker's authoritative positions every ``reconcile_interval_seconds``.
If drift exceeds tolerance, it emits a CRITICAL alert through the
:class:`AlertEngine`. It is also registered as an :class:`AlertEngine.Rule`
for fact-based evaluation.

On restart, the reconciler scans ``live_orders`` for orphan intents
(status=pending with no broker_id) and marks them ``orphan``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from kairon.live.alerts import Alert, Rule, Severity
from kairon.live.broker.base import OrderStatus, Position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fact types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftFact:
    """A position drift event: local and broker positions disagree."""

    symbol: str
    local_qty: float
    broker_qty: float
    drift_pct: float  # abs(local_qty - broker_qty) / max(local_qty, broker_qty, 1e-9)


@dataclass(frozen=True)
class OrphanFact:
    """An orphan order intent: pending with no broker acknowledgment."""

    order_id: str
    symbol: str
    age_seconds: float


# ---------------------------------------------------------------------------
# Broker protocol (minimal, for Reconciler's needs)
# ---------------------------------------------------------------------------


class _BrokerLike(Protocol):
    """Minimal broker interface the Reconciler needs."""

    async def get_positions(self, symbol: str | None = None) -> list[Position]: ...


class _StoreLike(Protocol):
    """Minimal store interface the Reconciler needs."""

    def get_positions(self) -> list[Position]: ...
    def write_position(self, position: Position) -> None: ...
    def delete_position(self, symbol: str) -> None: ...
    def get_order(self, order_id: str) -> Any | None: ...
    def update_order_status(self, order_id: str, status: OrderStatus, broker_id: str | None = None) -> None: ...
    def write_event(self, kind: str, severity: str = "info", payload_json: str = "{}") -> None: ...


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


class Reconciler(Rule):
    """Reconcile local position state against broker truth.

    Compares ``store.get_positions()`` against ``broker.get_positions()``
    and emits alerts on drift. Also handles orphan order recovery.

    Args:
        drift_tolerance_pct: Maximum acceptable drift as a percentage
            (0.05 = 5%). Drift above this triggers a CRITICAL alert.
        grace_seconds: Grace period before drift is flagged, to avoid
            false positives from positions opened/closed between ticks.
        reconcile_interval_seconds: Minimum seconds between reconcile calls.
        store: The :class:`LiveStore` for local position state.
        broker: The :class:`Broker` for authoritative position state.
    """

    def __init__(
        self,
        *,
        name: str = "reconciler",
        drift_tolerance_pct: float = 0.05,
        grace_seconds: float = 120.0,
        reconcile_interval_seconds: float = 30.0,
        symbols: tuple[str, ...] | None = None,
        store: _StoreLike | None = None,
        broker: _BrokerLike | None = None,
    ) -> None:
        super().__init__(name)
        self.drift_tolerance_pct = drift_tolerance_pct
        self.grace_seconds = grace_seconds
        self.reconcile_interval_seconds = reconcile_interval_seconds
        self._symbols = symbols
        self._store = store
        self._broker = broker
        self._last_reconcile: float = 0.0
        self._drift_cache: dict[str, float] = {}

    # --- AlertEngine.Rule interface -----------------------------------------

    def matches(self, fact: Any) -> Alert | None:
        """Evaluate drift or orphan facts."""
        if isinstance(fact, DriftFact):
            if fact.drift_pct > self.drift_tolerance_pct:
                return Alert(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Position drift for {fact.symbol}: "
                        f"local={fact.local_qty:.6f}, broker={fact.broker_qty:.6f}, "
                        f"drift={fact.drift_pct:.2%}"
                    ),
                    source=f"reconciler:drift:{fact.symbol}",
                    created_at=datetime.now(UTC),
                    extras={
                        "symbol": fact.symbol,
                        "local_qty": fact.local_qty,
                        "broker_qty": fact.broker_qty,
                        "drift_pct": fact.drift_pct,
                    },
                )
            return Alert(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"Minor position drift for {fact.symbol}: "
                    f"drift={fact.drift_pct:.2%} (within tolerance)"
                ),
                source=f"reconciler:drift:{fact.symbol}",
                created_at=datetime.now(UTC),
                extras={
                    "symbol": fact.symbol,
                    "drift_pct": fact.drift_pct,
                },
            )

        if isinstance(fact, OrphanFact):
            return Alert(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(
                    f"Orphan order {fact.order_id} for {fact.symbol} "
                    f"(age={fact.age_seconds:.0f}s)"
                ),
                source=f"reconciler:orphan:{fact.order_id}",
                created_at=datetime.now(UTC),
                extras={
                    "order_id": fact.order_id,
                    "symbol": fact.symbol,
                    "age_seconds": fact.age_seconds,
                },
            )

        return None

    # --- Async reconcile (called by TradingLoop) ---------------------------

    async def reconcile(self) -> list[Alert]:
        """Diff local positions against broker truth and emit alerts on drift.

        Called by the TradingLoop on a timer (every ``reconcile_interval_seconds``).
        Skips if called too soon (respects grace period and minimum interval).

        Returns a list of alerts for any drift or orphan conditions found.
        """
        now = time.time()

        # Throttle: skip if called too soon
        if now - self._last_reconcile < self.reconcile_interval_seconds:
            return []

        if self._store is None or self._broker is None:
            logger.warning("Reconciler has no store or broker; skipping reconcile")
            return []

        alerts: list[Alert] = []

        # 1. Position drift detection
        # Read the broker first (slow REST/await), then read local last so the
        # local snapshot reflects any fill that the fill-drain task persisted
        # while we were waiting on the broker. Reading local first and broker
        # second produces a read-skew: a fill landing during the broker fetch
        # updates local, but the alert compares a stale local read against the
        # fresh broker read, yielding a false 100% drift that self-heals noisily.
        try:
            broker_positions = await self._broker.get_positions()
            local_positions = self._store.get_positions()
        except Exception as e:
            logger.error("Reconciler failed to fetch positions: %s", e)
            return alerts

        self._last_reconcile = now

        # Build lookup maps
        local_map = {p.symbol: p for p in local_positions}
        managed_symbols = self._symbols if self._symbols is not None else None
        broker_map = {
            p.symbol: p
            for p in broker_positions
            if managed_symbols is None or p.symbol in managed_symbols
        }
        all_symbols = set(local_map) | set(broker_map)

        for symbol in all_symbols:
            local_pos = local_map.get(symbol)
            broker_pos = broker_map.get(symbol)

            local_qty = local_pos.qty if local_pos else 0.0
            broker_qty = broker_pos.qty if broker_pos else 0.0

            drift_pct = _compute_drift(local_qty, broker_qty)
            self._drift_cache[symbol] = drift_pct

            fact = DriftFact(
                symbol=symbol,
                local_qty=local_qty,
                broker_qty=broker_qty,
                drift_pct=drift_pct,
            )
            alert = self.matches(fact)
            if alert is not None:
                alerts.append(alert)

            # Update local store with broker truth only for managed symbols.
            # Foreign positions are ignored so multi-symbol workers sharing an
            # account do not overwrite each other's state.
            if managed_symbols is not None and symbol not in managed_symbols:
                continue
            if broker_pos is not None and broker_qty > 0:
                self._store.write_position(broker_pos)
            elif symbol in local_map and broker_qty == 0:
                # Position closed on broker but still in local store
                self._store.delete_position(symbol)

        # 2. Orphan order recovery
        orphan_alerts = self._recover_orphan_orders(now)
        alerts.extend(orphan_alerts)

        # Log reconcile event
        if self._store is not None:
            n_drifts = sum(1 for a in alerts if "drift" in a.source)
            self._store.write_event(
                kind="reconcile",
                severity="info",
                payload_json=f'{{"n_positions": {len(all_symbols)}, "n_drifts": {n_drifts}}}',
            )

        return alerts

    def _recover_orphan_orders(self, now: float) -> list[Alert]:
        """Scan for orphan order intents (pending with no broker_id)."""
        alerts: list[Alert] = []

        if self._store is None:
            return alerts

        # Access the underlying sqlite to scan for orphaned orders
        # We need to query orders with status='pending' and broker_id IS NULL
        try:
            import json

            from kairon.live.broker.base import OrderStatus

            # Use the store's connection directly
            conn = self._store._conn  # type: ignore[attr-defined]
            rows = conn.execute(
                "SELECT id, symbol, ts, broker_id FROM live_orders "
                "WHERE status = 'pending' AND broker_id IS NULL"
            ).fetchall()

            for row in rows:
                order_id, symbol, ts_str, _ = row
                # Parse the timestamp to compute age
                try:
                    ts = datetime.fromisoformat(ts_str)
                    age_seconds = (now - ts.timestamp()) if hasattr(ts, "timestamp") else 0.0
                except (ValueError, AttributeError):
                    age_seconds = 0.0

                # Only flag as orphan if older than grace period
                if age_seconds > self.grace_seconds:
                    fact = OrphanFact(
                        order_id=order_id,
                        symbol=symbol,
                        age_seconds=age_seconds,
                    )
                    alert = self.matches(fact)
                    if alert is not None:
                        alerts.append(alert)

                    # Mark as orphan in the store
                    self._store.update_order_status(order_id, OrderStatus.ORPHAN)

        except Exception as e:
            logger.error("Orphan recovery failed: %s", e)

        return alerts

    def get_drift(self, symbol: str) -> float | None:
        """Return the last computed drift percentage for a symbol, or None."""
        return self._drift_cache.get(symbol)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_drift(local_qty: float, broker_qty: float) -> float:
    """Compute drift as |local - broker| / max(local, broker, epsilon).

    Returns a value in [0, 1] where 0 means no drift.
    """
    if local_qty == 0 and broker_qty == 0:
        return 0.0
    denominator = max(abs(local_qty), abs(broker_qty), 1e-9)
    return abs(local_qty - broker_qty) / denominator


__all__ = ["DriftFact", "OrphanFact", "Reconciler"]