"""Guardian: risk checks and kill switch for the live trading loop.

The :class:`Guardian` is a :class:`AlertEngine.Rule` subclass that enforces
capital preservation constraints:

- Per-position equity cap (default 20%)
- Maximum total leverage (default 1.0x)
- Maximum open positions (default 5)
- Daily-loss kill switch (default 3% of equity → halt)
- Per-symbol cooldown after stop-loss hit (default 4h)

The :class:`TradingLoop` calls :meth:`Guardian.check_positions` and
:meth:`Guardian.check_daily_loss` synchronously on every tick, BEFORE
submitting any order to the broker. If a CRITICAL alert is returned,
the loop calls ``LiveStore.halt()`` and skips the order.

The Guardian is also registered as an :class:`AlertEngine.Rule` so it can
emit alerts through the standard channel when fact-based evaluation is used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kairon.live.alerts import Alert, Rule, Severity
from kairon.live.broker.base import OrderSide, Position


# ---------------------------------------------------------------------------
# Fact types the Guardian can evaluate via AlertEngine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionsFact:
    """Current positions and equity, fed to the Guardian on each tick."""

    positions: tuple[Position, ...]
    equity: float


@dataclass(frozen=True)
class DailyPnlFact:
    """Daily realized PnL and equity for kill-switch evaluation."""

    daily_pnl: float
    equity: float


@dataclass(frozen=True)
class CooldownFact:
    """Check whether a symbol is in cooldown after a stop-loss."""

    symbol: str
    now: float  # epoch seconds


# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------


class Guardian(Rule):
    """Risk guardian: enforces position limits, leverage caps, daily-loss
    kill switch, and per-symbol cooldowns.

    The Guardian is both:

    1. An :class:`AlertEngine.Rule` that evaluates :class:`PositionsFact`,
       :class:`DailyPnlFact`, and :class:`CooldownFact` via :meth:`matches`.
    2. A synchronous, in-tick check that the :class:`TradingLoop` calls
       directly via :meth:`check_positions` and :meth:`check_daily_loss`
       before submitting any order.

    Args:
        max_position_equity_fraction: Max fraction of equity per position.
        max_total_leverage: Max total leverage across all positions.
        max_open_positions: Max number of simultaneous open positions.
        max_daily_loss_pct: Max daily loss as fraction of equity before halt.
        cooldown_seconds: Per-symbol cooldown after SL hit (seconds).
        store: Optional :class:`LiveStore` for halting. If provided,
            CRITICAL daily-loss alerts also call ``store.halt()``.
    """

    def __init__(
        self,
        *,
        name: str = "guardian",
        max_position_equity_fraction: float = 0.20,
        max_total_leverage: float = 1.0,
        max_open_positions: int = 5,
        max_daily_loss_pct: float = 0.03,
        cooldown_seconds: float = 4 * 3600,
        store: Any | None = None,
    ) -> None:
        super().__init__(name)
        self.max_position_equity_fraction = max_position_equity_fraction
        self.max_total_leverage = max_total_leverage
        self.max_open_positions = max_open_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.cooldown_seconds = cooldown_seconds
        self._store = store
        self._cooldowns: dict[str, float] = {}  # symbol → epoch of last SL

    # --- AlertEngine.Rule interface -----------------------------------------

    def matches(self, fact: Any) -> Alert | None:
        """Evaluate a fact and return an Alert if a threshold is breached."""
        if isinstance(fact, PositionsFact):
            alerts = self.check_positions(fact.positions, fact.equity)
            return alerts[0] if alerts else None
        if isinstance(fact, DailyPnlFact):
            return self.check_daily_loss(fact.daily_pnl, fact.equity)
        if isinstance(fact, CooldownFact):
            if self.is_cooling_down(fact.symbol, fact.now):
                return Alert(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=f"Symbol {fact.symbol} is in cooldown",
                    source=f"cooldown:{fact.symbol}",
                    created_at=datetime.now(UTC),
                    extras={"symbol": fact.symbol},
                )
            return None
        return None

    # --- Direct-check interface (called by TradingLoop) --------------------

    def check_positions(
        self,
        positions: tuple[Position, ...] | list[Position],
        equity: float,
    ) -> list[Alert]:
        """Check position limits, leverage, and count.

        Called by the TradingLoop before submitting any order.
        Returns a list of CRITICAL alerts for each breached constraint.
        """
        if equity <= 0:
            return []

        positions_tuple = tuple(positions)
        alerts: list[Alert] = []

        # 1. Max open positions
        if len(positions_tuple) > self.max_open_positions:
            alerts.append(
                Alert(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Too many open positions: {len(positions_tuple)} > "
                        f"{self.max_open_positions}"
                    ),
                    source="guardian:max_positions",
                    created_at=datetime.now(UTC),
                    extras={
                        "n_positions": len(positions_tuple),
                        "max_open_positions": self.max_open_positions,
                    },
                )
            )

        # 2. Per-position equity cap
        for pos in positions_tuple:
            notional = pos.qty * pos.avg_entry
            fraction = notional / equity
            if fraction > self.max_position_equity_fraction:
                alerts.append(
                    Alert(
                        rule_name=self.name,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Position {pos.symbol} uses {fraction:.1%} of equity, "
                            f"exceeding {self.max_position_equity_fraction:.1%} cap"
                        ),
                        source=f"guardian:position_cap:{pos.symbol}",
                        created_at=datetime.now(UTC),
                        extras={
                            "symbol": pos.symbol,
                            "fraction": fraction,
                            "max_fraction": self.max_position_equity_fraction,
                        },
                    )
                )

        # 3. Total leverage
        total_notional = sum(p.qty * p.avg_entry for p in positions_tuple)
        if equity > 0 and total_notional / equity > self.max_total_leverage:
            leverage = total_notional / equity
            alerts.append(
                Alert(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Total leverage {leverage:.2f}x exceeds "
                        f"{self.max_total_leverage:.1f}x cap"
                    ),
                    source="guardian:total_leverage",
                    created_at=datetime.now(UTC),
                    extras={
                        "leverage": leverage,
                        "max_leverage": self.max_total_leverage,
                    },
                )
            )

        # 4. Halt if any CRITICAL alert and we have a store
        if alerts and self._store is not None:
            reasons = "; ".join(a.message for a in alerts)
            self._store.halt(reason=reasons)

        return alerts

    def check_daily_loss(self, daily_pnl: float, equity: float) -> Alert | None:
        """Check daily-loss kill switch.

        If the daily realized PnL exceeds ``max_daily_loss_pct * equity``,
        returns a CRITICAL alert and halts the store (if provided).
        """
        if equity <= 0:
            return None

        # daily_pnl is negative when losing
        loss_fraction = -daily_pnl / equity
        if loss_fraction > self.max_daily_loss_pct:
            alert = Alert(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(
                    f"Daily loss {loss_fraction:.2%} of equity exceeds "
                    f"{self.max_daily_loss_pct:.2%} limit (PnL={daily_pnl:.2f}, "
                    f"equity={equity:.2f})"
                ),
                source="guardian:daily_loss",
                created_at=datetime.now(UTC),
                extras={
                    "daily_pnl": daily_pnl,
                    "equity": equity,
                    "loss_fraction": loss_fraction,
                    "max_daily_loss_pct": self.max_daily_loss_pct,
                },
            )
            if self._store is not None:
                self._store.halt(reason=f"daily_loss_limit: {loss_fraction:.2%}")
            return alert

        return None

    # --- Cooldown management ------------------------------------------------

    def record_sl(self, symbol: str, now: float | None = None) -> None:
        """Record that a stop-loss was hit for ``symbol``, starting the cooldown."""
        if now is None:
            now = time.time()
        self._cooldowns[symbol] = now

    def is_cooling_down(self, symbol: str, now: float | None = None) -> bool:
        """Check if ``symbol`` is still in cooldown after a stop-loss.

        Returns True if the symbol should NOT be traded.
        """
        if now is None:
            now = time.time()
        last_sl = self._cooldowns.get(symbol)
        if last_sl is None:
            return False
        return (now - last_sl) < self.cooldown_seconds

    def cooldown_remaining(self, symbol: str, now: float | None = None) -> float:
        """Return seconds remaining in cooldown for ``symbol``, or 0.0."""
        if now is None:
            now = time.time()
        last_sl = self._cooldowns.get(symbol)
        if last_sl is None:
            return 0.0
        remaining = self.cooldown_seconds - (now - last_sl)
        return max(0.0, remaining)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "CooldownFact",
    "DailyPnlFact",
    "Guardian",
    "PositionsFact",
]