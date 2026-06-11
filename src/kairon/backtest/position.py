"""Position: a single open trade.

The backtest is long/flat only at this stage (Phase 5 ships the long-only
absolute-return PnL; the short-side is added in Phase 11 alongside the
PatchTST ensemble).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    """Direction of a position."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class Position:
    """A single open (or just-closed) position.

    ``opened_at`` is the bar timestamp at which the position was opened;
    ``closed_at`` is the bar at which it was exited (or ``None`` if still
    open). All prices are notional prices *per unit* (e.g. USD per share,
    USD per BTC).

    ``pnl`` is the realised PnL in cash units; ``pnl_bps`` is the same
    value expressed in bps of the entry notional. The two are redundant
    but convenient for downstream code.
    """

    symbol: str
    side: Side
    opened_at: datetime
    entry_price: float
    size: float  # units, not notional
    closed_at: datetime | None = None
    exit_price: float | None = None
    entry_costs: float = 0.0
    exit_costs: float = 0.0
    pnl: float = 0.0
    pnl_bps: float = 0.0

    @property
    def notional(self) -> float:
        return self.entry_price * self.size

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


__all__ = ["Position", "Side"]
