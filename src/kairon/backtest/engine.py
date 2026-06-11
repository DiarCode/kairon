"""Backtest engine: turn a (price, signal) series into a list of trades.

The engine here is deliberately small and pure — no IO, no async, no
broker connection. It runs in three phases:

1. **Signal → target position**. A signal in ``{-1, 0, 1}`` is
   converted to a target position. The "tolerance" parameter
   (``min_signal_change``) prevents thrashing: if a new signal is
   within ``min_signal_change`` of zero we stay flat.
2. **Position transitions → trades**. When the target differs from
   the current position, we close the current one and open a new one
   in the target direction. Costs are applied on both sides.
3. **Equity curve**. The mark-to-market equity is computed on every
   bar that the position is held; the closed-trade PnL is the realised
   PnL once the position is exited.

This is a *vector* backtester (one pass, no event loop) so it's
deterministic, fast, and easy to test. A walk-forward loop wraps it
in :mod:`kairon.backtest.engine`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from typing import Any

import numpy as np
import pyarrow as pa

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.backtest.position import Position, Side


@dataclass(frozen=True, slots=True)
class Trade:
    """A single round-trip trade. Alias for :class:`Position` with
    a stable `id` so mlflow can index it.

    Field semantics match :class:`Position`. We use this name because
    "trade" is the standard backtest vocabulary; "position" is the
    in-flight container.
    """

    id: str
    symbol: str
    side: Side
    opened_at: datetime
    entry_price: float
    size: float
    closed_at: datetime | None
    exit_price: float | None
    entry_costs: float
    exit_costs: float
    pnl: float
    pnl_bps: float
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_position(cls, p: Position, *, extras: dict[str, Any] | None = None) -> Trade:
        return cls(
            id=str(uuid.uuid4()),
            symbol=p.symbol,
            side=p.side,
            opened_at=p.opened_at,
            entry_price=p.entry_price,
            size=p.size,
            closed_at=p.closed_at,
            exit_price=p.exit_price,
            entry_costs=p.entry_costs,
            exit_costs=p.exit_costs,
            pnl=p.pnl,
            pnl_bps=p.pnl_bps,
            extras=extras or {},
        )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The output of a single backtest run."""

    symbol: str
    trades: tuple[Trade, ...]
    equity_curve: tuple[float, ...]  # mark-to-market equity at each bar
    timestamps: tuple[datetime, ...]
    cost: CostModel
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return float(sum(t.pnl for t in self.trades))

    @property
    def total_pnl_bps(self) -> float:
        return float(sum(t.pnl_bps for t in self.trades))

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.closed_at is not None]
        if not closed:
            return float("nan")
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else 0.0

    def to_table(self) -> pa.Table:
        """Serialize trades to a ``pa.Table`` for parquet / mlflow logging."""
        return pa.table(
            {
                "id": [t.id for t in self.trades],
                "symbol": [t.symbol for t in self.trades],
                "side": [t.side.value for t in self.trades],
                "opened_at": pa.array(
                    [t.opened_at for t in self.trades], type=pa.timestamp("ns")
                ),
                "closed_at": pa.array(
                    [t.closed_at for t in self.trades], type=pa.timestamp("ns")
                ),
                "entry_price": [t.entry_price for t in self.trades],
                "exit_price": [t.exit_price for t in self.trades],
                "size": [t.size for t in self.trades],
                "pnl": [t.pnl for t in self.trades],
                "pnl_bps": [t.pnl_bps for t in self.trades],
            }
        )


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------
def signals_to_target(
    signals: np.ndarray,
    *,
    min_signal_change: float = 0.0,
) -> np.ndarray:
    """Map a raw signal series to a target position series in ``{-1, 0, 1}``.

    The signal can be any real number; the sign determines direction.
    A *change threshold* (``min_signal_change``) prevents thrashing:
    we only switch positions when the new signal's absolute value is
    at least ``min_signal_change`` above zero. The threshold is in
    the *signal's* units, not bps.
    """
    if signals.ndim != 1:
        raise ValueError(f"signals must be 1-D, got shape {signals.shape}")
    if min_signal_change < 0:
        raise ValueError(f"min_signal_change must be >= 0, got {min_signal_change}")
    out = np.sign(signals).astype(np.int8)
    if min_signal_change > 0:
        out[np.abs(signals) < min_signal_change] = 0
    return out


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BacktestSpec:
    """Configuration for a single backtest run."""

    cost: CostModel = field(default_factory=lambda: DEFAULT_CRYPTO_COSTS)
    initial_equity: float = 10_000.0
    sizing: str = "fixed_fraction"  # "fixed_fraction" | "kelly" | "all_in"
    fraction: float = 1.0  # fraction of equity to risk per trade
    min_signal_change: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError(f"initial_equity must be > 0, got {self.initial_equity}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.sizing not in {"fixed_fraction", "kelly", "all_in"}:
            raise ValueError(f"unsupported sizing: {self.sizing!r}")
        if self.min_signal_change < 0:
            raise ValueError(f"min_signal_change must be >= 0, got {self.min_signal_change}")


def run_backtest(
    *,
    symbol: str,
    timestamps: np.ndarray,
    close: np.ndarray,
    signals: np.ndarray,
    spec: BacktestSpec | None = None,
) -> BacktestResult:
    """Run a single backtest.

    Parameters
    ----------
    symbol
        Asset symbol (e.g. ``"BTC/USDT"``).
    timestamps
        1-D array of bar timestamps (any sortable type, but typically
        ``datetime64[ns]``). Length must equal ``len(close)``.
    close
        1-D array of close prices.
    signals
        1-D array of raw signals. Converted to ``{-1, 0, 1}`` via
        :func:`signals_to_target` using ``spec.min_signal_change``.
    spec
        Backtest config. Defaults to long-only, 100% sizing, crypto
        cost model.
    """
    spec = spec or BacktestSpec()
    n = len(close)
    if len(timestamps) != n:
        raise ValueError(
            f"timestamps has {len(timestamps)} rows, close has {n}"
        )
    if len(signals) != n:
        raise ValueError(
            f"signals has {len(signals)} rows, close has {n}"
        )

    targets = signals_to_target(signals, min_signal_change=spec.min_signal_change)
    current_pos: Position | None = None
    trades: list[Trade] = []
    equity: float = spec.initial_equity
    equity_curve: list[float] = []
    # We treat the engine as a long/flat backtester. Side.LONG is target=+1,
    # FLAT is target=0. (Phase 5 ships long-only; Phase 11 adds shorts.)
    for i in range(n):
        ts = _to_python_ts(timestamps[i])
        price = float(close[i])
        target = int(targets[i])

        # Mark-to-market
        if current_pos is not None and current_pos.is_open:
            # Long-only: mtm = size * (price - entry)
            equity = spec.initial_equity + current_pos.size * (price - current_pos.entry_price)
        else:
            equity = spec.initial_equity
        equity_curve.append(equity)

        # Position transitions
        if current_pos is None and target == 0:
            continue
        if current_pos is None and target == 1:
            # Open long
            size = _size(equity, price, spec)
            notional = size * price
            entry_costs = spec.cost.total_cost(notional, "buy")
            current_pos = Position(
                symbol=symbol,
                side=Side.LONG,
                opened_at=ts,
                entry_price=price,
                size=size,
                entry_costs=entry_costs,
            )
            continue
        if current_pos is not None and target == 0:
            # Close (target=0 = flat)
            current_pos = _close_position(current_pos, ts, price, spec.cost)
            trades.append(Trade.from_position(current_pos))
            current_pos = None
            continue
        # We deliberately do nothing on SHORT signals yet — Phase 5
        # is long/flat only. A future engine will support shorts.

    # Close any still-open position at the last bar (mark-to-close)
    if current_pos is not None and current_pos.is_open:
        last_ts = _to_python_ts(timestamps[-1])
        last_price = float(close[-1])
        current_pos = _close_position(current_pos, last_ts, last_price, spec.cost)
        trades.append(Trade.from_position(current_pos))
        equity_curve[-1] = spec.initial_equity + current_pos.pnl - current_pos.entry_costs

    return BacktestResult(
        symbol=symbol,
        trades=tuple(trades),
        equity_curve=tuple(equity_curve),
        timestamps=tuple(_to_python_ts(t) for t in timestamps),
        cost=spec.cost,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _size(equity: float, price: float, spec: BacktestSpec) -> float:
    if spec.sizing == "all_in":
        return equity / price
    # fixed_fraction and kelly both use `fraction` for now
    return (equity * spec.fraction) / price


def _close_position(p: Position, ts: datetime, price: float, cost: CostModel) -> Position:
    exit_notional = p.size * price
    exit_costs = cost.total_cost(exit_notional, "sell")
    gross_pnl = p.size * (price - p.entry_price)
    net_pnl = gross_pnl - p.entry_costs - exit_costs
    pnl_bps = (net_pnl / (p.entry_price * p.size)) * 1e4 if p.entry_price > 0 else 0.0
    return Position(
        symbol=p.symbol,
        side=p.side,
        opened_at=p.opened_at,
        entry_price=p.entry_price,
        size=p.size,
        closed_at=ts,
        exit_price=price,
        entry_costs=p.entry_costs,
        exit_costs=exit_costs,
        pnl=net_pnl,
        pnl_bps=pnl_bps,
    )


def _to_python_ts(x: Any) -> datetime:
    """Convert a numpy/datetime value to a Python ``datetime``."""
    if isinstance(x, datetime):
        return x
    if hasattr(x, "item"):
        return x.item() if not isinstance(x, np.datetime64) else _from_dt64(x)
    if isinstance(x, np.datetime64):
        return _from_dt64(x)
    return x  # type: ignore[return-value]


def _from_dt64(x: np.datetime64) -> datetime:
    # numpy datetime64 → Python datetime via epoch seconds
    return datetime.fromtimestamp(x.astype("datetime64[s]").astype(int), tz=UTC).replace(tzinfo=None)


__all__ = [
    "BacktestResult",
    "BacktestSpec",
    "Trade",
    "run_backtest",
    "signals_to_target",
]
