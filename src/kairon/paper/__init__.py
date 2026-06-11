"""Paper trading engine — a stateful broker simulator.

A *paper trader* tracks positions, fills, and PnL for a *single* symbol
using the same cost model the backtester uses. The difference is that
the paper trader is *stateful* and *event-driven*: it accepts discrete
``Order`` objects and emits a stream of ``Fill`` and ``PortfolioState``
snapshots, instead of being driven by a static signal series.

Why a separate engine?

- **Honest cost accounting** — costs are applied at fill time using the
  same model the backtest uses, so paper and back are directly
  comparable.
- **Sequence-of-events audit** — every state change has a unique
  ``event_id`` and timestamp; you can replay the ledger and reproduce
  the final state.
- **Reconciliation** — a :meth:`PaperTrader.reconcile` method compares
  the internal state against an external source of truth (e.g. an
  exchange balance API) and surfaces any drift.

This module is *not* a broker adapter; it assumes someone else (the
ingestion layer) is calling :meth:`submit_order` with market/limit
orders it constructed from the live data.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.backtest.position import Side


# ---------------------------------------------------------------------------
# Orders & fills
# ---------------------------------------------------------------------------
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Order:
    """A request to enter or exit a position."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    size: float = 0.0  # units (not notional)
    limit_price: float | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Fill:
    """The result of a filled order."""

    order_id: str
    symbol: str
    side: OrderSide
    size: float
    price: float
    costs: float
    filled_at: datetime
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """A point-in-time view of a single symbol's position."""

    symbol: str
    side: Side
    size: float
    entry_price: float | None
    entry_cost_cash: float
    mark_price: float | None
    unrealised_pnl: float
    opened_at: datetime | None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """A point-in-time view of the whole book."""

    cash: float
    equity: float
    positions: tuple[PositionSnapshot, ...]
    realised_pnl: float
    unrealised_pnl: float
    timestamp: datetime
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_open(self) -> int:
        return sum(1 for p in self.positions if p.side != Side.FLAT)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@dataclass
class _InternalPosition:
    """Mutable state for an open position."""

    symbol: str
    side: Side
    size: float
    entry_price: float
    entry_costs_cash: float
    opened_at: datetime


@dataclass(frozen=True, slots=True)
class PaperTraderConfig:
    """Configuration for the paper trading engine."""

    cost: CostModel = field(default_factory=lambda: DEFAULT_CRYPTO_COSTS)
    initial_cash: float = 10_000.0
    allow_short: bool = False
    max_position_size: float = float("inf")
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {self.initial_cash}")
        if self.max_position_size <= 0:
            raise ValueError(f"max_position_size must be > 0, got {self.max_position_size}")


class PaperTrader:
    """A stateful paper broker.

    Usage::

        trader = PaperTrader(PaperTraderConfig(initial_cash=10_000))
        trader.on_price("BTC/USDT", 50_000.0)  # update mark
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, size=0.1)
        fill = trader.submit_order(order, fill_price=50_100.0)
        state = trader.snapshot()

    The engine is *single-threaded by construction*; in production
    wrap the call sequence in a lock to avoid races with the live
    data feed.
    """

    def __init__(self, config: PaperTraderConfig | None = None) -> None:
        self.config = config or PaperTraderConfig()
        self._cash: float = self.config.initial_cash
        self._positions: dict[str, _InternalPosition] = {}
        self._marks: dict[str, float] = {}
        self._fills: list[Fill] = []
        self._realised_pnl: float = 0.0
        self._last_event_id: int = 0
        self._started_at: datetime = datetime.now(UTC)
        self._events: list[dict[str, Any]] = []

    # -- public API --------------------------------------------------------
    @property
    def cash(self) -> float:
        return float(self._cash)

    @property
    def n_fills(self) -> int:
        return len(self._fills)

    @property
    def realised_pnl(self) -> float:
        return float(self._realised_pnl)

    def on_price(self, symbol: str, mark: float) -> None:
        """Update the mark price for a symbol (does not generate a fill)."""
        if mark <= 0:
            raise ValueError(f"mark must be > 0, got {mark}")
        self._marks[symbol] = float(mark)
        self._log_event("mark", {"symbol": symbol, "mark": mark})

    def submit_order(
        self,
        order: Order,
        *,
        fill_price: float | None = None,
        at: datetime | None = None,
    ) -> Fill | None:
        """Submit an order and attempt to fill it immediately.

        Market orders are always filled at ``fill_price`` (which the
        caller must supply from the live tape). Limit orders are filled
        only if the mark price crosses the limit; otherwise the order
        is recorded as ``PENDING`` and ignored.
        """
        if order.size <= 0:
            raise ValueError(f"order size must be > 0, got {order.size}")
        if order.symbol not in self._marks and fill_price is None:
            raise ValueError(
                f"symbol {order.symbol!r} has no mark price; "
                "call on_price() or pass fill_price"
            )
        fill_at = at or datetime.now(UTC)
        self._log_event(
            "order",
            {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "size": order.size,
                "type": order.order_type.value,
                "limit": order.limit_price,
            },
        )

        if order.order_type == OrderType.LIMIT:
            mark = self._marks.get(order.symbol)
            if mark is None or fill_price is None:
                return None
            if order.side == OrderSide.BUY and mark > (order.limit_price or float("inf")):
                return None
            if order.side == OrderSide.SELL and mark < (order.limit_price or -float("inf")):
                return None
        price = float(fill_price) if fill_price is not None else self._marks[order.symbol]
        if price <= 0:
            raise ValueError(f"fill_price must be > 0, got {price}")

        # Position-size sanity check
        if order.size > self.config.max_position_size:
            raise ValueError(
                f"order size {order.size} exceeds max_position_size {self.config.max_position_size}"
            )

        # Apply fill to internal state
        pos = self._positions.get(order.symbol)
        notional = order.size * price
        costs = self.config.cost.total_cost(notional, order.side.value)
        if pos is None or pos.side == Side.FLAT:
            # Opening
            if order.side == OrderSide.SELL and not self.config.allow_short:
                raise ValueError("short selling is not enabled in this paper trader")
            side = Side.LONG if order.side == OrderSide.BUY else Side.SHORT
            self._cash -= notional + costs
            self._positions[order.symbol] = _InternalPosition(
                symbol=order.symbol,
                side=side,
                size=order.size,
                entry_price=price,
                entry_costs_cash=costs,
                opened_at=fill_at,
            )
        # Reducing or reversing
        elif pos.side == Side.LONG and order.side == OrderSide.SELL:
            # Closing long (or reversing)
            close_qty = min(order.size, pos.size)
            # PnL on the closing portion only; costs are split pro-rata
            entry_share = pos.entry_costs_cash * (close_qty / pos.size) if pos.size else 0.0
            pnl = close_qty * (price - pos.entry_price) - entry_share - costs
            self._cash += close_qty * price - costs
            self._realised_pnl += pnl
            # Allocate remaining entry_costs pro-rata
            if close_qty >= pos.size:
                self._positions.pop(order.symbol, None)
            else:
                remaining = pos.size - close_qty
                self._positions[order.symbol] = _InternalPosition(
                    symbol=order.symbol,
                    side=pos.side,
                    size=remaining,
                    entry_price=pos.entry_price,
                    entry_costs_cash=pos.entry_costs_cash - entry_share,
                    opened_at=pos.opened_at,
                )
            # Reversal
            if order.size > close_qty + 1e-9:
                extra = order.size - close_qty
                if not self.config.allow_short:
                    # Short selling disabled: refuse the extra
                    raise ValueError(
                        "short selling is not enabled in this paper trader "
                        f"(requested extra {extra} on the short side)"
                    )
                extra_costs = self.config.cost.total_cost(extra * price, "buy")
                self._cash -= extra * price + extra_costs
                self._positions[order.symbol] = _InternalPosition(
                    symbol=order.symbol,
                    side=Side.LONG,
                    size=extra,
                    entry_price=price,
                    entry_costs_cash=extra_costs,
                    opened_at=fill_at,
                )
        elif pos.side == Side.SHORT and order.side == OrderSide.BUY:
            close_qty = min(order.size, pos.size)
            entry_share = pos.entry_costs_cash * (close_qty / pos.size) if pos.size else 0.0
            pnl = close_qty * (pos.entry_price - price) - entry_share - costs
            self._cash -= close_qty * price + costs
            self._realised_pnl += pnl
            if close_qty >= pos.size:
                self._positions.pop(order.symbol, None)
            else:
                remaining = pos.size - close_qty
                self._positions[order.symbol] = _InternalPosition(
                    symbol=order.symbol,
                    side=pos.side,
                    size=remaining,
                    entry_price=pos.entry_price,
                    entry_costs_cash=pos.entry_costs_cash - entry_share,
                    opened_at=pos.opened_at,
                )
            if order.size > close_qty + 1e-9:
                extra = order.size - close_qty
                extra_costs = self.config.cost.total_cost(extra * price, "buy")
                self._cash -= extra * price + extra_costs
                self._positions[order.symbol] = _InternalPosition(
                    symbol=order.symbol,
                    side=Side.LONG,
                    size=extra,
                    entry_price=price,
                    entry_costs_cash=extra_costs,
                    opened_at=fill_at,
                )
        # Same-side adding
        elif order.side == OrderSide.BUY:
            new_size = pos.size + order.size
            new_cost = pos.entry_costs_cash + costs
            new_entry = (
                (pos.entry_price * pos.size + price * order.size) / new_size
                if new_size > 0
                else price
            )
            self._cash -= order.size * price + costs
            self._positions[order.symbol] = _InternalPosition(
                symbol=order.symbol,
                side=pos.side,
                size=new_size,
                entry_price=new_entry,
                entry_costs_cash=new_cost,
                opened_at=pos.opened_at,
            )

        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            price=price,
            costs=costs,
            filled_at=fill_at,
        )
        self._fills.append(fill)
        self._log_event(
            "fill",
            {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "size": order.size,
                "price": price,
                "costs": costs,
            },
        )
        return fill

    def cancel_all(self, symbol: str | None = None) -> int:
        """No-op for the moment — paper trader fills immediately. Returns 0."""
        return 0

    def snapshot(self) -> PortfolioState:
        """Return a :class:`PortfolioState` snapshot of the book."""
        snaps: list[PositionSnapshot] = []
        unreal = 0.0
        for sym, pos in self._positions.items():
            mark = self._marks.get(sym)
            if mark is None:
                upnl = 0.0
            elif pos.side == Side.LONG:
                upnl = pos.size * (mark - pos.entry_price)
            elif pos.side == Side.SHORT:
                upnl = pos.size * (pos.entry_price - mark)
            else:
                upnl = 0.0
            unreal += upnl
            snaps.append(
                PositionSnapshot(
                    symbol=sym,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    entry_cost_cash=pos.entry_costs_cash,
                    mark_price=mark,
                    unrealised_pnl=upnl,
                    opened_at=pos.opened_at,
                )
            )
        return PortfolioState(
            cash=self._cash,
            equity=self._cash + unreal,
            positions=tuple(snaps),
            realised_pnl=self._realised_pnl,
            unrealised_pnl=unreal,
            timestamp=datetime.now(UTC),
        )

    def reconcile(
        self,
        external_cash: float,
        external_positions: dict[str, float],
    ) -> dict[str, Any]:
        """Compare internal state against an external source.

        ``external_positions`` is a ``{symbol: signed_size}`` map (long
        positive, short negative). Returns a dict with ``drift_cash``
        and ``drift_positions`` (per-symbol discrepancy).
        """
        snap = self.snapshot()
        drift_cash = external_cash - snap.cash
        drift_positions: dict[str, dict[str, float]] = {}
        seen = set()
        for sym, ext_size in external_positions.items():
            seen.add(sym)
            internal_size = next(
                (
                    p.size
                    * (1 if p.side == Side.LONG else (-1 if p.side == Side.SHORT else 0))
                )
                for p in snap.positions
                if p.symbol == sym
            )
            if abs(internal_size - ext_size) > 1e-9:
                drift_positions[sym] = {"internal": internal_size, "external": ext_size}
        for p in snap.positions:
            if p.symbol not in seen and p.size > 0:
                drift_positions[p.symbol] = {
                    "internal": p.size if p.side == Side.LONG else -p.size,
                    "external": 0.0,
                }
        return {
            "drift_cash": drift_cash,
            "drift_positions": drift_positions,
            "checked_at": datetime.now(UTC).isoformat(),
        }

    # -- helpers -----------------------------------------------------------
    def _log_event(self, kind: str, payload: dict[str, Any]) -> None:
        self._last_event_id += 1
        self._events.append(
            {
                "event_id": self._last_event_id,
                "kind": kind,
                "ts": datetime.now(UTC).isoformat(),
                **payload,
            }
        )

    def events(self) -> list[dict[str, Any]]:
        """Return a copy of the in-memory event log."""
        return list(self._events)

    def fills(self) -> tuple[Fill, ...]:
        """Return a copy of the fill history."""
        return tuple(self._fills)


# ---------------------------------------------------------------------------
# Convenience: a quick scenario runner
# ---------------------------------------------------------------------------
def run_paper_scenario(
    *,
    symbol: str,
    timestamps: np.ndarray,
    prices: np.ndarray,
    signals: np.ndarray,
    trader: PaperTrader | None = None,
    size_per_trade: float = 0.1,
) -> tuple[PaperTrader, PortfolioState]:
    """Replay a (price, signal) series through the paper trader.

    Returns the final :class:`PaperTrader` and a :class:`PortfolioState`
    snapshot. Useful for end-to-end smoke tests.
    """
    if trader is None:
        trader = PaperTrader()
    if len(timestamps) != len(prices) or len(prices) != len(signals):
        raise ValueError("timestamps, prices, and signals must have the same length")
    for i in range(len(prices)):
        ts = timestamps[i]
        price = float(prices[i])
        sig = float(signals[i])
        trader.on_price(symbol, price)
        if sig > 0:
            trader.submit_order(
                Order(symbol=symbol, side=OrderSide.BUY, size=size_per_trade),
                fill_price=price,
                at=ts if hasattr(ts, "year") else datetime.now(UTC),
            )
        elif sig < 0:
            trader.submit_order(
                Order(symbol=symbol, side=OrderSide.SELL, size=size_per_trade),
                fill_price=price,
                at=ts if hasattr(ts, "year") else datetime.now(UTC),
            )
    return trader, trader.snapshot()


__all__ = [
    "Fill",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperTrader",
    "PaperTraderConfig",
    "PortfolioState",
    "PositionSnapshot",
    "run_paper_scenario",
]
