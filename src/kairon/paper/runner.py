"""Composable paper-trading simulator (W7.gate).

The W7 batch (W7.1 latency, W7.2 partial-fill, W7.3 maker rebate) ships
three independent modules. Story W7.gate wires them into a SINGLE,
composable entry-point — :func:`run_simulation` — that takes a signal
stream, applies latency + fill + rebate in the documented order, and
returns a list of :class:`SimulatedTrade` records.

Why composable?
---------------
A v1 paper trader must answer "what would a *real* broker have done
if I sent these signals on this data feed?" The three knobs (latency,
fill, rebate) are independent and apply at different stages of the
order lifecycle:

1. **Latency** (W7.1) — the time between signal-emission and the
   order hitting the book. Latency shifts the entry/exit price by
   the mark-to-mid drift over the latency window.
2. **Partial fill** (W7.2) — the fraction of the order that actually
   fills when the L2 depth is known. No-L2 (the BTC-only path per
   W0) falls back to 100% fill.
3. **Maker rebate** (W7.3) — the per-side rebate on a *limit* order,
   which reduces the net cost in bps.

The :func:`run_simulation` entry-point is the W7.gate acceptance
criterion #3: a single function that composes the three modules. The
function is pure (no IO, no global state); the inputs are signal
streams (timestamps, prices, signals) and the outputs are simulated
trades with realised PnL, latency-adjusted entry/exit prices, and
net-cost-in-bps.

Why a "trade list" return?
---------------------------
The simulation result is a list of :class:`SimulatedTrade` records,
not a :class:`PortfolioState` or an equity curve. The trade list is
the load-bearing artefact for downstream W8.1 / W8.2 / W8.3 reporting
(CAS, DSR, PBO, max-drawdown) — all of those metrics are computed
from the per-trade PnL distribution, not from a portfolio snapshot.

This module is pure: no IO, no async, no global state. It is the
W7.gate deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.paper.cost import DEFAULT_REBATE_BPS, MakerRebateModel, RebateConfig
from kairon.paper.latency import LatencyConfig, LatencySimulator
from kairon.paper.partial_fill import PartialFillConfig, PartialFillSimulator


# Default order kind for the v1 paper trader. The W7.3 rebate model
# applies only to "limit" orders; "market" orders get the commission
# but no rebate. The v1 default is "market" (the simplest path) so
# the headline rebate in :file:`artifacts/w7_simulator.json` matches
# the commission-only W2.2 cost model.
DEFAULT_ORDER_KIND: Final[str] = "market"


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    """One round-trip trade emitted by :func:`run_simulation`.

    Attributes
    ----------
    entry_idx
        Index into the input ``prices`` array where the entry fill
        happened. The fill price is ``prices[entry_idx]`` shifted by
        the latency-induced drift.
    exit_idx
        Index into the input ``prices`` array where the exit fill
        happened. ``None`` for an *open* trade (still in flight at
        end-of-data) — the v1 paper trader does NOT force-close
        open positions.
    side
        ``+1`` for long, ``-1`` for short. The v1 path is
        long-only; a future story can extend the contract.
    entry_price
        The mark price at ``entry_idx`` plus the latency-induced
        drift in price units.
    exit_price
        The mark price at ``exit_idx`` plus the latency-induced
        drift. ``None`` for an open trade.
    filled_qty
        The actual filled quantity in base units (after partial-fill
        shrinkage). Equals ``order_size`` for the no-L2 / small-order
        paths.
    net_cost_bps
        The per-side net cost in bps of notional (commission minus
        rebate for limit orders; commission for market orders).
    latency_ms
        The sampled latency in ms for the entry fill.
    realised_pnl_cash
        The cash PnL for the trade, in price-units. ``None`` for
        an open trade.
    """

    entry_idx: int
    exit_idx: int | None
    side: int
    entry_price: float
    exit_price: float | None
    filled_qty: float
    net_cost_bps: float
    latency_ms: float
    realised_pnl_cash: float | None


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """The headline return value of :func:`run_simulation`.

    Attributes
    ----------
    trades
        The list of :class:`SimulatedTrade` records, in fill order.
    total_trades
        ``len(trades)``. Convenience for downstream reporters.
    closed_trades
        Number of trades with ``exit_idx is not None`` (the trades
        that contribute realised PnL).
    open_trades
        Number of trades with ``exit_idx is None`` (the trades still
        in flight at end-of-data).
    avg_latency_ms
        Mean entry-fill latency across the closed trades. Useful
        for the W7.gate ``latency_p50_ms`` reporting.
    p50_latency_ms
        50th-percentile entry-fill latency across all trades.
    p99_latency_ms
        99th-percentile entry-fill latency across all trades.
    fill_rate
        ``sum(filled_qty) / sum(order_size)`` across all trades.
        A value of 1.0 means every order filled completely (the
        no-L2 path).
    maker_rebate_bps
        The rebate-bps config used (defaults to the W7.3 default
        ``0.2`` bps; 0.0 for the commission-only path).
    """

    trades: list[SimulatedTrade]
    total_trades: int
    closed_trades: int
    open_trades: int
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    fill_rate: float
    maker_rebate_bps: float
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Composable configuration for :func:`run_simulation`.

    All sub-simulators are configurable independently; the W7.gate
    headline numbers (``latency_p50_ms``, ``latency_p99_ms``,
    ``fill_rate_at_default``, ``maker_rebate_bps``) are the
    documented W7.gate acceptance criterion.

    The defaults match the v1 contract:

    - Latency: lognormal(mean=50ms, sigma=0.5, max=500ms).
    - Partial fill: no-L2 path (100% fill for every order).
    - Maker rebate: 0.2 bps (the W7.3 default for limit orders).
    - Order kind: "market" (no rebate applied; commission only).
    """

    latency: LatencyConfig = field(default_factory=LatencyConfig)
    fill: PartialFillConfig = field(default_factory=PartialFillConfig)
    rebate: RebateConfig = field(default_factory=RebateConfig)
    cost: CostModel = field(default_factory=lambda: DEFAULT_CRYPTO_COSTS)
    order_kind: str = DEFAULT_ORDER_KIND
    order_size: float = 0.1  # base units (e.g. 0.1 BTC)
    initial_equity: float = 10_000.0

    def __post_init__(self) -> None:
        if self.order_kind not in ("limit", "market"):
            raise ValueError(
                f"order_kind must be 'limit' or 'market', got {self.order_kind!r}"
            )
        if self.order_size <= 0.0:
            raise ValueError(f"order_size must be > 0, got {self.order_size!r}")
        if self.initial_equity <= 0.0:
            raise ValueError(
                f"initial_equity must be > 0, got {self.initial_equity!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latency_price_drift(
    *,
    prices: np.ndarray,
    entry_idx: int,
    latency_ms: float,
    bar_duration_ms: float,
) -> float:
    """Estimate the latency-induced price drift in price units.

    A round-trip latency of ``latency_ms`` shifts the entry fill by
    the mark-to-mid drift over the latency window. The v1
    approximation is::

        drift = prices[entry_idx] * sigma_per_bar * (latency_ms / bar_duration_ms)

    where ``sigma_per_bar`` is the per-bar return std-dev estimated
    from the local 20-bar window. The result is the *price-units*
    drift (not a return); the simulation adds it to the mark price
    to get the fill price.

    The drift is signed: positive if the local trend is up. This
    matches the v1 convention that "slow fills cost you more in a
    trending market than in a ranging market".
    """
    n: int = int(prices.size)
    if entry_idx < 0 or entry_idx >= n:
        return 0.0
    if latency_ms <= 0.0 or bar_duration_ms <= 0.0:
        return 0.0
    # Local sigma estimate: 20-bar rolling std of log returns.
    start: int = max(1, entry_idx - 20)
    seg: np.ndarray = prices[start : entry_idx + 1].astype(np.float64, copy=False)
    if seg.size < 2:
        return 0.0
    log_rets: np.ndarray = np.diff(np.log(seg))
    sigma_per_bar: float = float(log_rets.std(ddof=0)) if log_rets.size > 0 else 0.0
    if sigma_per_bar <= 0.0:
        return 0.0
    # Local drift: sign of mean log return in the local window.
    drift_sign: float = float(np.sign(log_rets.mean())) if log_rets.size > 0 else 0.0
    # Number of bars the latency covers.
    n_bars: float = float(latency_ms) / float(bar_duration_ms)
    return float(prices[entry_idx] * sigma_per_bar * n_bars * drift_sign)


def _bar_duration_ms(*, timeframe: str) -> float:
    """Return the bar duration in milliseconds for a given timeframe.

    The v1 supports the four W2.2 horizons::

        "1d"  -> 86_400_000 ms
        "1h"  ->  3_600_000 ms
        "15m" ->    900_000 ms
        "5m"  ->    300_000 ms

    An unknown timeframe raises ``ValueError`` so the caller is
    forced to wire the right value (no silent fallback to "1h").
    """
    mapping: dict[str, float] = {
        "1d": 86_400_000.0,
        "1h": 3_600_000.0,
        "15m": 900_000.0,
        "5m": 300_000.0,
    }
    if timeframe not in mapping:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; "
            f"supported: {sorted(mapping)}"
        )
    return mapping[timeframe]


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
def run_simulation(
    *,
    prices: np.ndarray,
    signals: np.ndarray,
    config: SimulationConfig | None = None,
    timeframe: str = "1h",
    l2_depths: np.ndarray | None = None,
) -> SimulationResult:
    """Run the W7 composable paper-trader simulation.

    Parameters
    ----------
    prices
        1-D ``np.ndarray`` of mark prices, in chronological order.
        Length ``N``.
    signals
        1-D ``np.ndarray`` of signals in ``{-1, 0, +1}``. Length
        ``N``. A signal of ``+1`` opens a long on the next bar;
        ``-1`` opens a short (or, in the v1 long-only path, closes
        a long); ``0`` does nothing.
    config
        The :class:`SimulationConfig`. Default: the v1 defaults
        (50ms lognormal latency, 100% fill, 0.2 bps rebate, market
        orders).
    timeframe
        Bar duration. One of ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``.
        Default ``"1h"``.
    l2_depths
        Optional 1-D ``np.ndarray`` of L2 top-of-book depth in base
        units, length ``N``. When ``None`` (the BTC-only path per
        W0), the partial-fill simulator returns 100% fill.

    Returns
    -------
    SimulationResult
        The headline result with the trade list, latency stats, and
        the W7.gate reporting fields. The function NEVER returns an
        equity curve (that is the W8.1 / W8.2 backtest engine's job).
    """
    cfg: SimulationConfig = config or SimulationConfig()
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got ndim={prices.ndim}")
    if signals.ndim != 1:
        raise ValueError(f"signals must be 1-D, got ndim={signals.ndim}")
    n: int = int(prices.size)
    if signals.size != n:
        raise ValueError(
            f"signals has length {signals.size}, prices has length {n}"
        )
    if l2_depths is not None and l2_depths.shape[0] != n:
        raise ValueError(
            f"l2_depths has length {l2_depths.shape[0]}, prices has length {n}"
        )
    bar_ms: float = _bar_duration_ms(timeframe=timeframe)

    latency_sim: LatencySimulator = LatencySimulator(cfg.latency)
    fill_sim: PartialFillSimulator = PartialFillSimulator(cfg.fill)
    rebate_model: MakerRebateModel = MakerRebateModel(cfg.rebate)
    net_cost_bps_per_side: float = rebate_model.net_cost_bps(cfg.order_kind)  # type: ignore[arg-type]

    trades: list[SimulatedTrade] = []
    open_position: SimulatedTrade | None = None
    total_order_qty: float = 0.0
    total_filled_qty: float = 0.0
    latencies_ms: list[float] = []

    for i in range(n):
        sig: int = int(np.sign(float(signals[i])))
        price_i: float = float(prices[i])
        if sig == 0:
            continue
        # 1) Latency: sample a fresh round-trip latency for this fill.
        latency_ms: float = float(latency_sim.sample())
        latencies_ms.append(latency_ms)
        # 2) Latency-induced price drift.
        drift: float = _latency_price_drift(
            prices=prices,
            entry_idx=i,
            latency_ms=latency_ms,
            bar_duration_ms=bar_ms,
        )
        fill_price: float = max(0.0, price_i + drift)
        # 3) Partial fill: query the L2 depth if available.
        l2_depth: float | None = None
        if l2_depths is not None:
            l2_depth = float(l2_depths[i])
        fill_result = fill_sim.simulate(qty=cfg.order_size, l2_depth=l2_depth)
        total_order_qty += float(cfg.order_size)
        total_filled_qty += float(fill_result.filled_qty)
        # 4) If we have an open position, check whether the signal
        #    closes it (or reverses).
        if open_position is not None:
            # Long: close on sig==-1. Short: close on sig==+1.
            if (open_position.side == 1 and sig == -1) or (
                open_position.side == -1 and sig == 1
            ):
                exit_price: float = fill_price
                # Realised PnL in price units (per filled unit).
                pnl_per_unit: float = (exit_price - open_position.entry_price) * (
                    open_position.side
                )
                pnl_cash: float = pnl_per_unit * float(open_position.filled_qty)
                # Subtract the round-trip cost drag from realised PnL.
                rt_cost_cash: float = (
                    float(open_position.filled_qty)
                    * 0.5
                    * (open_position.entry_price + exit_price)
                    * (2.0 * net_cost_bps_per_side)
                    / 1e4
                )
                pnl_cash -= rt_cost_cash
                closed: SimulatedTrade = SimulatedTrade(
                    entry_idx=open_position.entry_idx,
                    exit_idx=i,
                    side=open_position.side,
                    entry_price=open_position.entry_price,
                    exit_price=exit_price,
                    filled_qty=open_position.filled_qty,
                    net_cost_bps=open_position.net_cost_bps,
                    latency_ms=open_position.latency_ms,
                    realised_pnl_cash=pnl_cash,
                )
                trades.append(closed)
                open_position = None
                continue
        # 5) Open a new position (or, in the long-only path, skip
        #    shorts).
        if open_position is None and sig != 0:
            if sig < 0:
                # Long-only v1: skip shorts (the W6.4 short-side
                # sizer is a future story). The signal is recorded
                # as "skipped" via the open-trades counter.
                continue
            open_position = SimulatedTrade(
                entry_idx=i,
                exit_idx=None,
                side=1,
                entry_price=fill_price,
                exit_price=None,
                filled_qty=float(fill_result.filled_qty),
                net_cost_bps=float(net_cost_bps_per_side),
                latency_ms=latency_ms,
                realised_pnl_cash=None,
            )

    closed_trades: int = sum(1 for t in trades if t.realised_pnl_cash is not None)
    open_trades: int = len(latencies_ms) - closed_trades
    if latencies_ms:
        arr: np.ndarray = np.asarray(latencies_ms, dtype=np.float64)
        p50: float = float(np.percentile(arr, 50))
        p99: float = float(np.percentile(arr, 99))
        avg: float = float(arr.mean())
    else:
        p50 = 0.0
        p99 = 0.0
        avg = 0.0
    fill_rate: float = float(total_filled_qty / total_order_qty) if total_order_qty > 0 else 0.0
    maker_rebate_bps: float = (
        float(cfg.rebate.rebate_bps) if cfg.order_kind == "limit" else 0.0
    )

    return SimulationResult(
        trades=trades,
        total_trades=len(trades),
        closed_trades=closed_trades,
        open_trades=open_trades,
        avg_latency_ms=avg,
        p50_latency_ms=p50,
        p99_latency_ms=p99,
        fill_rate=fill_rate,
        maker_rebate_bps=maker_rebate_bps,
    )


__all__ = [
    "DEFAULT_ORDER_KIND",
    "DEFAULT_REBATE_BPS",
    "SimulatedTrade",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
]
