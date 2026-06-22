"""Bybit TESTNET-calibrated cost model + fidelity helpers for the scalping sim.

The shared :class:`kairon.backtest.cost.CostModel` is venue-agnostic; this module
provides a preset calibrated to Bybit's USDT-perpetual TESTNET fee schedule and a
pair of pure helpers that compute the *expected* net PnL / loss fraction for a
trade from the same primitives the engine uses. The Phase 1.3 fidelity gate
asserts the engine's realized trade math equals these helpers — i.e. there is
one implementation of the risk/cost math, shared between live and sim.

Fee calibration (Bybit USDT perp, taker):
    * commission: 0.055% per side = 5.5 bps (scalping uses market/taker orders
      for both entry and exit; attached TP/SL triggers are taker fills).
    * half-spread: ~1 bp (testnet books are thin but generally tight at the top).
    * slippage: ~1 bp static (a market order crosses the book; the attached-stop
      exit already pays the stop-level-vs-crossing spread inside the engine via
      :func:`kairon.live.pure_fns.stop_exit_price`).
Round-trip = 2 * (5.5 + 1 + 1) = 15 bps of notional.

TESTNET-only label: these numbers reflect the testnet fee schedule, which Bybit
mirrors from mainnet. Do not mix with a mainnet cost model.
"""

from __future__ import annotations

from kairon.backtest.cost import CostModel
from kairon.backtest.position import Side

# Bybit USDT-perp TESTNET taker cost preset.
BYBIT_TESTNET_COSTS: CostModel = CostModel(
    commission_bps=5.5,
    slippage_bps=1.0,
    half_spread_bps=1.0,
    impact_coefficient=0.0,
    min_trade_bps=0.0,
)


def round_trip_cost_bps(cost: CostModel) -> float:
    """Total round-trip cost in bps of notional (entry + exit)."""
    return cost.round_trip_bps


def fidelity_expected_net_pnl(
    *,
    side: Side,
    qty: float,
    entry_price: float,
    exit_price: float,
    cost: CostModel,
) -> float:
    """The net PnnL the engine MUST record for a trade with these inputs.

    ``gross = qty * (exit - entry)`` for long, ``qty * (entry - exit)`` for short;
    fees = entry notional cost + exit notional cost (both taker). This is a pure
    restatement of :func:`kairon.backtest.scalping_engine._close_trade`'s math so
    the fidelity gate can assert the engine reproduces it exactly — proving the
    sim and the pure risk functions share one implementation.
    """
    if side is Side.LONG:
        gross = qty * (exit_price - entry_price)
    else:
        gross = qty * (entry_price - exit_price)
    entry_cost = cost.total_cost(qty * entry_price, "entry")
    exit_cost = cost.total_cost(qty * exit_price, "exit")
    return gross - entry_cost - exit_cost


def implied_loss_fraction(
    *, qty: float, sl_distance: float, bankroll: float, cost: CostModel, entry_price: float,
) -> float:
    """Fraction of the bankroll lost when the SL is hit, fees included.

    The pure risk primitive gives ``implied_risk = qty * sl_distance / bankroll``
    (the gross loss fraction); this subtracts the round-trip fee fraction of the
    entry notional to give the net realized loss fraction the engine records.
    Used by the fidelity gate to check the documented SOL min-lot-overshoot case:
    a 0.074 risk-sized qty bumped to the 0.1 min lot loses ~3.4% of a $10 bankroll
    on an SL hit, not the 2.5% target — the exact overshoot the
    :func:`kairon.live.pure_fns.post_rounding_guard` exists to detect.
    """
    if bankroll <= 0 or entry_price <= 0:
        return 0.0
    gross_loss_fraction = (qty * sl_distance) / bankroll
    fee_fraction = (cost.round_trip_bps / 1e4) * (qty * entry_price) / bankroll
    return gross_loss_fraction + fee_fraction


__all__ = [
    "BYBIT_TESTNET_COSTS",
    "fidelity_expected_net_pnl",
    "implied_loss_fraction",
    "round_trip_cost_bps",
]
