"""Pure, side-effect-free trading functions shared by the live orchestrator and
the vectorized backtest.

These are extracted from :class:`~kairon.live.orchestrator.TradingLoop` so the
backtest can replay bars with the *exact* risk-sizing, stop, and exit semantics
without instantiating the async/live-only orchestrator. Everything here is a
pure function of its arguments — no I/O, no async, no broker, no clocks — so it
is deterministic and unit-testable in isolation.

The live orchestrator imports and calls these; the backtest engine imports and
calls the same functions. Drift between live and sim is therefore impossible by
construction (there is one implementation).
"""

from __future__ import annotations

import math
from typing import Literal


def risk_size_qty(
    *,
    bankroll: float,
    risk_per_trade: float,
    sl_distance: float | None,
    notional_cap_qty: float,
) -> float:
    """Fixed-fractional risk sizing, capped by the leverage notional.

    ``qty = (risk_per_trade * bankroll) / sl_distance``, capped by
    ``notional_cap_qty`` so a tight stop cannot blow past the position cap.
    When ``sl_distance`` is None/<=0 (no usable stop), falls back to notional
    sizing. Returns a non-negative magnitude (the caller applies direction).
    """
    if risk_per_trade > 0 and sl_distance is not None and sl_distance > 0:
        risk_qty = (risk_per_trade * bankroll) / sl_distance
        return min(risk_qty, notional_cap_qty)
    return notional_cap_qty


def implied_risk(qty: float, sl_distance: float | None, bankroll: float) -> float:
    """Fraction of the bankroll actually risked by ``qty`` over ``sl_distance``.

    ``implied_risk = qty * sl_distance / bankroll``. This is the post-rounding
    truth check: the broker may floor/bump the intended quantity to the min
    lot, and (with confidence-scaled sizing) the intended quantity may be
    inflated by a multiplier. Whatever quantity actually trades, this is the
    fraction of the bankroll lost if the stop is hit. Returns 0.0 when the
    inputs are not usable (no stop distance, no bankroll).
    """
    if sl_distance is None or sl_distance <= 0 or bankroll <= 0:
        return 0.0
    return (abs(qty) * sl_distance) / bankroll


def clamp_effective_risk(
    *, risk_per_trade: float, multiplier: float, cap: float
) -> float:
    """Runtime hard-clamp on confidence-scaled risk — the final authority.

    ``effective_risk = min(risk_per_trade * multiplier, cap)``. The cap is
    applied AFTER scaling so a confidence multiplier (e.g. 1.5x on a 0.15
    base) can never breach the inviolable risk cap (0.2). The config's
    ``risk_per_trade`` is validated ``le=0.2`` at *load* time only; this clamp
    is the *runtime* guarantee.
    """
    return min(risk_per_trade * multiplier, cap)


def post_rounding_guard(
    *,
    raw_qty: float,
    min_qty: float,
    sl_distance: float | None,
    bankroll: float,
    risk_per_trade: float,
    tol: float,
    enforce_risk_cap: bool,
    allow_min_lot_overshoot: bool,
) -> tuple[float, str | None]:
    """Decide the effective trade quantity and whether it breaches the risk cap.

    Returns ``(effective_qty, breach_reason)``. ``breach_reason`` is None when
    the trade is safe to send; otherwise it is one of:

    * ``"below_min_lot"`` — the risk-sized qty is below the broker min and the
      user has not enabled min-lot overshoot; the orchestrator skips.
    * ``"risk_cap_breach_overshoot"`` — min-lot overshoot is allowed and would
      bump qty up to ``min_qty``, but the implied risk exceeds
      ``risk_per_trade * (1 + tol)``; skip to keep the cap inviolable.
    * ``"risk_cap_breach"`` — the (already >= min) quantity's implied risk
      exceeds the cap (e.g. confidence-scaled sizing inflated it); skip.

    When ``breach_reason`` is None, ``effective_qty`` is the magnitude to send
    (``raw_qty``, or ``min_qty`` when overshoot-bumped). The caller applies
    direction.
    """
    magnitude = abs(raw_qty)
    if magnitude < min_qty:
        if not allow_min_lot_overshoot:
            return magnitude, "below_min_lot"
        effective = min_qty
        if enforce_risk_cap and sl_distance is not None and sl_distance > 0:
            if implied_risk(effective, sl_distance, bankroll) > risk_per_trade * (1.0 + tol):
                return effective, "risk_cap_breach_overshoot"
        return effective, None
    # magnitude >= min_qty: the broker floors DOWN to the lot step, so implied
    # risk only shrinks for the plain risk-sized path. The defensive check
    # still matters once confidence-scaled sizing inflates the intended qty.
    if enforce_risk_cap and sl_distance is not None and sl_distance > 0:
        if implied_risk(magnitude, sl_distance, bankroll) > risk_per_trade * (1.0 + tol):
            return magnitude, "risk_cap_breach"
    return magnitude, None


def atr_sl_tp(
    *,
    close: float,
    atr: float,
    atr_sl_mult: float,
    max_sl_pct: float,
    rr_ratio: float,
    direction: float,
) -> tuple[float | None, float | None, float]:
    """ATR-based per-signal SL/TP, capped to ``max_sl_pct`` of price.

    Returns ``(sl_price, tp_price, sl_distance)``. For a short, SL is above
    entry and TP below; for a long, SL below and TP above. When ``direction``
    is 0 (flat), returns ``(None, None, 0.0)``. The stop distance is capped to
    ``max_sl_pct`` of close so a volatile bar cannot produce an absurdly wide
    stop that collapses risk-sized quantity below the min lot.
    """
    if direction == 0 or close <= 0:
        return None, None, 0.0
    atr = atr if atr and math.isfinite(atr) and atr > 0 else close * 0.001
    sl_distance = min(atr_sl_mult * atr, close * max_sl_pct)
    if sl_distance <= 0:
        sl_distance = close * max_sl_pct
    if direction < 0:  # short: SL above, TP below
        sl_price = close + sl_distance
        tp_price = close - rr_ratio * sl_distance
    else:  # long: SL below, TP above
        sl_price = close - sl_distance
        tp_price = close + rr_ratio * sl_distance
    return sl_price, tp_price, sl_distance


def flip_to_flat_target(
    *, current_signed: float, direction: float, attach_stops: bool
) -> float:
    """Position-flip protection target quantity.

    When ``attach_stops`` is True and the signal flips AGAINST an open
    position, close to flat (target_qty = 0.0) rather than reversing in a
    single market order. A reversal rides the wrong way until the next flip
    and is the main source of whipsaw churn and oversized losses. Otherwise
    the caller's normal target (the signal's raw qty) is used; this helper
    only returns 0.0 when the flip-to-flat condition holds, else returns the
    passed-in ``current_signed`` sentinel meaning "no override" — see usage in
    the orchestrator, which calls this only to detect the flip condition.
    """
    if (
        attach_stops
        and current_signed != 0.0
        and direction != 0.0
        and (direction > 0) != (current_signed > 0)
    ):
        return 0.0
    return current_signed  # no override (caller checks for the 0.0 sentinel)


def stop_exit_price(
    *, side_is_long: bool, sl_price: float, tp_price: float,
    hit_sl: bool, hit_tp: bool,
) -> float:
    """Exit price for a software-stop close at the stop LEVEL.

    The stop executes at the level where the attached SL/TP sits, not at the
    crossing price — using the crossing price would overstate the realized
    loss when the exchange already filled at the stop.
    """
    return sl_price if hit_sl else tp_price


def min_bankroll_to_clear_min_lot(
    *,
    min_qty: float,
    price: float,
    sl_distance_pct: float,
    risk_per_trade: float,
) -> float:
    """Smallest bankroll at which risk-sized qty clears the broker min lot.

    ``risk_qty = (risk_per_trade * bankroll) / (price * sl_distance_pct)``;
    solving for ``risk_qty >= min_qty`` gives
    ``bankroll >= min_qty * price * sl_distance_pct / risk_per_trade``. Used by
    the startup preflight to tell the user exactly how much stake unlocks each
    symbol. ``sl_distance_pct`` is the stop distance as a fraction of price
    (e.g. 0.02 for a 2% stop).
    """
    if risk_per_trade <= 0 or sl_distance_pct <= 0 or price <= 0:
        return float("inf")
    return (min_qty * price * sl_distance_pct) / risk_per_trade


def classify_symbol_risk_cap(
    *,
    symbol: str,
    bankroll: float,
    risk_per_trade: float,
    leverage: float,
    allocation: float,
    min_qty: float,
    price: float,
    sl_distance_pct: float,
) -> dict[str, object]:
    """Preflight classification for one symbol against the risk cap.

    Returns a dict with: ``symbol``, ``clears_min_lot`` (bool),
    ``risk_qty`` (the risk-sized qty at this bankroll), ``min_qty``,
    ``implied_risk_pct`` (the fraction of bankroll risked if the min-lot
    overshoot were used), ``min_bankroll_to_clear`` (the smallest stake that
    makes risk-sized qty clear the min lot), and ``verdict`` — one of
    ``"tradeable"``, ``"skip_below_min_lot"``, ``"skip_risk_cap_breach"``.
    """
    sl_distance = price * sl_distance_pct
    notional_cap_qty = (bankroll * leverage * allocation) / price if price > 0 else 0.0
    risk_qty = risk_size_qty(
        bankroll=bankroll, risk_per_trade=risk_per_trade,
        sl_distance=sl_distance, notional_cap_qty=notional_cap_qty,
    )
    min_bankroll = min_bankroll_to_clear_min_lot(
        min_qty=min_qty, price=price, sl_distance_pct=sl_distance_pct,
        risk_per_trade=risk_per_trade,
    )
    clears = risk_qty >= min_qty
    overshoot_implied = implied_risk(min_qty, sl_distance, bankroll) if bankroll > 0 else 0.0
    if clears:
        verdict: Literal["tradeable", "skip_below_min_lot", "skip_risk_cap_breach"] = "tradeable"
    elif overshoot_implied > risk_per_trade * 1.10:
        verdict = "skip_risk_cap_breach"
    else:
        verdict = "skip_below_min_lot"
    return {
        "symbol": symbol,
        "clears_min_lot": clears,
        "risk_qty": risk_qty,
        "min_qty": min_qty,
        "implied_risk_pct": overshoot_implied,
        "min_bankroll_to_clear": min_bankroll,
        "verdict": verdict,
    }


__all__ = [
    "atr_sl_tp",
    "clamp_effective_risk",
    "classify_symbol_risk_cap",
    "flip_to_flat_target",
    "implied_risk",
    "min_bankroll_to_clear_min_lot",
    "post_rounding_guard",
    "risk_size_qty",
    "stop_exit_price",
]
