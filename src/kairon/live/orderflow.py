"""Order-flow / microstructure features (Phase 4b, entry-timing axis).

Pure functions over a snapshot of the exchange order book — **no I/O** — so the
same code is unit-tested and reused by the live strategy. Bybit's
``get_orderbook`` returns ``{"result": {"b": [[price, size], ...], "a": [...]}}``
(see ``broker/bybit.py:_best_crossing_price``); this module takes the bid/ask
level lists in that shape and derives the microstructure summary used as an
extra confluence input.

Why this is a *separate axis* from win-rate (the honest framing):
* The Phase 4 universe backtest (``reports/scalping_edge_phase3.md``) showed the
  edge is SOL-mr_long-specific and that breadth / ensemble are refuted. Order
  flow does **not** raise win-rate on the historical bar store — there is no
  historical L2 book to backtest it against — so it cannot be validated offline
  the way the setup matrix was. It improves **entry timing** (is the book
  absorbing the fade or leaning into it?), a live-only signal. It therefore
  ships **opt-in / off-by-default**; the drift kill-switch is the out-of-sample
  guardrail if it is turned on.
"""

from __future__ import annotations

from dataclasses import dataclass

# Cap on depth_ratio so a near-empty ask side does not blow the ratio to inf
# and poison downstream confidence math. Beyond ~50x the book is effectively
# one-sided and the signal is saturated; larger values carry no information.
_DEPTH_RATIO_CAP = 50.0


@dataclass(frozen=True, slots=True)
class OrderFlowSnapshot:
    """Microstructure summary of one order-book snapshot.

    * ``mid`` — mid price (best_bid + best_ask) / 2.
    * ``spread_pct`` — (best_ask - best_bid) / mid, always >= 0.
    * ``imbalance`` — bid depth / (bid depth + ask depth) over the top levels,
      in [0, 1]. >0.5 = bid-heavy (support under price); <0.5 = ask-heavy.
    * ``depth_ratio`` — bid depth / ask depth, capped at ``_DEPTH_RATIO_CAP``.
    * ``bid_depth`` / ``ask_depth`` — summed size over the top levels.
    * ``best_bid`` / ``best_ask`` — top-of-book prices.
    """

    mid: float
    spread_pct: float
    imbalance: float
    depth_ratio: float
    bid_depth: float
    ask_depth: float
    best_bid: float
    best_ask: float


def _levels_to_floats(levels: list, depth: int) -> list[tuple[float, float]]:
    """Coerce Bybit ``[price, size]`` levels to ``(float, float)`` tuples.

    Tolerates either lists or tuples, missing size (treated as 0), and
    non-numeric strings (treated as 0). Stops at ``depth`` valid-or-zero rows.
    """
    out: list[tuple[float, float]] = []
    for lvl in levels[:depth]:
        try:
            price = float(lvl[0]) if len(lvl) > 0 else 0.0
            size = float(lvl[1]) if len(lvl) > 1 else 0.0
        except (TypeError, ValueError, IndexError):
            price, size = 0.0, 0.0
        out.append((price, size))
    return out


def compute_orderflow(
    bids: list, asks: list, depth: int = 5,
) -> OrderFlowSnapshot | None:
    """Derive an :class:`OrderFlowSnapshot` from Bybit bid/ask level lists.

    Returns ``None`` when the book is empty or crossed (no usable top-of-book),
    so callers can treat a missing snapshot as "no order-flow signal" rather
    than crashing on a thin testnet book.
    """
    b = _levels_to_floats(bids, depth)
    a = _levels_to_floats(asks, depth)
    # Top-of-book = first level with a positive price on each side (skip any
    # leading garbage / zero-price levels Bybit occasionally surfaces).
    best_bid = next((p for p, _ in b if p > 0.0), 0.0)
    best_ask = next((p for p, _ in a if p > 0.0), 0.0)
    if best_bid <= 0.0 or best_ask <= 0.0 or best_ask < best_bid:
        return None

    mid = (best_bid + best_ask) / 2.0
    spread_pct = (best_ask - best_bid) / mid if mid > 0 else 0.0

    bid_depth = sum(sz for _, sz in b if sz > 0)
    ask_depth = sum(sz for _, sz in a if sz > 0)
    total = bid_depth + ask_depth
    imbalance = (bid_depth / total) if total > 0 else 0.5
    # Guard the one-sided cases: empty ask -> ratio saturated bid-heavy; empty
    # bid -> ratio saturated ask-heavy (use a small floor, not 0, so the ratio
    # stays finite and the alignment signal is unambiguous).
    if ask_depth > 0:
        depth_ratio = min(bid_depth / ask_depth, _DEPTH_RATIO_CAP)
    elif bid_depth > 0:
        depth_ratio = _DEPTH_RATIO_CAP
    else:
        depth_ratio = 1.0

    return OrderFlowSnapshot(
        mid=mid,
        spread_pct=spread_pct,
        imbalance=imbalance,
        depth_ratio=depth_ratio,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        best_bid=best_bid,
        best_ask=best_ask,
    )


def orderflow_alignment(snapshot: OrderFlowSnapshot, direction: float) -> float:
    """How aligned the book is with a trade of the given direction.

    * ``direction > 0`` (long / mr_long bounce): bid-heavy book supports the
      bounce; returns a value in ``(-1, 1)`` where >0 = supportive.
    * ``direction < 0`` (short / mr_short fade): ask-heavy book supports the
      fade; sign flipped accordingly.
    * ``direction == 0``: 0.0 (no signal).

    The magnitude is ``2 * |imbalance - 0.5|`` scaled so a perfectly one-sided
    book (imbalance 1.0 or 0.0) maps to +/-1.0 and a balanced book maps to 0.0.
    """
    if direction == 0.0:
        return 0.0
    # imbalance in [0,1]; deviation from 0.5 in [0, 0.5]; *2 -> [0,1].
    raw = 2.0 * (snapshot.imbalance - 0.5)
    return raw if direction > 0 else -raw


__all__ = ["OrderFlowSnapshot", "compute_orderflow", "orderflow_alignment"]
