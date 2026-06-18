"""Vol-aware position sizing with a Kelly cap.

Story W6.5 ships the *vol-aware sizer* — a position-sizing
primitive that scales size to a fixed fraction of equity, modulated
by the model's predicted return magnitude and the realised (or
forecast) vol target. The vol-aware form is the load-bearing
contract per plan §W6.5: the sizer must scale linearly with the
predicted edge and inversely with the realised vol so the
risk-adjusted position size is stable across regimes.

Backwards-compatibility
-----------------------
The W6.5 sizer co-exists with the W3-era sizing helpers in
:mod:`kairon.portfolio` (``fixed_fraction_size``, ``kelly_size``,
``vol_target_size``, ``size_position``). The vol-aware function
here is *new* (additive) — it does NOT replace the W3 helpers. The
W3 helpers continue to be the call-site for the existing backtest
engine; the vol-aware function is the call-site for the W6.4
multi-head model + the W6.5 sizer pipeline (and the W6.5 max-drawdown
test, which is the load-bearing acceptance criterion).

Function
--------
:func:`size_position_vol_aware` is the single public function. It
takes an equity, a price, a predicted magnitude, a realised vol
target, and two caps (``kelly_cap`` and ``max_position_equity_fraction``).
The result is a position SIZE in *units of the asset* (notional /
price), suitable for direct submission to the
:class:`kairon.paper.PaperTrader` or :class:`kairon.backtest.engine`.

The two caps are independent: the sizer first computes the raw
fraction ``predicted_magnitude / realized_vol_target``, then caps
it at the smaller of ``kelly_cap`` and
``max_position_equity_fraction``. The result is the position's
fraction of equity; the final size is ``fraction * equity / price``.

This module is pure: no IO, no async, no global state. The
function is the W6.5 acceptance-criterion surface and the W6.5
max-drawdown test calls it once per bar.
"""

from __future__ import annotations

import math
from typing import Final

# Default Kelly cap. Per plan §W6.5: ``kelly_cap: float = 0.25``.
# The cap is a hard upper bound on the sizer's fraction-of-equity
# output, independent of the position's max-equity fraction. The
# Kelly cap is the W6.5 acceptance criterion #2: even an unbounded
# predicted_magnitude must result in ``size <= 0.25 * equity / price``.
DEFAULT_KELLY_CAP: Final[float] = 0.25

# Default max position equity fraction. Per plan §W6.5: this
# defaults to ``KaironSettings().max_position_equity_fraction`` =
# 0.20 (see ``kairon.config.__init__`` line 69). The function
# accepts the value as a required keyword argument so the sizer
# has no implicit dependency on the global settings object;
# callers (the W6.5 test, the paper trader) wire the value
# explicitly.
DEFAULT_MAX_POSITION_EQUITY_FRACTION: Final[float] = 0.20


def size_position_vol_aware(
    equity: float,
    price: float,
    *,
    predicted_magnitude: float,
    realized_vol_target: float,
    kelly_cap: float = DEFAULT_KELLY_CAP,
    max_position_equity_fraction: float = DEFAULT_MAX_POSITION_EQUITY_FRACTION,
    direction: float | None = None,
) -> float:
    """Vol-aware position sizing with a Kelly cap and optional direction.

    Computes

        fraction = predicted_magnitude / realized_vol_target
        fraction = clip(fraction, 0, min(kelly_cap, max_position_equity_fraction))
        size     = sign(direction) * fraction * equity / price

    The two caps are independent guards:

    - ``kelly_cap`` is the *theoretical* upper bound on the
      fraction-of-equity (Kelly's rule says never bet more than
      the Kelly fraction; we default to 0.25 per plan §W6.5).
    - ``max_position_equity_fraction`` is the *operational* upper
      bound (the Kairon settings cap; defaults to 0.20 per
      ``KaironSettings().max_position_equity_fraction``).

    The effective cap is ``min(kelly_cap,
    max_position_equity_fraction)``; the sizer clips the
    fraction-of-equity to that bound. A negative
    ``predicted_magnitude`` is clipped to zero (the magnitude is
    always interpreted as an unsigned edge). When ``direction`` is
    supplied, the returned size is signed: negative for short
    signals (``direction < 0``). When ``direction`` is ``None``,
    the legacy long-only behavior is preserved (size is always
    non-negative).

    Parameters
    ----------
    equity
        Current account equity (cash + mark-to-market). Must be
        > 0.
    price
        The asset's current price. Must be > 0.
    predicted_magnitude
        The model's predicted return magnitude (e.g. the W6.4
        magnitude head's output, in return units). Must be
        finite; negative values are clipped to 0.
    realized_vol_target
        The realised vol target for the position-sizing rule
        (e.g. a 1h realised vol for a 1h horizon). Must be > 0.
    kelly_cap
        The Kelly cap, in fraction-of-equity units. Default
        ``0.25`` (plan §W6.5 default).
    max_position_equity_fraction
        The operational cap on the position's fraction of
        equity, in fraction-of-equity units. Default
        ``0.20`` (the Kairon settings default).
    direction
        Optional signed signal direction. If ``direction < 0`` the
        returned size is negative (short). If ``None`` or
        ``direction >= 0`` the returned size is non-negative.

    Returns
    -------
    float
        The position size in *units of the asset* (notional /
        price). Signed when ``direction`` is supplied: negative
        for short signals. Always finite. Equals
        ``sign(direction) * fraction * equity / price`` where
        ``fraction = min(max(predicted_magnitude /
        realized_vol_target, 0), min(kelly_cap,
        max_position_equity_fraction))``.

    Raises
    ------
    ValueError
        If ``equity <= 0``, ``price <= 0``, ``realized_vol_target
        <= 0``, ``kelly_cap <= 0`` or ``kelly_cap > 1``, or
        ``max_position_equity_fraction <= 0`` or
        ``max_position_equity_fraction > 1``. Non-finite
        ``predicted_magnitude`` also raises.
    """
    if not (math.isfinite(equity) and equity > 0):
        raise ValueError(f"equity must be > 0, got {equity!r}")
    if not (math.isfinite(price) and price > 0):
        raise ValueError(f"price must be > 0, got {price!r}")
    if not (math.isfinite(realized_vol_target) and realized_vol_target > 0):
        raise ValueError(
            f"realized_vol_target must be > 0, got {realized_vol_target!r}"
        )
    if not (math.isfinite(kelly_cap) and 0.0 < kelly_cap <= 1.0):
        raise ValueError(
            f"kelly_cap must be in (0, 1], got {kelly_cap!r}"
        )
    if not (
        math.isfinite(max_position_equity_fraction)
        and 0.0 < max_position_equity_fraction <= 1.0
    ):
        raise ValueError(
            f"max_position_equity_fraction must be in (0, 1], got "
            f"{max_position_equity_fraction!r}"
        )
    if not math.isfinite(predicted_magnitude):
        raise ValueError(
            f"predicted_magnitude must be finite, got {predicted_magnitude!r}"
        )

    # The effective cap is the tighter of the two independent
    # guards. A function that always takes the smaller cap is
    # the documented v1 contract; the W6.5 acceptance criteria
    # 1 (max_position_equity_fraction cap) and 2 (kelly_cap)
    # each pin ONE of the two caps in isolation, so the
    # intersection must be applied here.
    effective_cap: float = min(kelly_cap, max_position_equity_fraction)

    # Raw fraction: predicted_magnitude / realized_vol_target.
    # Magnitude is unsigned; direction determines the sign of the output.
    raw_fraction: float = max(0.0, predicted_magnitude / realized_vol_target)

    # Clip the fraction to the effective cap.
    fraction: float = min(raw_fraction, effective_cap)

    size: float = fraction * equity / price
    if direction is not None and direction < 0:
        size = -size

    return float(size)


__all__ = [
    "DEFAULT_KELLY_CAP",
    "DEFAULT_MAX_POSITION_EQUITY_FRACTION",
    "size_position_vol_aware",
]
