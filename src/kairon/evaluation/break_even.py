"""Per-trade break-even accuracy formula.

Story W2.2 publishes the headline viability metric: for a
trade with a given expected move ``R`` (in bps of price) and a
round-trip cost ``C`` (in bps of notional), the minimum
classification accuracy required for the trade to be
expectation-positive is

    p* = 0.5 + C / (2R)

The formula comes from the symmetric-bet decomposition of a
binary trade: with edge ``+R`` on a correct call and ``-R`` on
an incorrect call, the break-even accuracy ``p*`` is the
accuracy at which expected return equals the round-trip cost
``C``:

    p* * R - (1 - p*) * R = C
    R * (2p* - 1) = C
    p* = 0.5 + C / (2R)

When ``R`` is exactly the expected absolute return ``E[|r|]``
(per the plan: the conservative annualized-to-horizon move
``sigma * sqrt(252 * 24 * 60 / seconds_per_bar)``), the result
is the "per-trade break-even accuracy" that the W2.5 GO/NO-GO
gate reads out of ``reports/break_even_w2.md``.

The function is pure: no IO, no async, no global state. It is
the load-bearing primitive for the W2.2 break-even table and
for the cost-sensitivity shock in W2.3.
"""

from __future__ import annotations


def break_even_accuracy(
    *,
    expected_move_bps: float,
    round_trip_cost_bps: float,
) -> float:
    """Return the per-trade break-even accuracy as a fraction in (0.5, 1.0].

    Implements ``p* = 0.5 + C / (2R)`` where ``C`` is the round-trip
    cost in bps and ``R`` is the expected move magnitude in bps.
    The result is a fraction in the open-lower / closed-upper
    half-open interval ``(0.5, 1.0]``: a perfect bet (``C=0``)
    needs accuracy ``> 0.5`` by an infinitesimal amount, and a
    bet that costs as much as it can possibly move (``C == R``)
    needs accuracy exactly ``1.0``.

    Parameters
    ----------
    expected_move_bps
        The expected move magnitude in basis points of price.
        For the W2.2 table this is the conservative annualized-
        to-horizon move ``sigma * sqrt(252 * 24 * 60 / seconds_per_bar)``
        where ``sigma`` is the per-bar return standard deviation.
        Must be > 0.
    round_trip_cost_bps
        The round-trip trading cost in basis points of notional
        (commission + slippage + half-spread for entry AND exit;
        see :attr:`kairon.backtest.cost.CostModel.round_trip_bps`).
        Must be >= 0.

    Returns
    -------
    float
        The break-even accuracy in the open-lower / closed-upper
        interval ``(0.5, 1.0]``. Values > 1.0 are impossible in
        the cost model (they would mean the trade costs more than
        it can possibly move); the function returns ``1.0`` as the
        saturating upper bound so downstream tables do not need
        a separate saturation check. The clamp is documented and
        the function does NOT raise on saturation.

    Raises
    ------
    ValueError
        If ``expected_move_bps`` is not > 0 (the formula is
        undefined at ``R=0`` and the symmetric-bet decomposition
        collapses) or if either argument is non-finite. The
        validation matches the style of
        :meth:`kairon.backtest.impact.AlmgrenChrissModel.compute`
        and the existing per-trade validators in
        :mod:`kairon.evaluation.eta_calibration`.
    """
    import math

    if not (math.isfinite(expected_move_bps) and math.isfinite(round_trip_cost_bps)):
        raise ValueError(
            "expected_move_bps and round_trip_cost_bps must be finite, got "
            f"expected_move_bps={expected_move_bps!r}, "
            f"round_trip_cost_bps={round_trip_cost_bps!r}"
        )
    if expected_move_bps <= 0:
        # The formula is undefined at R=0 and the symmetric-bet
        # decomposition collapses. We refuse rather than return
        # +inf, which would silently mark every trade as
        # unviable and poison the W2.5 GO/NO-GO gate.
        raise ValueError(
            f"expected_move_bps must be > 0, got {expected_move_bps!r}"
        )
    if round_trip_cost_bps < 0:
        raise ValueError(
            f"round_trip_cost_bps must be >= 0, got {round_trip_cost_bps!r}"
        )

    raw: float = 0.5 + round_trip_cost_bps / (2.0 * expected_move_bps)
    # Saturate at 1.0: a trade whose cost equals its move
    # magnitude has p* = 1.0 exactly; a trade whose cost EXCEEDS
    # its move magnitude has p* > 1.0, which is impossible in
    # the cost model. Clamping to 1.0 keeps the table finite
    # and signals "the trade is not viable" to the viable
    # threshold (viable iff p* <= 0.6).
    if raw > 1.0:
        return 1.0
    return raw


__all__ = [
    "break_even_accuracy",
]
