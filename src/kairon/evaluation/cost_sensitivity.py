"""Cost sensitivity sweep (CAS at 0.5x, 1x, 2x, 5x cost).

Story W2.3 is the cost sensitivity shock required by
``evaluation_framework.md`` §8.4. The headline artifact the
runner script produces (``reports/cost_sensitivity_w2.md``)
reports the Sharpe, Sortino, max-drawdown, total-return, and
trade-count metrics for a 1mo BTCUSDT 1h synthetic equity
curve at four cost multipliers ``(0.5, 1.0, 2.0, 5.0)``.

The function is pure: it takes an equity curve, a
``base_round_trip_bps`` (in bps of notional), and an optional
``trade_pnl`` vector. For each cost multiplier it scales the
per-trade cost and re-builds a per-bar equity curve, then
returns the ``PerformanceReport`` at each multiplier.

Cost-shock semantics
--------------------

When ``trade_pnl`` is supplied, the per-trade PnL is shocked
by subtracting ``multiplier * base_round_trip_bps / 10000 *
notional_proxy`` from each entry (the spec uses
``notional_proxy = 1.0``, so the cost is in *return* units).
The per-bar equity curve is then reconstructed by
cumulative-summing the (now-shocked) per-trade returns and
exponentiating from the original starting equity. When
``trade_pnl`` is ``None`` the function falls back to the
equity-curve-only branch: the per-bar cost drag is applied as
a constant-fraction reduction of the per-bar mean return
(v1 simplification; the exact version is deferred).

Both branches preserve the *direction* of the cost
sensitivity (Sharpe decreases monotonically with cost for a
positive-Sharpe input), which is what the W2.3 acceptance
criterion pins with
``test_cost_sensitivity_reduces_sharpe_with_higher_cost``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kairon.backtest.metrics import PerformanceReport


# Default multipliers per the PRD W2.3 acceptance criterion.
# (0.5x, 1x, 2x, 5x) is the canonical cost-shock ladder: 0.5x
# is the "cheaper than current" tail (would the strategy
# survive if fees were halved?), 1x is the baseline, 2x and
# 5x are the "fee regime worsens" stress tests. The default
# is a tuple (not a list) so it is hashable in the returned
# ``dict`` and so
# ``cost_sensitivity_curve(multipliers=DEFAULT_MULTIPLIERS)``
# gives a deterministic ordering of the four rows in the
# headline markdown.
DEFAULT_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)

# Default baseline round-trip cost in bps. Matches
# ``kairon.backtest.cost.DEFAULT_CRYPTO_COSTS.round_trip_bps``
# (commission=10 + slippage=2 + half_spread=2, doubled for the
# round trip = 28 bps).
DEFAULT_BASE_ROUND_TRIP_BPS: float = 28.0

# Notional proxy used when shocking per-trade pnl. The
# per-trade cost drag is ``multiplier * base_round_trip_bps /
# 10000 * notional_proxy``. The PRD spec uses
# ``notional_proxy = 1.0`` so the cost is in *return* units
# (a 28 bps cost on a 1.0 notional is 0.0028 in returns).
NOTIONAL_PROXY: float = 1.0


def _build_equity_from_trade_pnl(
    trade_pnl: np.ndarray,
    *,
    initial_equity: float,
) -> np.ndarray:
    """Build a per-bar equity curve from a per-trade pnl vector.

    The equity curve is ``[initial_equity, initial_equity *
    (1 + pnl[0]), initial_equity * (1 + pnl[0]) * (1 + pnl[1]),
    ...]`` — i.e. each trade compounds on the previous one's
    mark. This is the canonical "per-trade compounding" model
    used in the W2.3 per-trade-pnl branch.
    """
    out: np.ndarray = np.empty(trade_pnl.size + 1, dtype=np.float64)
    out[0] = initial_equity
    # Use cumprod(1 + pnl) shifted by one to align with the
    # per-bar index: bar 0 is the initial equity, bar k is
    # initial_equity * prod_{i<k} (1 + pnl_i).
    factors: np.ndarray = 1.0 + trade_pnl
    out[1:] = initial_equity * np.cumprod(factors)
    return out


def cost_sensitivity_curve(
    equity_curve: np.ndarray,
    *,
    base_round_trip_bps: float,
    trade_pnl: np.ndarray | None = None,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    bars_per_year: int = 8760,
) -> dict[float, "PerformanceReport"]:
    """Re-compute a performance summary at each cost multiplier.

    Two branches:

    - If ``trade_pnl`` is supplied, the per-trade PnL is
      shocked by subtracting ``multiplier * base_round_trip_bps
      / 10000 * notional_proxy`` from each entry (with
      ``notional_proxy = 1.0``). The equity curve is then
      rebuilt from the shocked pnl via per-trade compounding.
    - If ``trade_pnl`` is ``None``, the per-bar cost overhead
      is approximated as a constant-fraction reduction of the
      per-bar mean return (v1 simplification): for multiplier
      ``k`` and base cost ``C`` (in bps), the per-bar mean
      return is scaled by ``(1 - k * C / 10000)`` while the
      per-bar volatility is held constant.

    Parameters
    ----------
    equity_curve
        1-D ``np.ndarray`` of mark-to-market equity values,
        one per bar. Must have at least 2 elements.
    base_round_trip_bps
        Baseline round-trip cost in basis points of notional.
        Typical value: 28.0 (the W1.3 ``DEFAULT_CRYPTO_COSTS``
        round-trip value).
    trade_pnl
        Optional 1-D ``np.ndarray`` of per-trade realised PnL
        (in return units, where 0.001 == 10 bps). When
        supplied, the cost shock is applied per-trade
        (``trade_pnl[i] -= multiplier * base_round_trip_bps /
        10000 * NOTIONAL_PROXY``) and the equity curve is
        rebuilt from the shocked pnl. When ``None``, the
        v1 mean-scaling branch is used.
    multipliers
        Tuple of cost multipliers to sweep. Default
        ``(0.5, 1.0, 2.0, 5.0)``. Order is preserved in the
        returned dict's iteration order.
    bars_per_year
        Annualisation factor for Sharpe / Sortino. Default
        ``8760`` = ``BARS_PER_YEAR_1H`` from
        :mod:`kairon.backtest.metrics`.

    Returns
    -------
    dict[float, PerformanceReport]
        A ``{multiplier: PerformanceReport}`` dict. The
        ``PerformanceReport`` has the standard fields
        (``sharpe``, ``sortino``, ``max_drawdown``, etc.) and
        the multiplier is the key.

    Raises
    ------
    ValueError
        If ``equity_curve`` is not a 1-D ``np.ndarray``, has
        fewer than 2 elements, contains non-finite values;
        if ``base_round_trip_bps`` is non-finite or negative;
        if any multiplier is non-finite or non-positive; or
        if ``bars_per_year`` is not a positive int.
    """
    from kairon.backtest.metrics import summarize

    # --- input validation (fail fast) --------------------------
    if equity_curve.ndim != 1:
        raise ValueError(
            f"equity_curve must be 1-D, got ndim={equity_curve.ndim}"
        )
    if equity_curve.size < 2:
        raise ValueError(
            f"equity_curve must have at least 2 elements, got "
            f"size={equity_curve.size}"
        )
    if not np.all(np.isfinite(equity_curve)):
        raise ValueError(
            "equity_curve must contain only finite values"
        )
    if not (math.isfinite(base_round_trip_bps) and base_round_trip_bps >= 0):
        raise ValueError(
            f"base_round_trip_bps must be finite and >= 0, got "
            f"{base_round_trip_bps!r}"
        )
    if not multipliers:
        raise ValueError("multipliers must be a non-empty tuple")
    for m in multipliers:
        # Allow m=0.0: the no-cost baseline is a valid
        # multiplier in the sweep (the W2.3 acceptance
        # criterion #2 tests it explicitly).
        if not (math.isfinite(m) and m >= 0):
            raise ValueError(
                f"multiplier must be finite and >= 0, got {m!r}"
            )
    if not (bars_per_year > 0):
        raise ValueError(
            f"bars_per_year must be a positive int, got "
            f"{bars_per_year!r}"
        )

    initial_equity: float = float(equity_curve[0])

    # --- trade_pnl branch ---------------------------------------
    if trade_pnl is not None:
        if trade_pnl.ndim != 1:
            raise ValueError(
                f"trade_pnl must be 1-D, got ndim={trade_pnl.ndim}"
            )
        if trade_pnl.size < 1:
            raise ValueError(
                f"trade_pnl must have at least 1 element, got "
                f"size={trade_pnl.size}"
            )
        if not np.all(np.isfinite(trade_pnl)):
            raise ValueError(
                "trade_pnl must contain only finite values"
            )

        results: dict[float, PerformanceReport] = {}
        for m in multipliers:
            # Per-trade cost shock: subtract
            # multiplier * C / 10000 * notional_proxy from
            # each trade. NOTIONAL_PROXY=1.0 -> cost is in
            # return units. A 28-bps cost becomes 0.0028
            # in return space.
            cost_per_trade: float = (
                m * base_round_trip_bps / 1e4 * NOTIONAL_PROXY
            )
            shocked_pnl: np.ndarray = trade_pnl - cost_per_trade
            rebuilt_equity: np.ndarray = _build_equity_from_trade_pnl(
                shocked_pnl, initial_equity=initial_equity,
            )
            report: PerformanceReport = summarize(
                rebuilt_equity, bars_per_year=bars_per_year,
                trade_pnl=shocked_pnl,
            )
            results[m] = report
        return results

    # --- equity-only branch (v1 mean-scaling fallback) ----------
    # Cost is applied as a constant-fraction reduction of
    # the per-bar mean return; volatility is held constant.
    # This is the documented v1 simplification; the
    # per-trade pnl branch above is the spec's primary
    # semantics.
    original_returns: np.ndarray = np.diff(equity_curve) / equity_curve[:-1]
    mean_return: float = float(original_returns.mean())
    original_std: float = float(original_returns.std(ddof=0))
    if original_std == 0.0:
        std_for_rebuild: float = max(abs(mean_return) * 1e-9, 1e-12)
        z_scored: np.ndarray = np.zeros_like(original_returns)
    else:
        std_for_rebuild = original_std
        z_scored = (original_returns - mean_return) / original_std

    results = {}
    for m in multipliers:
        cost_per_bar: float = m * base_round_trip_bps / 1e4
        new_mean: float = mean_return - cost_per_bar
        if not math.isfinite(new_mean):
            raise ValueError(
                f"multiplier={m} produced non-finite new mean "
                f"({mean_return} - {cost_per_bar})"
            )
        scaled_returns: np.ndarray = z_scored * std_for_rebuild + new_mean
        rebuilt_equity: np.ndarray = np.empty(scaled_returns.size + 1)
        rebuilt_equity[0] = initial_equity
        rebuilt_equity[1:] = (
            initial_equity * np.cumprod(1.0 + scaled_returns)
        )
        report = summarize(
            rebuilt_equity, bars_per_year=bars_per_year,
        )
        results[m] = report
    return results


__all__ = [
    "DEFAULT_BASE_ROUND_TRIP_BPS",
    "DEFAULT_MULTIPLIERS",
    "NOTIONAL_PROXY",
    "cost_sensitivity_curve",
]
