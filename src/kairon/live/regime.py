"""Market-regime classification for the scalping strategy (data-discovered).

The 8-week testnet backtest (see ``memory/scalping-setup-edge-findings`` and
:mod:`kairon.backtest.scalping_engine`) showed the strategy's edge is
**mean-reversion**, and mean-reversion only pays in **ranges** — it bleeds in
trends (mr_short SOL 15m won 25% in trending slices, 70% mr_long in ranging
slices). This module classifies the current bar into a regime from indicators
the strategy already computes (ADX, Bollinger bandwidth, EMA slope) so the
setup-selection matrix can gate mean-reversion entries to ranging regimes and
trend-following entries to trending regimes.

Pure functions of their arguments — no I/O, no async — so they are shared by the
live strategy and the backtest (one implementation, no drift).

Thresholds are *data-discovered* from the per-setup edge breakdown:
    * ADX < 20  -> range (mean-reversion pays).
    * ADX > 25  -> trend (mean-reversion bleeds; trend-following *might* pay,
      though the backtest showed testnet trend-following has no edge either —
      the matrix default kills it regardless).
    * Bollinger bandwidth > 6% of price -> volatile (wide stops; MR allowed but
      the risk cap binds).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    """Coarse market regime for setup gating."""

    RANGE = "range"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    VOLATILE = "volatile"


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """Data-discovered regime thresholds (fraction of price / ADX units)."""

    range_adx: float = 20.0
    trend_adx: float = 25.0
    volatile_bb_width: float = 0.06


def classify_regime(
    *,
    adx: float,
    bb_width_pct: float,
    ema_slope: float,
    thresholds: RegimeThresholds | None = None,
) -> Regime:
    """Classify the current bar into a regime.

    ``ema_slope`` is the signed slope of the slow EMA (fast EMA minus slow EMA,
    as a fraction of price, or any consistent signed trend proxy) — positive =
    uptrend, negative = downtrend. ``bb_width_pct`` is the Bollinger bandwidth
    ``(upper - lower) / mid`` as a fraction. Volatility wins over the range/trend
    split when the bandwidth is extreme (a volatile regime breaks both MR and
    trend-following stops, so the caller treats it conservatively).
    """
    th = thresholds or RegimeThresholds()
    if bb_width_pct >= th.volatile_bb_width:
        return Regime.VOLATILE
    if adx <= th.range_adx:
        return Regime.RANGE
    if adx >= th.trend_adx:
        return Regime.TREND_UP if ema_slope >= 0 else Regime.TREND_DOWN
    # Between range_adx and trend_adx: weak/ambiguous — treat as range (MR still
    # allowed; trend-following is not).
    return Regime.RANGE


def mean_reversion_allowed(regime: Regime) -> bool:
    """Mean-reversion entries are allowed in ranges and volatile regimes.

    Disallowed in strong trends (ADX>=trend_adx) — that is where MR bleeds.
    """
    return regime in (Regime.RANGE, Regime.VOLATILE)


def trend_following_allowed(regime: Regime) -> bool:
    """Trend-following entries are allowed only in trending regimes.

    The backtest showed testnet trend-following has no edge, so the default
    setup matrix disables those setups regardless; this helper exists for an
    opt-in trend-following matrix.
    """
    return regime in (Regime.TREND_UP, Regime.TREND_DOWN)


__all__ = [
    "Regime",
    "RegimeThresholds",
    "classify_regime",
    "mean_reversion_allowed",
    "trend_following_allowed",
]
