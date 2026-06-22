"""Setup-selection matrix — which setups fire, gated by regime (data-discovered).

The 8-week testnet backtest per-setup edge breakdown
(``memory/scalping-setup-edge-findings``) showed:

* **mean-reversion** (mr_short, mr_long) is the only edge — keep.
* **momentum trend-following** (momentum_short, momentum_long) has single-digit
  win rates and -1R expectancy — kill.
* **breakout / breakdown** (volume-surge) is negative on testnet (volume rarely
  surges) — kill.

This module encodes that as a frozen :class:`SetupMatrix` plus a regime gate, an
exhaustion guard, an MTF bias flag, and a confidence-calibration flag. The
matrix is **opt-in**: ``ScalpingStrategy(setup_matrix=None)`` keeps the legacy
"all setups fire" behaviour byte-for-byte (existing tests unchanged); passing
:data:`MEAN_REVERSION_ONLY` applies the data-recommended selectivity.

Pure data — no I/O — so it is shared by the live strategy and the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

from kairon.live.regime import Regime, mean_reversion_allowed, trend_following_allowed

# Canonical setup ids (match the engine's _setup_id_from_justifications tags).
SETUP_IDS: tuple[str, ...] = (
    "mr_short",
    "mr_long",
    "momentum_short",
    "momentum_long",
    "breakdown",
    "breakout",
)


@dataclass(frozen=True, slots=True)
class SetupMatrix:
    """Per-setup enable flags + regime/exhaustion/MTF/calibration gates.

    Defaults preserve the legacy strategy (all setups enabled, no gates) so
    ``ScalpingStrategy()`` behaviour is unchanged when no matrix is passed.
    """

    # Per-setup enable flags.
    enable_mr_short: bool = True
    enable_mr_long: bool = True
    enable_momentum_short: bool = True
    enable_momentum_long: bool = True
    enable_breakdown: bool = True
    enable_breakout: bool = True
    # Gates (default off = legacy behaviour).
    regime_gate: bool = False
    exhaustion_guard: bool = False
    mtf_bias: bool = False
    confidence_calibration: bool = False
    # Runtime risk cap for confidence-scaled sizing (the inviolable ceiling; the
    # config's risk_per_trade is validated le=0.2 at load time, this is the
    # runtime clamp via kairon.live.pure_fns.clamp_effective_risk).
    risk_cap: float = 0.2

    def enabled(self, setup_id: str) -> bool:
        return {
            "mr_short": self.enable_mr_short,
            "mr_long": self.enable_mr_long,
            "momentum_short": self.enable_momentum_short,
            "momentum_long": self.enable_momentum_long,
            "breakdown": self.enable_breakdown,
            "breakout": self.enable_breakout,
        }.get(setup_id, False)

    def regime_ok(self, setup_id: str, regime: Regime) -> bool:
        """Regime gate: MR only in ranges/volatile; trend-following only in trends."""
        if not self.regime_gate:
            return True
        if setup_id in ("mr_short", "mr_long", "breakdown", "breakout"):
            # breakdown/breakout are structurally MR-adjacent on testnet (volume
            # confirmation is noise), so gate them as MR (ranges only).
            return mean_reversion_allowed(regime)
        if setup_id in ("momentum_short", "momentum_long"):
            return trend_following_allowed(regime)
        return True

    def allowed(self, setup_id: str, regime: Regime) -> bool:
        """A setup fires only if enabled AND regime-allowed."""
        return self.enabled(setup_id) and self.regime_ok(setup_id, regime)


# Data-recommended preset: mean-reversion only, with all gates on.
MEAN_REVERSION_ONLY = SetupMatrix(
    enable_mr_short=True,
    enable_mr_long=True,
    enable_momentum_short=False,
    enable_momentum_long=False,
    enable_breakdown=False,
    enable_breakout=False,
    regime_gate=True,
    exhaustion_guard=True,
    mtf_bias=True,
    confidence_calibration=True,
)


# Phase 4 data-driven tightening: mr_long only. The universe backtest
# (``scripts/analyze_symbol_edge.py`` across BTC/ETH/LINK/SOL/XRP x 5m/15m)
# showed mr_short is a *universal* loser on testnet — SOL 5m 28%, XRP 5m 45%,
# LINK 5m 42%, all negative PnL — while mr_long is the only edge (SOL 5m 78%,
# SOL 15m 83%). Killing mr_short removes the drag: SOL 5m blended +20.32 ->
# mr_long-only +29.97 (78% win). This is the honest win-rate lever — selectivity
# by killing the losing side, not by tightening floors (which would also starve
# the winning mr_long). Long-only contradicts the original "short-tilted" tilt,
# but the testnet data overrides it: on this venue, shorting mean-reversion loses.
LONG_ONLY = SetupMatrix(
    enable_mr_short=False,
    enable_mr_long=True,
    enable_momentum_short=False,
    enable_momentum_long=False,
    enable_breakdown=False,
    enable_breakout=False,
    regime_gate=True,
    exhaustion_guard=True,
    mtf_bias=True,
    confidence_calibration=True,
)


__all__ = [
    "LONG_ONLY",
    "MEAN_REVERSION_ONLY",
    "SETUP_IDS",
    "SetupMatrix",
]
