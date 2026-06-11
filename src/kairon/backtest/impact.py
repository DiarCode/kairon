"""Almgren-Chriss market impact model (placeholder).

The W1.3 implementation ships a **pure-compute placeholder**. The
default impact coefficient ``eta = 0.5`` follows the Almgren 2005
convention and is intentionally **not** calibrated to any real venue.
Calibration happens in W2 from real ccxt public-trade prints; the
``is_calibrated`` flag flips to ``True`` only after a successful
calibration. Story 3.7 (W3-4) will re-run the meta-label pipeline
if the W2-calibrated ``eta`` drifts from this placeholder by more
than 2x.

The model is intentionally small (no IO, no async) so it is trivial
to A/B-test against the existing constant-bps cost model. The
formula is the square-root impact model from Almgren & Chriss
(2000/2005):

    impact = eta * sigma * sqrt(qty / adv) * price

That is the temporary price excursion (in *price units*) caused by
exECUTING a parent order of size ``qty`` against average daily
volume ``adv``, scaled by the asset's recent volatility ``sigma``.
Side (``buy`` / ``sell``) is symmetric: real-world buy/sell
asymmetry comes from the bid-ask spread, which is already captured
by ``CostModel.half_spread_bps``. See ``docs/architecture.md``
section 7 ("cost-aware by default") and
``docs/evaluation_framework.md`` section 3 for context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# Default impact coefficient per Almgren (2005) convention. This is a
# pure-compute placeholder; W2 will overwrite it with a calibrated value
# per asset class and persist it to ``configs/cost/{symbol}.yaml``.
DEFAULT_ETA: float = 0.5

# Literal type for trade side. ``sell`` is symmetric to ``buy`` in this
# model (impact is a temporary excursion; the directional cost asymmetry
# lives in the spread component, not the impact term).
Side = Literal["buy", "sell"]

# Default regime-eta multipliers (W9.3). The cost model multiplies
# the calibrated ``eta`` by the regime's multiplier before the
# Almgren-Chriss formula. Trending and ranging are 1.0 (no
# adjustment); volatile is 1.2 (markets are slightly more
# impact-sensitive in volatile regimes); stressed is 1.5 (the
# market-impact amplification under stress, per the W9 audit
# panels' recommendations and the 4x sigma-shock literature).
DEFAULT_REGIME_ETA_MULTIPLIERS: dict[str, float] = {
    "trending": 1.0,
    "ranging": 1.0,
    "volatile": 1.2,
    "stressed": 1.5,
}


@dataclass(frozen=True, slots=True)
class AlmgrenChrissModel:
    """Square-root market impact model (Almgren & Chriss, 2005).

    Parameters
    ----------
    eta
        Impact coefficient. ``eta=0.5`` is the placeholder default
        (Almgren 2005 convention). W2 will overwrite this with a
        calibrated value per asset class.
    is_calibrated
        ``False`` for the W1.3 placeholder. Flips to ``True`` only
        after the W2 calibration pass over real ccxt public-trade
        prints. Consumers (backtest engine, evaluation harness, the
        cost-ML re-work loop in story 3.7) MUST treat
        ``is_calibrated=False`` output as a heuristic, not a
        measurement.

    Notes
    -----
    The model is deterministic and pure: same inputs, same output, no
    IO, no async. It is a ``frozen=True, slots=True`` dataclass (not
    a pydantic model) to match the project pattern in
    ``src/kairon/backtest/cost.py``. Validation lives in
    ``__post_init__`` so the type stays a plain dataclass.
    """

    eta: float = DEFAULT_ETA
    is_calibrated: bool = False
    # Regime-conditional multipliers (W9.3). ``compute()`` multiplies
    # the calibrated ``eta`` by ``regime_eta_multipliers[regime]``
    # before the Almgren-Chriss formula. Trending and ranging are
    # 1.0 (no adjustment); volatile is 1.2; stressed is 1.5
    # (the impact-amplification under stress, per the W9 audit
    # panels' recommendations). Callers can override the defaults
    # by passing a custom ``regime_eta_multipliers`` field.
    # The field is compared=False so additive changes don't break
    # legacy equality (matches the CostModel.impact_model pattern).
    regime_eta_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REGIME_ETA_MULTIPLIERS),
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.eta <= 0 or math.isnan(self.eta):
            raise ValueError(f"eta must be > 0 and finite, got {self.eta}")
        # Runtime guard: pyright/static-types blocks non-bool, but Python
        # does not enforce the annotation at construction time. ``is True
        # / is False`` is identity-precise (rejects 0/1 ints).
        if self.is_calibrated is not True and self.is_calibrated is not False:
            raise ValueError(
                f"is_calibrated must be a bool, got {self.is_calibrated!r}"
            )
        # Validate the multipliers: keys must be the 4 Regime values
        # (the W9.3 contract); each value must be > 0 and finite.
        expected_keys = {"trending", "ranging", "volatile", "stressed"}
        actual_keys = set(self.regime_eta_multipliers.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"regime_eta_multipliers keys must be exactly "
                f"{sorted(expected_keys)}, got {sorted(actual_keys)}"
            )
        for k, v in self.regime_eta_multipliers.items():
            if math.isnan(float(v)):
                raise ValueError(
                    f"regime_eta_multipliers[{k!r}] must be a finite "
                    f"number, got {v!r}"
                )
            if float(v) <= 0.0:
                raise ValueError(
                    f"regime_eta_multipliers[{k!r}] must be > 0, got {v}"
                )

    def _regime_multiplier(self, regime: str | None) -> float:
        """Return the multiplier for ``regime``; 1.0 if ``regime`` is None.

        Passing ``regime=None`` is the legacy path: callers that
        don't know the regime get the unmultiplied ``eta`` (the
        W1.3 contract). Passing an unknown regime string raises.
        """
        if regime is None:
            return 1.0
        if regime not in self.regime_eta_multipliers:
            raise ValueError(
                f"unknown regime {regime!r}; expected one of "
                f"{sorted(self.regime_eta_multipliers.keys())}"
            )
        return float(self.regime_eta_multipliers[regime])

    def compute(
        self,
        price: float,
        qty: float,
        adv: float,
        sigma: float,
        *,
        regime: str | None = None,
        side: Side = "buy",
    ) -> float:
        """Return the temporary price impact in **price units**.

        Formula: ``eta_eff * sigma * sqrt(qty / adv) * price``, where
        ``eta_eff = eta * regime_eta_multipliers[regime]`` (or
        ``eta`` if ``regime is None``). The multiplier is the W9.3
        regime-eta coupling: a stressed regime amplifies the impact
        by 1.5x (the documented default).

        Parameters
        ----------
        price
            Current mid-price of the asset (in price units; e.g. 50000
            for BTC). Must be > 0.
        qty
            Order size in base units (e.g. number of BTC). Must be > 0.
        adv
            Average daily volume in base units (e.g. BTC traded per
            day). Must be > 0.
        sigma
            Recent volatility of the asset (per-bar or per-day return
            std-dev; a 0.01 means 1% volatility). Must be >= 0.
        side
            ``"buy"`` or ``"sell"``. The impact magnitude is symmetric
            across sides in this model; the directional asymmetry in
            real markets lives in the spread component of
            ``CostModel``, not in the impact term.
        regime
            One of ``"trending"``, ``"ranging"``, ``"volatile"``,
            ``"stressed"``. When ``None`` (the legacy contract),
            the multiplier is 1.0 and ``compute()`` returns the
            unmultiplied impact. The W9.3 cost-regime coupling
            passes the BOCPD-detected regime label.

        Returns
        -------
        float
            Temporary price impact in price units (NOT bps). The
            caller can convert to bps via ``compute_bps`` or via
            ``10000 * impact / price``.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if math.isnan(price):
            raise ValueError(f"price must be a real number, got {price!r}")
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        if math.isnan(qty):
            raise ValueError(f"qty must be a real number, got {qty!r}")
        if qty <= 0:
            raise ValueError(f"qty must be > 0, got {qty}")
        if math.isnan(adv):
            raise ValueError(f"adv must be a real number, got {adv!r}")
        if adv <= 0:
            raise ValueError(f"adv must be > 0, got {adv}")
        if math.isnan(sigma):
            raise ValueError(f"sigma must be a real number, got {sigma!r}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        eta_eff: float = self.eta * self._regime_multiplier(regime)
        return eta_eff * sigma * math.sqrt(qty / adv) * price

    def compute_bps(
        self,
        price: float,
        qty: float,
        adv: float,
        sigma: float,
        *,
        regime: str | None = None,
        side: Side = "buy",
    ) -> float:
        """Return the temporary price impact in basis points (bps).

        Equivalent to ``10000 * compute(price, qty, adv, sigma, side=side, regime=regime) / price``
        with full input validation. Use this when the caller works in
        bps (e.g. when adding the impact term to ``CostModel.total_cost``,
        which is expressed in bps of notional).
        """
        impact_price = self.compute(price, qty, adv, sigma, regime=regime, side=side)
        return 10000.0 * impact_price / price


__all__ = [
    "DEFAULT_ETA",
    "DEFAULT_REGIME_ETA_MULTIPLIERS",
    "AlmgrenChrissModel",
    "Side",
]
