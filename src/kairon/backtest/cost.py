"""Trading cost model.

The backtester needs an honest, configurable cost model to know what
"saw 60% accuracy" is really worth. The three things that make a
backtest realistic:

- **Commission**: per-side fixed-fee (or per-share for stocks).
- **Slippage**: market-impact model; default = linear in traded size.
- **Half-spread**: how much you cross the book on entry. For crypto
  perps this is the maker-taker spread of the venue; for stocks it's
  half the bid-ask.

The cost model is intentionally small — just the three numbers — so
it's trivial to A/B-test. A future model will add:
- Funding rate (perps, every 8h)
- Borrow fee (shorts)
- Latency-bounded stop-loss slippage
- Stale-quote guard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from kairon.backtest.impact import AlmgrenChrissModel


@dataclass(frozen=True, slots=True)
class CostModel:
    """Per-trade cost in *price-return* units (e.g. 0.0005 == 5 bps).

    All costs are per-side. For a round-trip you pay twice. Costs
    are stored as fractions of notional (0.0005 == 5 bps).

    Parameters
    ----------
    commission_bps
        Exchange / broker fee, in basis points of notional.
    slippage_bps
        Static slippage estimate. Add a *dynamic* slippage component
        in the backtester (size × impact coefficient) on top of this.
    half_spread_bps
        Half the bid-ask spread. We treat the entry fill as one
        half-spread inside the book, the exit as the other half.
    impact_coefficient
        Multiplier on |trade size| / ADV to add linear market impact.
    min_trade_bps
        Trades smaller than this (in bps of equity) are skipped — they
        cost more in round-trip fees than they're worth.
    impact_model
        Optional Almgren-Chriss market impact model. When provided
        **and** the caller passes ``adv`` and ``sigma`` to
        :meth:`total_cost`, an additional size-scaled impact term
        (in bps) is added on top of commission + slippage + spread.
        The default ``None`` keeps the legacy constant-bps behaviour
        for the existing 394 baseline tests.
    """

    commission_bps: float = 5.0
    slippage_bps: float = 2.0
    half_spread_bps: float = 3.0
    impact_coefficient: float = 0.0
    min_trade_bps: float = 1.0
    impact_model: "AlmgrenChrissModel | None" = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.commission_bps < 0:
            raise ValueError(f"commission_bps must be >= 0, got {self.commission_bps}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        if self.half_spread_bps < 0:
            raise ValueError(f"half_spread_bps must be >= 0, got {self.half_spread_bps}")
        if self.impact_coefficient < 0:
            raise ValueError(f"impact_coefficient must be >= 0, got {self.impact_coefficient}")
        if self.min_trade_bps < 0:
            raise ValueError(f"min_trade_bps must be >= 0, got {self.min_trade_bps}")
        if self.impact_model is not None:
            # Late import to avoid a circular dep: impact.py imports
            # nothing from cost.py, but cost.py pulling impact.py at
            # module load would create a reverse edge once the engine
            # grows to import both. TYPE_CHECKING is the static-type
            # side; this is the runtime side.
            from kairon.backtest.impact import AlmgrenChrissModel as _ACM

            # Runtime class identity check (not ``isinstance``, which
            # pyright flags as redundant when the field is statically
            # typed). Catches subclass confusion / duck-typed mistakes
            # when callers bypass static-type enforcement.
            if type(self.impact_model) is not _ACM:
                raise ValueError(
                    "impact_model must be an AlmgrenChrissModel or None, got "
                    f"{type(self.impact_model).__name__}"
                )

    @property
    def round_trip_bps(self) -> float:
        """Total cost of an entry+exit trade (in bps of notional)."""
        return 2.0 * (self.commission_bps + self.slippage_bps + self.half_spread_bps)

    def total_cost(
        self,
        notional: float,
        side: str,
        *,
        adv: float | None = None,
        sigma: float | None = None,
        price: float | None = None,
        qty: float | None = None,
    ) -> float:
        """Return the absolute cost of a single-side trade (in cash units).

        When ``impact_model`` is set on this ``CostModel`` and the
        caller supplies ``adv``, ``sigma``, ``price`` and ``qty``
        (the four inputs the Almgren-Chriss formula needs), an
        additional impact term in bps is added to the commission +
        slippage + half-spread total. Without all four of those
        inputs, the impact model is silently ignored so that the
        legacy 2-arg call site (``engine.py``) keeps producing
        byte-identical results.
        """
        if notional < 0:
            raise ValueError(f"notional must be >= 0, got {notional}")
        if side not in {"buy", "sell", "entry", "exit"}:
            raise ValueError(f"side must be buy/sell/entry/exit, got {side!r}")
        # Per-side fee + slippage + half-spread
        bps = self.commission_bps + self.slippage_bps + self.half_spread_bps
        # Optional size-scaled impact term (Almgren-Chriss).
        if (
            self.impact_model is not None
            and adv is not None
            and sigma is not None
            and price is not None
            and qty is not None
        ):
            # side mapping: entry/buy => "buy", exit/sell => "sell"
            impact_side = "buy" if side in ("buy", "entry") else "sell"
            bps += self.impact_model.compute_bps(
                price=price, qty=qty, adv=adv, sigma=sigma, side=impact_side
            )
        return notional * bps / 1e4

    def should_trade(self, expected_edge_bps: float) -> bool:
        """Round-trip cost filter: skip trades whose edge doesn't clear fees.

        Always return ``False`` for non-positive edges. Otherwise compare
        the expected edge in bps to the round-trip cost in bps.
        """
        if expected_edge_bps <= 0:
            return False
        return expected_edge_bps >= self.round_trip_bps


# Sensible defaults for the two main venues.
DEFAULT_CRYPTO_COSTS: Final[CostModel] = CostModel(
    commission_bps=10.0,   # 0.10% taker (Binance/Bybit default)
    slippage_bps=2.0,
    half_spread_bps=2.0,
    impact_coefficient=0.0,
    min_trade_bps=5.0,
)
DEFAULT_STOCK_COSTS: Final[CostModel] = CostModel(
    commission_bps=2.0,    # Interactive Brokers tiered
    slippage_bps=1.0,
    half_spread_bps=2.0,
    impact_coefficient=0.0,
    min_trade_bps=3.0,
)


__all__ = [
    "DEFAULT_CRYPTO_COSTS",
    "DEFAULT_STOCK_COSTS",
    "CostModel",
]
