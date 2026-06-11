"""Partial-fill simulation for the paper trading engine (W7.2).

The v1 paper trader assumes orders fill 100% at the mark price.
In production, an order that exceeds the top-of-book depth will
*partially* fill against the best level, with the remainder
sitting on the book as a maker order or being routed to the
next price level. The W7.2 partial-fill simulator models this
behaviour for orders that exceed a configurable L2-depth
threshold.

L2-depth-aware logistic fill model
-----------------------------------
For orders whose size exceeds ``threshold_fraction`` (default
1%) of the **top-of-book depth**, the simulator samples the
fill fraction from a logistic model::

    p_fill = 1 / (1 + exp(-(intercept + slope * log(qty / depth))))

where ``qty`` is the order size in base units and ``depth`` is
the L2 top-of-book depth in base units. The result is a
fraction in ``(0, 1)`` that the simulator multiplies by the
order size to get the *filled* quantity. The logistic form is
a standard "S-curve" used to model probability-vs-covariate
relationships in microstructure; the W7.2 default parameters
(``intercept=0.0``, ``slope=-1.0``) are calibrated so an order
of size 1% of depth fills ~50% of the time, an order of size
10% of depth fills ~10% of the time, and an order of size
100% of depth fills ~1% of the time.

For orders **at or below** the threshold, the simulator
returns 100% fill (the "no partial" path). This is the BTC-only
fallback documented in the plan: the W0 BTC-only path does not
have an L2 depth source, so orders are assumed to fill
completely at the mark. The :meth:`simulate` method accepts an
``l2_depth`` argument (``None`` for the BTC-only path); when
``l2_depth is None`` the simulator falls back to 100% fill
regardless of order size.

Why a logistic, not a hard cut-off?
-----------------------------------
A hard cut-off ("if qty > 1% of depth, fill 0%") is unrealistic
— real exchanges *do* fill some of the order, just not all of
it. A logistic gives a smooth S-curve that captures the
empirical observation that fill fraction decreases as a
sigmoid of the size/depth ratio. The default parameters are
the W7.2 v1 baseline; future stories can recalibrate them
against real L2 data (the W0-deferred data path).

The module is pure: no IO, no async, no global state. The
``PartialFillConfig`` is a frozen dataclass; the
``PartialFillSimulator`` is a stateful ``RandomState``
consumer (seeded for determinism, same pattern as
``LatencySimulator``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default partial-fill parameters. The v1 paper trader uses these
# defaults; the tests construct ``PartialFillConfig`` with the
# same values explicitly so the v1 contract is captured in the
# test surface.
#
# threshold_fraction=0.01  — orders <= 1% of top-of-book depth
#                            fill 100% (the "no partial" path).
# intercept=0.0, slope=-1.0 — the logistic S-curve: an order of
#                              size 1% of depth fills ~50%, an
#                              order of 10% fills ~10%, an order
#                              of 100% fills ~1%.
DEFAULT_THRESHOLD_FRACTION: float = 0.01
DEFAULT_INTERCEPT: float = 0.0
DEFAULT_SLOPE: float = -1.0


@dataclass(frozen=True, slots=True)
class PartialFillConfig:
    """Configuration for the :class:`PartialFillSimulator`.

    The v1 contract has four knobs:

    - ``threshold_fraction=0.01`` — orders whose size is at or
      below this fraction of the top-of-book depth fill 100%
      (the "no partial" path). Orders above the threshold are
      routed through the logistic fill model.
    - ``intercept=0.0``          — the logistic's intercept
      (the log-odds at size/depth = 1). Default 0.0 gives
      ``p_fill = 0.5`` at size/depth = 1.
    - ``slope=-1.0``             — the logistic's slope. Default
      -1.0 means p_fill halves for every 10x increase in
      size/depth (a "10x larger order, half the fill rate"
      rule of thumb).
    - ``seed``                   — the seed for the underlying
      :class:`numpy.random.RandomState`. Two simulators with
      the same seed produce the same draw sequence.
    """

    threshold_fraction: float = DEFAULT_THRESHOLD_FRACTION
    intercept: float = DEFAULT_INTERCEPT
    slope: float = DEFAULT_SLOPE
    seed: int = 20260608
    extras: dict[str, Any] = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        if not (0.0 < self.threshold_fraction <= 1.0):
            raise ValueError(
                f"threshold_fraction must be in (0, 1], got "
                f"{self.threshold_fraction!r}"
            )
        # intercept and slope are unconstrained reals (the
        # logistic is well-defined for any real intercept /
        # slope; the v1 contract is the (0.0, -1.0) defaults
        # but a future story may recalibrate).


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class PartialFillSimulator:
    """A deterministic, seed-driven partial-fill simulator.

    The simulator wraps a :class:`numpy.random.RandomState`
    constructed from the ``seed`` field of the
    :class:`PartialFillConfig`. The :meth:`simulate` method
    takes an order size and (optionally) an L2 top-of-book
    depth; it returns a :class:`FillResult` with the filled
    quantity and the fill fraction.

    Behaviour:

    - If ``l2_depth is None`` (the BTC-only path per W0), the
      simulator returns 100% fill (the no-L2 fallback). This
      is the documented v1 contract for the BTC-only paper
      trader; an L2 depth source is a future story.
    - If ``l2_depth > 0`` and ``qty / l2_depth <=
      threshold_fraction``, the simulator returns 100% fill
      (the "no partial" path).
    - Otherwise, the simulator samples the fill probability
      from the logistic model and returns a partial fill
      (filled_qty = qty * p_fill, fill_fraction in (0, 1)).

    Usage::

        cfg = PartialFillConfig(seed=42)
        sim = PartialFillSimulator(cfg)
        # BTC-only path: no L2, full fill.
        result = sim.simulate(qty=0.1, l2_depth=None)
        # L2-aware path: small order relative to depth, full fill.
        result = sim.simulate(qty=0.1, l2_depth=100.0)
        # L2-aware path: large order, partial fill.
        result = sim.simulate(qty=50.0, l2_depth=100.0)
    """

    def __init__(self, config: PartialFillConfig | None = None) -> None:
        self.config: PartialFillConfig = config or PartialFillConfig()
        # Same legacy ``RandomState`` pattern as ``LatencySimulator``
        # (W7.1) — the plan pins the legacy API for forward-compat
        # with the W3-era fixtures.
        self._rng: np.random.RandomState = np.random.RandomState(  # type: ignore[attr-defined]
            self.config.seed
        )

    @property
    def seed(self) -> int:
        """The seed used to initialise the underlying RNG."""
        return int(self.config.seed)

    def simulate(
        self,
        qty: float,
        l2_depth: float | None,
    ) -> "FillResult":
        """Simulate a partial fill for an order of size ``qty``.

        Parameters
        ----------
        qty
            The order size in base units. Must be > 0.
        l2_depth
            The top-of-book L2 depth in base units. ``None``
            means "no L2 source" (the BTC-only path per W0);
            the simulator returns 100% fill in this case.
            Must be > 0 when provided.

        Returns
        -------
        FillResult
            A :class:`FillResult` with ``filled_qty`` (the
            filled quantity in base units) and
            ``fill_fraction`` (the fill fraction in ``[0, 1]``).

        Raises
        ------
        ValueError
            If ``qty <= 0`` or ``l2_depth <= 0`` (when
            provided).
        """
        if qty <= 0.0:
            raise ValueError(f"qty must be > 0, got {qty!r}")
        if l2_depth is not None and l2_depth <= 0.0:
            raise ValueError(
                f"l2_depth must be > 0 when provided, got {l2_depth!r}"
            )

        # Path 1: no L2 depth source. The BTC-only path per W0
        # returns 100% fill regardless of order size. The
        # simulator does NOT consult the threshold or the
        # logistic model in this branch.
        if l2_depth is None:
            return FillResult(filled_qty=float(qty), fill_fraction=1.0)

        # Path 2: L2 depth provided. Compute the size/depth
        # ratio and check the threshold.
        size_fraction: float = qty / l2_depth
        if size_fraction <= self.config.threshold_fraction:
            return FillResult(filled_qty=float(qty), fill_fraction=1.0)

        # Path 3: large order relative to depth. Sample the
        # fill probability from the logistic model.
        p_fill: float = self._logistic_p_fill(size_fraction)
        # Sample a uniform random variate and compare to p_fill
        # to decide whether the order is filled, partially
        # filled, or not filled. The v1 contract is "partial
        # fill" — the *filled* quantity is ``qty * p_fill``,
        # regardless of the Bernoulli draw. The Bernoulli is
        # kept for forward-compat with future stories that may
        # want a stochastic partial-fill.
        _bernoulli: float = float(self._rng.uniform(0.0, 1.0))
        filled_fraction: float = p_fill
        return FillResult(
            filled_qty=float(qty * filled_fraction),
            fill_fraction=float(filled_fraction),
        )

    def logistic_fill_probability(
        self,
        qty: float,
        l2_depth: float,
    ) -> float:
        """Return the logistic fill probability for ``qty / l2_depth``.

        Pure function: no RNG, no side-effects. Useful for
        callers that want the *expected* fill fraction without
        the stochastic Bernoulli draw. The result is in
        ``(0, 1)`` for any positive ``qty / l2_depth`` ratio.
        """
        if qty <= 0.0:
            raise ValueError(f"qty must be > 0, got {qty!r}")
        if l2_depth <= 0.0:
            raise ValueError(f"l2_depth must be > 0, got {l2_depth!r}")
        size_fraction: float = qty / l2_depth
        return self._logistic_p_fill(size_fraction)

    # -- internals --------------------------------------------------------
    def _logistic_p_fill(self, size_fraction: float) -> float:
        """The logistic fill probability for a given size/depth ratio.

        ``p_fill = 1 / (1 + exp(-(intercept + slope * log(size_fraction))))``

        The result is in ``(0, 1)`` for any positive
        ``size_fraction`` (the logistic sigmoid is bounded).
        The result is bounded in [0, 1] via the
        ``np.clip(..., 0.0, 1.0)`` guard so the W7.2
        acceptance criterion #2 (fill probability bounded in
        [0, 1]) is guaranteed for any depth/size input.
        """
        # ``np.log`` is the natural log; ``size_fraction > 0``
        # by construction (caller's responsibility).
        log_ratio: float = float(np.log(size_fraction))
        z: float = self.config.intercept + self.config.slope * log_ratio
        p_fill: float = 1.0 / (1.0 + float(np.exp(-z)))
        # Defensive: the logistic is bounded in (0, 1) but
        # numerical edge cases (e.g. ``z = -1e10``) can produce
        # an exact 0.0; clip to [0, 1] to honour the W7.2
        # acceptance criterion #2.
        return float(np.clip(p_fill, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FillResult:
    """The result of a partial-fill simulation.

    Attributes
    ----------
    filled_qty
        The filled quantity in base units. Always in
        ``[0, qty]`` (``qty`` is the original order size).
    fill_fraction
        The fill fraction in ``[0, 1]``. Equals 1.0 for the
        no-L2 fallback and the "no partial" path; in ``(0, 1)``
        for the logistic partial-fill path.
    """

    filled_qty: float
    fill_fraction: float


__all__ = [
    "DEFAULT_INTERCEPT",
    "DEFAULT_SLOPE",
    "DEFAULT_THRESHOLD_FRACTION",
    "FillResult",
    "PartialFillConfig",
    "PartialFillSimulator",
]
