"""Latency simulation for the paper trading engine (W7.1).

The paper trader must answer "how long does it take between the
model emitting a signal and the order actually hitting the book?"
The honest answer is: it depends on the venue, the order type,
and the time of day. For the v1 paper trader we use a
**lognormal** latency model — a well-known heavy-tailed wait-time
distribution in queueing theory that captures the "most fills
arrive in ~50ms but a long tail exists" reality of crypto
exchange round-trip times.

The simulator is a thin wrapper around ``numpy.random.RandomState``
(``seed``-seeded, deterministic) that draws from a
``lognormal(mean_ms=50, sigma=0.5, max_ms=500)`` distribution and
clamps the output to ``max_ms`` to keep tail events bounded.

Why lognormal?
--------------
Lognormal latency matches both (a) the empirical observation that
most exchange round-trips are <100ms and (b) the heavy-tail
behaviour (a 1-in-1000 round-trip can be 5-10x the median) that
the existing backtest engine does NOT model. A sizer / order
router that is calibrated to a deterministic 50ms latency will
systematically under-estimate tail-event slippage.

Why clamp at max_ms?
--------------------
The v1 paper trader is a research tool, not a production
exchange connector. A 30-second round-trip (a frozen WebSocket
or a rate-limit timeout) is a *connectivity* event, not a
*latency* event; the connectivity layer belongs in
``kairon.live``. Clamping at ``max_ms`` keeps the simulated
latency in the regime the model is calibrated for. The plan
documents the 500ms cap as the v1 contract.

Why a ``RandomState`` (not the newer ``Generator``)?
----------------------------------------------------
``numpy.random.RandomState`` is the *legacy* API. The plan
specifies it explicitly so the simulator is deterministic
across numpy versions. The W7.1 acceptance criterion
``test_deterministic_seeded`` requires that two simulators
constructed with the same seed produce the *identical*
sequence; both the ``RandomState`` and ``Generator`` paths
satisfy this, but the plan pins the legacy API for
forward-compat with the W3-era fixtures that already use it.

The module is pure: no IO, no async, no global state. The
``LatencyConfig`` is a frozen dataclass (per the project's
strict-typed style); the ``LatencySimulator`` is a stateful
``RandomState`` consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default lognormal parameters. The v1 paper trader uses these
# defaults; the tests construct ``LatencyConfig`` with the same
# values explicitly so the v1 contract is captured in the test
# surface (not just in the docstring).
#
# mean_ms=50 — median round-trip is 50ms (the v1 baseline).
# sigma=0.5   — modest heavy tail; a 99th-percentile round-trip is
#               ~exp(50ms * sigma*2.33) ≈ 158ms (the "5-sigma-equivalent
#               on the log scale" rule of thumb). With the max_ms clamp
#               the 99.9th percentile is 500ms.
# max_ms=500  — the v1 contract's upper bound; a 500ms+ round-trip is
#               treated as a connectivity event, not a latency event,
#               and is handled at the ingestion layer.
DEFAULT_MEAN_MS: float = 50.0
DEFAULT_SIGMA: float = 0.5
DEFAULT_MAX_MS: float = 500.0


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    """Configuration for the :class:`LatencySimulator`.

    All defaults match the v1 paper-trader baseline:

    - ``mean_ms=50.0`` — the median round-trip latency in ms.
    - ``sigma=0.5``    — the lognormal scale; 0.5 gives a 99th
                          percentile of ~158ms (no clamp).
    - ``max_ms=500.0`` — the hard upper bound; the v1 contract
                          clamps the draw to ``min(draw, max_ms)``
                          so a frozen-WS timeout does not leak
                          into the latency model.
    - ``seed``         — the seed for the underlying
                          :class:`numpy.random.RandomState`.
                          Two ``LatencyConfig`` instances with the
                          same ``seed`` produce the same draw
                          sequence.
    """

    mean_ms: float = DEFAULT_MEAN_MS
    sigma: float = DEFAULT_SIGMA
    max_ms: float = DEFAULT_MAX_MS
    seed: int = 20260608
    extras: dict[str, Any] = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        if self.mean_ms <= 0.0:
            raise ValueError(f"mean_ms must be > 0, got {self.mean_ms!r}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma!r}")
        if self.max_ms <= 0.0:
            raise ValueError(f"max_ms must be > 0, got {self.max_ms!r}")
        if self.max_ms < self.mean_ms:
            raise ValueError(
                f"max_ms ({self.max_ms}) must be >= mean_ms ({self.mean_ms})"
            )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class LatencySimulator:
    """A deterministic, seed-driven lognormal latency simulator.

    The simulator wraps a :class:`numpy.random.RandomState`
    constructed from the ``seed`` field of the
    :class:`LatencyConfig`. The :meth:`sample` method draws one
    round-trip latency in ms; :meth:`sample_many` draws a batch
    of size ``n``.

    The draws are clamped to ``max_ms`` so the long-tail
    (a 30-second frozen-WS timeout) does not leak into the
    latency model. The clamp is the v1 contract; downstream
    consumers that need the raw, unclamped draw can use
    :meth:`sample_raw`.

    Usage::

        cfg = LatencyConfig(seed=42)
        sim = LatencySimulator(cfg)
        latency_ms = sim.sample()             # one draw
        latencies = sim.sample_many(1000)     # 1000 draws
    """

    def __init__(self, config: LatencyConfig | None = None) -> None:
        self.config: LatencyConfig = config or LatencyConfig()
        # The plan specifies a ``numpy.random.RandomState`` (the
        # legacy API). ``RandomState`` is the v1 contract; the
        # ``Generator`` API is the v2 path. The plan pins the
        # legacy API so the W7.1 simulator is forward-compat
        # with the W3-era fixtures that use ``RandomState``.
        self._rng: np.random.RandomState = np.random.RandomState(  # type: ignore[attr-defined]
            self.config.seed
        )

    @property
    def seed(self) -> int:
        """The seed used to initialise the underlying RNG."""
        return int(self.config.seed)

    def sample(self) -> float:
        """Draw a single round-trip latency in ms (clamped to ``max_ms``)."""
        return float(self._draw_clamped(1)[0])

    def sample_many(self, n: int) -> np.ndarray:  # type: ignore[type-arg]
        """Draw ``n`` round-trip latencies in ms (clamped to ``max_ms``)."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        return self._draw_clamped(n)

    def sample_raw(self) -> float:
        """Draw a single round-trip latency in ms (NOT clamped).

        Returns the raw, unclamped lognormal draw so callers that
        need the natural long-tail behaviour (e.g. the W7.4
        latency-aware sizer, future story) can use the
        unclamped distribution. The v1 contract is the *clamped*
        :meth:`sample`; the raw draw is a power-user surface.
        """
        return float(self._draw_unclamped(1)[0])

    # -- internals --------------------------------------------------------
    def _draw_unclamped(self, n: int) -> np.ndarray:  # type: ignore[type-arg]
        """Draw ``n`` lognormal samples WITHOUT the ``max_ms`` clamp.

        ``numpy.random.RandomState.lognormal`` parameterises the
        lognormal distribution as ``exp(N(mu, sigma^2))`` where
        ``mu`` is the *mean of the underlying normal* and
        ``sigma`` is the *std of the underlying normal*. The
        v1 contract is the pair ``(mean_ms, sigma)`` where
        ``mean_ms`` is the *median* of the resulting lognormal
        (i.e. ``exp(mu) = mean_ms``) and ``sigma`` is the
        underlying normal's std. This convention matches
        ``scipy.stats.lognorm`` with ``s=sigma`` and
        ``scale=mean_ms`` (so the W7.1 KS test in
        ``test_lognormal_shape`` can use scipy as the
        reference distribution).
        """
        mu: float = float(np.log(self.config.mean_ms))
        return self._rng.lognormal(mean=mu, sigma=self.config.sigma, size=n)

    def _draw_clamped(self, n: int) -> np.ndarray:  # type: ignore[type-arg]
        """Draw ``n`` lognormal samples WITH the ``max_ms`` clamp."""
        raw: np.ndarray = self._draw_unclamped(n)
        return np.minimum(raw, self.config.max_ms)


__all__ = [
    "DEFAULT_MAX_MS",
    "DEFAULT_MEAN_MS",
    "DEFAULT_SIGMA",
    "LatencyConfig",
    "LatencySimulator",
]
