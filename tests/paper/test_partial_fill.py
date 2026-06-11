"""Tests for the W7.2 partial-fill simulator.

Two tests pin the W7.2 acceptance criteria:

1. ``test_l2_required`` — when no L2 depth data is provided
   (``l2_depth=None``), the simulator falls back to 100% fill
   regardless of order size. This is the BTC-only path per W0
   (no L2 depth source).
2. ``test_logistic_fill_probability_bounded`` — the logistic
   fill probability is in ``[0, 1]`` for any positive
   depth/size input. The logistic sigmoid is mathematically
   bounded in (0, 1), but the test guards against numerical
   edge cases (e.g. a very large negative ``z`` producing an
   exact 0.0).
"""

from __future__ import annotations

import numpy as np
import pytest

from kairon.paper.partial_fill import (
    DEFAULT_INTERCEPT,
    DEFAULT_SLOPE,
    DEFAULT_THRESHOLD_FRACTION,
    FillResult,
    PartialFillConfig,
    PartialFillSimulator,
)


# ---------------------------------------------------------------------------
# W7.2 acceptance criterion #1: no L2 depth -> 100% fill fallback
# ---------------------------------------------------------------------------
def test_l2_required() -> None:
    """Without L2 depth, fall back to 100% fill (the BTC-only path).

    The W0 BTC-only fallback does not provide an L2 depth
    source. Per the plan, the W7.2 partial-fill simulator
    returns 100% fill (no partial) when ``l2_depth`` is
    ``None``, regardless of order size. This is the v1
    contract for the BTC paper trader; an L2 depth source is
    a future story.
    """
    cfg: PartialFillConfig = PartialFillConfig(seed=42)
    sim: PartialFillSimulator = PartialFillSimulator(cfg)

    # Small order: 100% fill.
    r_small: FillResult = sim.simulate(qty=0.01, l2_depth=None)
    assert isinstance(r_small, FillResult)
    assert r_small.fill_fraction == pytest.approx(1.0)
    assert r_small.filled_qty == pytest.approx(0.01)

    # Large order: also 100% fill (the no-L2 fallback does not
    # consult the threshold or the logistic model).
    r_large: FillResult = sim.simulate(qty=1_000_000.0, l2_depth=None)
    assert r_large.fill_fraction == pytest.approx(1.0)
    assert r_large.filled_qty == pytest.approx(1_000_000.0)

    # Negative validation: qty <= 0 raises.
    with pytest.raises(ValueError, match="qty"):
        sim.simulate(qty=0.0, l2_depth=None)
    with pytest.raises(ValueError, match="qty"):
        sim.simulate(qty=-1.0, l2_depth=None)

    # l2_depth=0 is also invalid (when explicitly provided).
    with pytest.raises(ValueError, match="l2_depth"):
        sim.simulate(qty=0.1, l2_depth=0.0)


# ---------------------------------------------------------------------------
# W7.2 acceptance criterion #2: logistic fill probability is in [0, 1]
# ---------------------------------------------------------------------------
def test_logistic_fill_probability_bounded() -> None:
    """The logistic fill probability is bounded in [0, 1] for any input.

    The simulator's ``logistic_fill_probability`` (and the
    internal ``_logistic_p_fill`` it consults) is the standard
    logistic sigmoid ``1 / (1 + exp(-z))`` where
    ``z = intercept + slope * log(qty / l2_depth)``. The
    sigmoid is bounded in ``(0, 1)`` for any real ``z``; the
    test sweeps a wide range of ``qty / l2_depth`` ratios
    (from 1e-6 to 1e6) to pin the bound under extreme inputs.
    """
    cfg: PartialFillConfig = PartialFillConfig(seed=20260608)
    sim: PartialFillSimulator = PartialFillSimulator(cfg)

    # Sanity: the default parameters are the v1 contract.
    assert cfg.threshold_fraction == DEFAULT_THRESHOLD_FRACTION
    assert cfg.intercept == DEFAULT_INTERCEPT
    assert cfg.slope == DEFAULT_SLOPE

    # Sweep a wide range of qty / l2_depth ratios. The
    # logistic's log-space parameterisation means the bound
    # is symmetric in log-space, so we sweep both directions.
    ratios: list[float] = [1e-6, 1e-3, 0.5, 1.0, 2.0, 100.0, 1e3, 1e6]
    for ratio in ratios:
        # Construct a pair (qty, l2_depth) with the given
        # ratio. The absolute values do not matter — only the
        # ratio enters the logistic.
        qty: float = ratio
        l2_depth: float = 1.0
        p: float = sim.logistic_fill_probability(qty=qty, l2_depth=l2_depth)
        assert 0.0 <= p <= 1.0, (
            f"logistic fill probability must be in [0, 1] for "
            f"ratio={ratio}; got p={p}"
        )
        assert np.isfinite(p), (
            f"logistic fill probability must be finite for "
            f"ratio={ratio}; got p={p}"
        )

    # At ratio=1.0 (qty == l2_depth), the default logistic
    # parameters (intercept=0.0, slope=-1.0) give
    # ``p = 1 / (1 + exp(0)) = 0.5``.
    p_at_one: float = sim.logistic_fill_probability(qty=1.0, l2_depth=1.0)
    assert p_at_one == pytest.approx(0.5, abs=1e-9), (
        f"logistic fill probability at ratio=1.0 must be 0.5 with "
        f"default parameters; got {p_at_one}"
    )

    # At very large ratio (qty >> l2_depth), the logistic
    # collapses to ~0 (the slope is negative). The clip
    # ensures the result is bounded.
    p_huge: float = sim.logistic_fill_probability(qty=1e9, l2_depth=1.0)
    assert 0.0 <= p_huge <= 1e-6

    # At very small ratio (qty << l2_depth), the logistic
    # collapses to ~1. The clip ensures the result is bounded.
    p_tiny: float = sim.logistic_fill_probability(qty=1e-9, l2_depth=1.0)
    assert 1.0 - 1e-6 <= p_tiny <= 1.0

    # The simulate() method's stochastic Bernoulli path also
    # honours the [0, 1] bound (the filled_fraction is the
    # logistic output, which is already in [0, 1]).
    r: FillResult = sim.simulate(qty=10.0, l2_depth=1.0)
    assert 0.0 <= r.fill_fraction <= 1.0
    assert 0.0 <= r.filled_qty <= 10.0
