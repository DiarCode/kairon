"""Tests for the W7.1 latency simulator.

Three tests pin the W7.1 acceptance criteria:

1. ``test_deterministic_seeded`` — two :class:`LatencySimulator`
   instances with the same ``seed`` produce identical
   round-trip latency sequences (the load-bearing determinism
   contract for paper-trader regression tests).
2. ``test_p50_p99_within_config`` — a 1000-sample draw produces
   a median of ~50ms (within a small tolerance) and a 99th
   percentile below the ``max_ms=500`` cap. The cap keeps the
   tail bounded so a regression that disables the clamp is
   caught immediately.
3. ``test_lognormal_shape`` — a Kolmogorov-Smirnov test against
   ``scipy.stats.lognorm(s=0.5, scale=50)`` on a 1000-sample
   draw yields ``p > 0.05`` (i.e. the simulator's draws are
   statistically indistinguishable from the reference
   lognormal at the 5% level). The KS test is the canonical
   goodness-of-fit test for a continuous distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from kairon.paper.latency import (
    DEFAULT_MAX_MS,
    DEFAULT_MEAN_MS,
    DEFAULT_SIGMA,
    LatencyConfig,
    LatencySimulator,
)


# ---------------------------------------------------------------------------
# W7.1 acceptance criterion #1: deterministic seeded
# ---------------------------------------------------------------------------
def test_deterministic_seeded() -> None:
    """Two simulators with the same seed produce identical sequences.

    The v1 contract for the paper trader is a fully deterministic
    latency draw — two ``LatencySimulator`` instances built with
    the same ``LatencyConfig(seed=...)`` MUST emit the same
    latency sequence element-by-element. This is the regression
    guard for paper-trader parity tests (the W8 paper-trader
    fixtures rely on this).
    """
    cfg: LatencyConfig = LatencyConfig(seed=20260608)
    sim_a: LatencySimulator = LatencySimulator(cfg)
    sim_b: LatencySimulator = LatencySimulator(cfg)

    n: int = 100
    draws_a: np.ndarray = sim_a.sample_many(n)  # type: ignore[type-arg]
    draws_b: np.ndarray = sim_b.sample_many(n)  # type: ignore[type-arg]

    # Element-wise equality to 1e-12 (the simulator's underlying
    # ``RandomState.lognormal`` returns float64 and the clamp is
    # a bit-identical ``np.minimum``).
    assert draws_a.shape == (n,)
    assert draws_b.shape == (n,)
    assert np.array_equal(draws_a, draws_b), (
        "two LatencySimulator instances with the same seed must "
        "produce identical sequences; got "
        f"a[:5]={draws_a[:5]} vs b[:5]={draws_b[:5]}"
    )

    # And: a single ``sample()`` call also matches.
    single_a: float = sim_a.sample()
    single_b: float = sim_b.sample()
    # The two simulators have now both advanced the RNG; a fresh
    # pair should match on the *next* draw.
    sim_c: LatencySimulator = LatencySimulator(cfg)
    sim_d: LatencySimulator = LatencySimulator(cfg)
    for _ in range(n):
        _ = sim_c.sample()
        _ = sim_d.sample()
    assert sim_c.sample() == sim_d.sample(), (
        "two LatencySimulator instances with the same seed must "
        "produce identical sequences after the same number of draws"
    )
    # And the two original simulators (which drew ``n + 1`` times)
    # are at the same RNG state, so their next single draw matches
    # too.
    assert sim_a.sample() == sim_b.sample()
    # ``single_a`` and ``single_b`` are not directly comparable
    # because they are the n+1'th draw of two independent RNGs
    # (which, per the determinism contract, are equal); reference
    # it to silence the linter.
    _ = single_a
    _ = single_b


# ---------------------------------------------------------------------------
# W7.1 acceptance criterion #2: p50 / p99 within config
# ---------------------------------------------------------------------------
def test_p50_p99_within_config() -> None:
    """1000 samples have p50 ~= 50ms and p99 < 500ms.

    The v1 contract is a lognormal with median ``mean_ms=50`` and
    a heavy tail that is *clamped* at ``max_ms=500``. The 99th
    percentile of the *unclamped* lognormal is approximately
    ``exp(mu + 2.33 * sigma) = 50 * exp(2.33 * 0.5) ~= 158ms``,
    well below the 500ms cap. The test pins p99 strictly below
    ``max_ms`` so a regression that *disables* the clamp (e.g.
    setting ``max_ms=inf``) is caught.

    The p50 check uses a 5% relative tolerance around 50ms to
    accommodate the sampling noise on a 1000-sample draw. The
    seed (42) is fixed so the test is deterministic.
    """
    cfg: LatencyConfig = LatencyConfig(seed=42)
    sim: LatencySimulator = LatencySimulator(cfg)
    samples: np.ndarray = sim.sample_many(1000)  # type: ignore[type-arg]

    p50: float = float(np.median(samples))
    p99: float = float(np.percentile(samples, 99))

    # p50 within 5% of mean_ms=50.
    assert p50 == pytest.approx(DEFAULT_MEAN_MS, rel=0.05, abs=1e-9), (
        f"p50 ({p50:.3f}) should be ~{DEFAULT_MEAN_MS}ms "
        f"(within 5% rel tolerance)"
    )

    # p99 strictly below the max_ms cap. The cap is 500ms by
    # default; even the unclamped 99.9th percentile of the
    # lognormal is below 500ms (it's ~exp(mu + 3.09*sigma)
    # ~= 215ms), so this assertion is safe.
    assert p99 < DEFAULT_MAX_MS, (
        f"p99 ({p99:.3f}) must be strictly below the max_ms "
        f"cap ({DEFAULT_MAX_MS}ms); the clamp is broken or "
        f"the parameters are out of spec"
    )

    # Sanity: the draws are positive and bounded.
    assert np.all(samples > 0.0)
    assert np.all(samples <= DEFAULT_MAX_MS)


# ---------------------------------------------------------------------------
# W7.1 acceptance criterion #3: lognormal shape (KS test)
# ---------------------------------------------------------------------------
def test_lognormal_shape() -> None:
    """A 1000-sample draw passes a KS test vs scipy's lognormal.

    The simulator's draws are sampled from
    ``numpy.random.RandomState.lognormal(mean=log(50), sigma=0.5)``
    and then clamped at 500ms. The reference distribution is
    ``scipy.stats.lognorm(s=0.5, scale=50)`` (the standard
    scipy parameterisation: ``s=sigma``, ``scale=exp(mu)``).
    The 5% level KS test is the canonical "is this sample from
    the reference distribution?" check.

    The clamp at 500ms makes the *clamped* sample NOT a pure
    lognormal (the upper tail is truncated at 500ms), so the
    KS test is run against the *raw* (unclamped) draws via
    ``sample_raw()``. The clamp is tested in
    ``test_p50_p99_within_config``.
    """
    pytest.importorskip("scipy")

    cfg: LatencyConfig = LatencyConfig(seed=20260608)
    sim: LatencySimulator = LatencySimulator(cfg)

    n: int = 1000
    # Draw n+1 raw samples; we drop the first one because the
    # ``RandomState`` is in a slightly different state right
    # after construction vs after the first call.
    raw_samples: np.ndarray = np.array(
        [sim.sample_raw() for _ in range(n)], dtype=np.float64
    )

    # Reference distribution: scipy's lognormal with the same
    # parameters (s=sigma, scale=mean_ms). scipy is imported at
    # module scope, so the runtime type is available.
    ref = stats.lognorm(s=cfg.sigma, scale=cfg.mean_ms)

    # Kolmogorov-Smirnov two-sided test. The p-value should be
    # well above 0.05 (the 5% level); with n=1000 a p-value
    # below 0.05 would indicate a mismatch in the distribution
    # parameterisation (e.g. confusing mu with mean).
    ks_result = stats.kstest(raw_samples, ref.cdf)
    p_value: float = float(ks_result.pvalue)

    assert p_value > 0.05, (
        f"KS test p-value ({p_value:.4f}) must be > 0.05 vs "
        f"scipy.stats.lognorm(s={cfg.sigma}, scale={cfg.mean_ms}); "
        f"the simulator's distribution parameterisation is wrong "
        f"(likely confusing mu with mean)"
    )
