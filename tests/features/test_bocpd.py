"""Tests for the BOCPD regime detector (W9.2).

The two PRD-required tests are:

- :func:`test_catches_synthetic_injection` — 90% of injected regime
  shifts are detected within +/- 5 bars, false-alarm rate < 5%.
- :func:`test_bocpd_no_l2_required` — the detector operates on
  bar-level data only; no L2 source is needed.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from kairon.features.regime import (
    BOCPDConfig,
    BOCPDRegimeDetector,
    Regime,
)


# ---------------------------------------------------------------------------
# PRD W9.2: test_catches_synthetic_injection
# ---------------------------------------------------------------------------
def test_catches_synthetic_injection() -> None:
    """Inject 10 regime shifts into a synthetic vol/spread series; assert
    >= 90% are detected within +/- 5 bars and the false-alarm rate
    is < 5%.

    The injection pattern is: a 200-bar warmup at a stable regime
    (vol=0.005), then 10 distinct regime segments of 200 bars each
    (2000 bars). The "true shifts" are the 10 boundaries between
    the 10 segments (bars 200, 400, 600, ..., 2000). The 200-bar
    warmup gives the detector time to learn the initial vol mean
    so the first shift is detectable.
    """
    rng = np.random.default_rng(20260608)
    n_per_segment = 200
    n_warmup = 200
    n_segments = 10
    n_bars = n_warmup + n_per_segment * n_segments
    # 10 distinct (vol, spread) means, alternating low/high so each
    # shift is well-defined (low -> high or high -> low) and the
    # detector cannot cheat by monotone trends.
    vol_means = [0.005, 0.020, 0.005, 0.030, 0.005, 0.040, 0.005, 0.030, 0.005, 0.050]
    spread_means = [5.0, 15.0, 5.0, 20.0, 5.0, 12.0, 5.0, 25.0, 5.0, 18.0]
    vol = np.zeros(n_bars, dtype=np.float64)
    sp = np.zeros(n_bars, dtype=np.float64)
    # Warmup segment: vol=0.005, spread=5.0 (matches segment 0).
    vol[:n_warmup] = 0.005 + rng.normal(0.0, 0.0005, size=n_warmup)
    sp[:n_warmup] = 5.0 + rng.normal(0.0, 0.5, size=n_warmup)
    for i in range(n_segments):
        lo = n_warmup + i * n_per_segment
        hi = n_warmup + (i + 1) * n_per_segment
        vol[lo:hi] = vol_means[i] + rng.normal(0.0, vol_means[i] * 0.1, size=n_per_segment)
        sp[lo:hi] = spread_means[i] + rng.normal(0.0, 1.0, size=n_per_segment)
    # True changepoints: the 10 segment boundaries.
    true_shifts: list[int] = [
        n_warmup + k * n_per_segment for k in range(1, n_segments)
    ] + [n_warmup]  # the first shift (warmup -> segment 0) is also detectable
    true_shifts = sorted(set(true_shifts))
    # Build a detector with a moderate S_MAX so changepoints are
    # detected with a small lag.
    cfg = BOCPDConfig(s_max=80, hazard_rate=0.02)
    detector = BOCPDRegimeDetector(cfg)
    detected = detector.changepoints(vol, sp)
    # Count true positives: a true shift is "detected" if any
    # detector-reported changepoint is within +/- 5 bars.
    tolerance = 5
    matched: set[int] = set()
    for ts in true_shifts:
        for d in detected:
            if abs(d - ts) <= tolerance and ts not in matched:
                matched.add(ts)
                break
    recall = len(matched) / len(true_shifts)
    # False alarms: detector-reported changepoints that do NOT
    # match any true shift within the tolerance.
    false_alarms = [
        d
        for d in detected
        if not any(abs(d - ts) <= tolerance for ts in true_shifts)
    ]
    n_non_shift_bars = n_bars - len(true_shifts)
    false_alarm_rate = len(false_alarms) / n_non_shift_bars
    assert recall >= 0.90, (
        f"recall {recall:.3f} < 0.90 (detected {len(matched)} of "
        f"{len(true_shifts)} true shifts within +/- {tolerance} bars; "
        f"true_shifts={true_shifts}, detected={detected})"
    )
    assert false_alarm_rate < 0.05, (
        f"false-alarm rate {false_alarm_rate:.4f} >= 0.05 "
        f"({len(false_alarms)} false positives in {n_non_shift_bars} "
        f"non-shift bars)"
    )


# ---------------------------------------------------------------------------
# PRD W9.2: test_bocpd_no_l2_required
# ---------------------------------------------------------------------------
def test_bocpd_no_l2_required() -> None:
    """The detector operates on bar-level data only; no L2 source is needed.

    We construct a bar-level OHLCV table (the v1 schema), compute
    realized vol + spread from it, and feed the result to the
    detector. The test asserts:
      1. The detector accepts the input without an L2 depth column.
      2. The detector returns a non-empty list of BOCPDState per bar.
      3. The detector's last_state is set (i.e. update() ran).
      4. The detector's regime labels are one of the 4 Regime values.
    """
    from datetime import UTC, datetime

    from kairon.data.io import OHLCV_SCHEMA

    n = 250
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.01, n))
    table = pa.table(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC) for _ in range(n)],
            "open": (close - 0.5).tolist(),
            "high": (close + 1.0).tolist(),
            "low": (close - 1.0).tolist(),
            "close": close.tolist(),
            "volume": [10.0] * n,
        },
        schema=OHLCV_SCHEMA,
    )
    # Build bar-level realized vol and spread WITHOUT any L2 source.
    # realized vol: rolling 20-bar std of log returns.
    log_close = np.log(np.array(table.column("close").to_pylist(), dtype=np.float64))
    log_returns = np.diff(log_close, prepend=log_close[0])
    realized_vol = np.array(
        [
            float(log_returns[max(0, i - 19) : i + 1].std(ddof=0))
            for i in range(n)
        ],
        dtype=np.float64,
    )
    # spread_bps: half the (high - low) / close * 1e4 (a bar-level
    # proxy for the bid-ask spread when no L2 data is available).
    high = np.array(table.column("high").to_pylist(), dtype=np.float64)
    low = np.array(table.column("low").to_pylist(), dtype=np.float64)
    spread_bps = (high - low) / close * 1e4
    detector = BOCPDRegimeDetector(BOCPDConfig(s_max=50))
    states = detector.detect(realized_vol, spread_bps)
    assert len(states) == n
    assert detector.last_state is not None
    # All regime labels must be valid Regime values.
    valid = {r.value for r in Regime}
    for s in states:
        assert s.regime.value in valid
    # The detector is a no-L2 contract: the inputs are 1-D arrays
    # of (vol, spread). Verify we can call update with two floats
    # and still get a valid BOCPDState.
    one = detector.update(0.01, 5.0)
    assert one.regime.value in valid


# ---------------------------------------------------------------------------
# Supplementary tests
# ---------------------------------------------------------------------------
def test_bocpd_default_config_validates() -> None:
    """The default BOCPDConfig is constructable and the detector runs."""
    detector = BOCPDRegimeDetector()
    s = detector.update(0.01, 5.0)
    assert s.run_length_posterior.shape == (detector.config.s_max,)
    assert abs(s.run_length_posterior.sum() - 1.0) < 1e-6


def test_bocpd_rejects_invalid_config() -> None:
    """The detector raises on out-of-range config values."""
    with pytest.raises(ValueError, match=r"s_max"):
        BOCPDRegimeDetector(BOCPDConfig(s_max=2))
    with pytest.raises(ValueError, match=r"hazard_rate"):
        BOCPDRegimeDetector(BOCPDConfig(hazard_rate=1.5))
    with pytest.raises(ValueError, match=r"hazard_rate"):
        BOCPDRegimeDetector(BOCPDConfig(hazard_rate=0.0))
    with pytest.raises(ValueError, match=r"kappa_0"):
        BOCPDRegimeDetector(BOCPDConfig(kappa_0=0.0))
    with pytest.raises(ValueError, match=r"vol_scale"):
        BOCPDRegimeDetector(BOCPDConfig(vol_scale=0.0))


def test_bocpd_reset_clears_state() -> None:
    """After reset(), the detector returns to the initial state.

    The first update after reset has a run-length posterior that
    is concentrated near 0 (the initial state); the posterior mean
    is < 1.0 because the changepoint probability is non-zero on
    the first bar.
    """
    detector = BOCPDRegimeDetector()
    for _ in range(10):
        detector.update(0.01, 5.0)
    detector.reset()
    assert detector.last_state is None
    s = detector.update(0.01, 5.0)
    # After reset, the run-length posterior mass is near 0 (the
    # changepoint probability dominates the first bar's
    # posterior). The MAP may be 0 or 1 depending on the
    # hazard; the test asserts the mean is small (< 1.5) which
    # is the robust check.
    assert s.run_length_mean < 1.5
    # The run-length posterior sums to 1 (a valid probability
    # distribution).
    assert abs(s.run_length_posterior.sum() - 1.0) < 1e-6


def test_bocpd_stress_override_fires_on_high_vol() -> None:
    """A bar with vol_z > stress_z is labelled STRESSED (rule override)."""
    # stress_z default = 3.0; vol_scale default = 0.01. So a vol of
    # 0.05 (= 5x the scale) should trigger STRESSED.
    detector = BOCPDRegimeDetector()
    s = detector.update(0.05, 5.0)
    assert s.regime == Regime.STRESSED


def test_bocpd_volatile_override_fires_on_moderate_vol() -> None:
    """A bar with volatile_z < vol_z <= stress_z is labelled VOLATILE."""
    # volatile_z default = 1.5; vol_scale default = 0.01. So a vol
    # of 0.02 (= 2x the scale, between 1.5 and 3.0) should trigger
    # VOLATILE.
    detector = BOCPDRegimeDetector()
    s = detector.update(0.02, 5.0)
    assert s.regime == Regime.VOLATILE


def test_bocpd_label_table_returns_pa_array() -> None:
    """label_table returns a pyarrow string array of per-bar labels."""
    rng = np.random.default_rng(0)
    vol = rng.normal(0.01, 0.001, 100)
    sp = rng.normal(5.0, 0.5, 100)
    detector = BOCPDRegimeDetector()
    arr = detector.label_table(vol, sp)
    assert isinstance(arr, pa.Array)
    assert arr.type == pa.string()
    assert len(arr) == 100
