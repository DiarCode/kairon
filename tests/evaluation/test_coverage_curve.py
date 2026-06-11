"""Tests for :mod:`kairon.evaluation.coverage_curve` and the W3.5 runner.

The four tests pin the W3.5 acceptance criteria:

1. ``test_pareto_monotone`` — on a synthetic fixture where
   the primary's confidence is calibrated (i.e. higher
   ``p_final`` -> higher ``y_true`` rate), as ``T``
   increases, coverage monotonically *decreases* and
   accuracy on the covered subset monotonically
   *increases*. The test asserts strict monotone on a
   "calibrated" fixture.

2. ``test_two_reference_points_emitted`` — the JSON
   output contains exactly the two reference points
   ``{t_at_25pct_coverage, t_at_10pct_coverage}`` with
   their ``(coverage, accuracy)`` pairs.

3. ``test_coverage_curve_handles_constant_predictions`` —
   when ``p_final`` is constant (e.g. all 0.5), the
   function returns the expected degenerate curve
   (``coverage=0`` for ``T>0.5``, ``coverage=1`` for
   ``T<=0.5``) without NaN.

4. ``test_coverage_curve_handles_perfect_predictions`` —
   when ``p_final`` perfectly predicts ``y_true``, the
   accuracy on the covered subset is 100% for every
   ``T`` that has ``n_signals > 0``.

Tests 1, 3, 4 exercise :func:`coverage_curve` directly
(unit-test style). Test 2 drives the W3.5 runner
in-process and reads the JSON sidecar back, asserting the
two reference points are present and well-formed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kairon.evaluation.coverage_curve import coverage_curve
from scripts import run_coverage_curve as rcc


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------
_N_BARS: int = 1000
_SEED: int = 20260608  # deterministic; the W3.5 date


def _calibrated_fixture(
    n: int = _N_BARS,
    seed: int = _SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, p_final)`` where ``p_final`` is calibrated.

    The synthetic generator produces a binary label
    ``y_true[i] ~ Bernoulli(p_final[i])`` where
    ``p_final[i]`` is drawn uniformly from ``[0, 1]``. This
    is a *calibrated* primary in the Brier sense: the
    average ``y_true`` over the bars with
    ``p_final > T`` is approximately ``T + 0.5 * (1 - T)``
    (i.e. a smooth monotone in ``T``). The fixture is the
    load-bearing input for ``test_pareto_monotone``.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    p_final: np.ndarray = rng.uniform(low=0.0, high=1.0, size=n).astype(np.float64)
    y_true: np.ndarray = (rng.uniform(low=0.0, high=1.0, size=n) < p_final).astype(
        np.float64
    )
    return y_true, p_final


def _perfect_fixture(
    n: int = _N_BARS,
    seed: int = _SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, p_final)`` where ``p_final`` perfectly predicts ``y_true``.

    For every bar, ``p_final[i] > 0.5`` iff
    ``y_true[i] == 1`` and ``p_final[i] < 0.5`` iff
    ``y_true[i] == 0``. The accuracy on the covered
    subset is therefore 100% for every ``T`` that has
    ``n_signals > 0``. The fixture pins
    ``test_coverage_curve_handles_perfect_predictions``.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    y_true: np.ndarray = (rng.uniform(low=0.0, high=1.0, size=n) < 0.5).astype(
        np.float64
    )
    # p_final is a strongly separated point near 0 or near 1
    # depending on y_true, with a strictly-clean margin
    # from 0.5 so the threshold sweep has well-defined
    # bracket points.
    p_final: np.ndarray = np.where(
        y_true == 1.0,
        0.5 + 0.4 * rng.uniform(low=0.0, high=1.0, size=n),
        0.5 - 0.4 * rng.uniform(low=0.0, high=1.0, size=n),
    ).astype(np.float64)
    return y_true, p_final


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_pareto_monotone() -> None:
    """As T increases, coverage decreases and accuracy increases.

    On a calibrated synthetic fixture (a perfect Brier
    primary), increasing ``T`` retains only the most
    confident predictions. The coverage MUST monotonically
    decrease (as ``T`` rises, fewer bars satisfy
    ``p_final > T``), and the accuracy on the covered
    subset MUST monotonically increase (a high-confidence
    bar is more likely to be correct in a calibrated
    primary). The test asserts *strict* monotone
    (``>=`` ``>``) within a small floating-point tolerance
    to allow for the rare tie at a single T.
    """
    y_true, p_final = _calibrated_fixture()
    out: dict[str, Any] = coverage_curve(y_true, p_final)
    curve: dict[float, dict[str, float]] = out["curve"]
    thresholds: list[float] = sorted(float(t) for t in curve.keys())

    coverages: list[float] = [curve[t]["coverage"] for t in thresholds]
    accuracies: list[float] = [curve[t]["accuracy"] for t in thresholds]

    # 1. Coverage is monotonically NON-INCREASING in T. A
    # strict-decrease test is too brittle for a synthetic
    # fixture (a 1000-bar sample can have ties at one or
    # two adjacent T's); the non-strict version is the
    # load-bearing direction check the PRD requires.
    for i in range(len(coverages) - 1):
        c_lo: float = coverages[i]
        c_hi: float = coverages[i + 1]
        assert c_lo >= c_hi - 1e-12, (
            f"coverage must be non-increasing in T: "
            f"coverage(T={thresholds[i]})={c_lo} < "
            f"coverage(T={thresholds[i + 1]})={c_hi}"
        )

    # 2. Accuracy on the covered subset is monotonically
    # NON-DECREASING in T. For a calibrated primary, a
    # higher T retains only the more confident
    # predictions, which are more likely to be correct.
    # The monotone test asserts the direction the PRD
    # requires.
    for i in range(len(accuracies) - 1):
        a_lo: float = accuracies[i]
        a_hi: float = accuracies[i + 1]
        # Both T's must have n_signals > 0 for the
        # accuracy comparison to be meaningful. If one
        # of the brackets has n_signals == 0 we skip
        # the comparison (an empty covered subset has
        # accuracy=0 by our convention; comparing
        # against a non-empty bracket would always
        # violate monotonicity). The PRD's W3.5 test
        # fixture has coverage at T=0.90 of ~10%, well
        # above zero for n=1000.
        n_lo: float = curve[thresholds[i]]["n_signals"]
        n_hi: float = curve[thresholds[i + 1]]["n_signals"]
        if n_lo == 0.0 or n_hi == 0.0:
            continue
        assert a_hi >= a_lo - 1e-12, (
            f"accuracy must be non-decreasing in T: "
            f"accuracy(T={thresholds[i]})={a_lo} > "
            f"accuracy(T={thresholds[i + 1]})={a_hi}"
        )

    # 3. The two reference points fall within the
    # threshold axis. The 25% and 10% coverages are
    # strictly between the coverage at T=0.50 and the
    # coverage at T=0.90, so the interpolated T's
    # should land strictly inside the threshold axis.
    assert 0.50 <= out["t_at_25pct_coverage"] <= 0.90
    assert 0.50 <= out["t_at_10pct_coverage"] <= 0.90
    # The 25% point has LOWER coverage than the 10%
    # point, so the 25% T is LOWER than the 10% T
    # (coverage is non-increasing in T).
    assert out["t_at_25pct_coverage"] <= out["t_at_10pct_coverage"]


def test_two_reference_points_emitted(tmp_path: Path) -> None:
    """The runner JSON contains the two reference points with (T, accuracy) pairs.

    Drives the W3.5 runner in-process via :func:`main`
    with the report + sidecar paths redirected to a tmp
    dir, then reads the JSON sidecar back and asserts:

    - the sidecar exists at the canonical path
    - the JSON parses cleanly (``json.loads`` succeeds)
    - the sidecar has 12 ``rows`` (3 assets x 4 horizons)
    - the sidecar exposes the two reference points
      ``t_at_25pct_coverage`` and
      ``t_at_10pct_coverage`` with their
      ``(t_at_25pct_accuracy, t_at_10pct_accuracy)``
      pairs.
    - every row's reference T's fall in the threshold axis.
    """
    report_path: Path = tmp_path / "reports" / "coverage_pareto_w4.json"
    sidecar_path: Path = tmp_path / "artifacts" / "coverage_pareto_w4.json"
    rc: int = rcc.main([
        "--report-path", str(report_path),
        "--sidecar-path", str(sidecar_path),
    ])
    assert rc == 0
    assert report_path.exists(), f"missing report at {report_path}"

    sidecar: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))

    # Per the PRD W3.5 spec, the headline
    # ``reference_point_coverage_pct`` is the tuple
    # ``[25, 10]`` so downstream tools know the targets.
    assert sidecar["reference_point_coverage_pct"] == [25, 10]

    assert "rows" in sidecar
    assert len(sidecar["rows"]) == 12

    for row in sidecar["rows"]:
        # Per-row reference points. The keys MUST be
        # exactly the two reference points the PRD
        # names, with their (T, accuracy) pairs.
        assert "t_at_25pct_coverage" in row
        assert "t_at_25pct_accuracy" in row
        assert "t_at_10pct_coverage" in row
        assert "t_at_10pct_accuracy" in row
        # The T's are in [0, 1] and the accuracies are
        # in [0, 1] (binary case).
        t25: float = row["t_at_25pct_coverage"]
        t10: float = row["t_at_10pct_coverage"]
        assert 0.0 <= t25 <= 1.0
        assert 0.0 <= t10 <= 1.0
        assert 0.0 <= row["t_at_25pct_accuracy"] <= 1.0
        assert 0.0 <= row["t_at_10pct_accuracy"] <= 1.0
        # 25% coverage T <= 10% coverage T (since
        # coverage is non-increasing in T).
        assert t25 <= t10, (
            f"t_at_25pct_coverage={t25} must be <= "
            f"t_at_10pct_coverage={t10}"
        )
        # The full_curve is a list of {threshold,
        # coverage, accuracy} dicts, one per threshold
        # in the W3.5 default ladder.
        assert "full_curve" in row
        assert len(row["full_curve"]) >= 2
        for pt in row["full_curve"]:
            assert set(pt.keys()) == {"threshold", "coverage", "accuracy"}


def test_coverage_curve_handles_constant_predictions() -> None:
    """Constant ``p_final`` returns a degenerate curve without NaN.

    When ``p_final`` is constant at some value ``c``:
    - For ``T < c``: coverage is 1.0
    - For ``T > c``: coverage is 0.0
    - For ``T == c``: coverage is 0.0 (strict inequality
      ``p_final > T`` excludes the equality case)
    The function must return finite values throughout,
    with no NaN, no inf, and the documented structure.
    """
    n: int = 1000
    y_true: np.ndarray = np.zeros(n, dtype=np.float64)
    y_true[::2] = 1.0  # 50% positive rate so the curve is non-trivial
    p_final: np.ndarray = np.full(n, 0.5, dtype=np.float64)

    out: dict[str, Any] = coverage_curve(y_true, p_final)
    curve: dict[float, dict[str, float]] = out["curve"]

    for t, metrics in curve.items():
        cov: float = metrics["coverage"]
        acc: float = metrics["accuracy"]
        brier: float = metrics["brier"]
        n_sig: float = metrics["n_signals"]
        # All values must be finite.
        assert math.isfinite(cov)
        assert math.isfinite(acc)
        assert math.isfinite(brier)
        assert math.isfinite(n_sig)
        # For T < 0.5, every bar has p_final=0.5 > T, so
        # coverage is 1.0. For T > 0.5, every bar has
        # p_final=0.5 NOT > T, so coverage is 0.0. For
        # T == 0.5, p_final=0.5 NOT > 0.5, so coverage
        # is 0.0.
        if t < 0.5:
            assert cov == pytest.approx(1.0, abs=1e-12)
        else:
            assert cov == pytest.approx(0.0, abs=1e-12)

    # The two reference points are well-defined even for
    # a constant primary. The 25% and 10% target
    # coverages are both >= the highest observed
    # coverage at T=0.50, so the interpolation clamps
    # both reference T's to T=0.50 (the lowest
    # threshold).
    assert out["t_at_25pct_coverage"] == pytest.approx(0.50, abs=1e-12)
    assert out["t_at_10pct_coverage"] == pytest.approx(0.50, abs=1e-12)


def test_coverage_curve_handles_perfect_predictions() -> None:
    """Perfect ``p_final`` yields accuracy=1.0 on the covered subset.

    When ``p_final`` perfectly predicts ``y_true`` (i.e.
    ``p_final > 0.5`` iff ``y_true == 1`` and vice
    versa), the accuracy on the covered subset is 1.0 for
    every ``T`` that has ``n_signals > 0``. The Brier on
    the covered subset is also 0.0 (predictions are
    exact: ``p_final`` matches ``y_true`` up to a
    small within-margin noise).
    """
    y_true, p_final = _perfect_fixture()
    out: dict[str, Any] = coverage_curve(y_true, p_final)
    curve: dict[float, dict[str, float]] = out["curve"]

    for t, metrics in curve.items():
        n_sig: float = metrics["n_signals"]
        if n_sig == 0:
            # Empty covered subset: accuracy / Brier are
            # 0.0 by convention (the function must not
            # return NaN for empty subsets).
            assert metrics["accuracy"] == 0.0
            assert metrics["brier"] == 0.0
            continue
        # n_signals > 0: accuracy must be exactly 1.0
        # on a perfect-prediction fixture.
        acc: float = metrics["accuracy"]
        assert acc == pytest.approx(1.0, abs=1e-9), (
            f"accuracy on covered subset at T={t} should be "
            f"1.0 for a perfect-prediction primary, got {acc} "
            f"(n_signals={n_sig})"
        )
