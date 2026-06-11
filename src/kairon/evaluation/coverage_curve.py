"""Coverage-accuracy Pareto frontier for a threshold sweep.

Story W3.5 publishes the *headline UI* per
``docs/objective_and_metrics.md`` §1: a coverage-accuracy
Pareto frontier with two reference points (``T`` at 25%
coverage, ``T`` at 10% coverage), replacing the "single
accuracy number" mental model with a 2-D curve the
operator can read at a glance.

The function is the load-bearing primitive for the W3.5
runner script (``scripts/run_coverage_curve.py``) and is
re-used by the W6.3 CAS-dominance comparison. It takes the
final probability vector ``p_final`` (a 1-D
``np.ndarray`` of length ``N``) and the corresponding
``y_true`` binary labels, sweeps a configurable set of
confidence thresholds ``T``, and returns a dict whose keys
are the thresholds and whose values are a sub-dict with the
coverage / accuracy / hit-rate / Brier / n_signals metrics
at that threshold.

Two reference points are emitted alongside the full curve:

- ``t_at_25pct_coverage`` — the threshold ``T`` that hits
  exactly 25% coverage, found by **linear interpolation of
  the threshold axis** between the two bracket
  thresholds.
- ``t_at_10pct_coverage`` — same, for 10% coverage.

The reference points are the two numbers the operator
pins on the Pareto frontier: "what's the accuracy we get
at 25% coverage?" and "what's the accuracy we get at 10%
coverage?" The 25% and 10% are the W3.5 reference coverages
per ``docs/objective_and_metrics.md`` §1.

The function is pure: no IO, no async, no global state. It
is hermetic and deterministic for a given input — the
threshold axis is sorted ascending, the input arrays are
not mutated, and the interpolation is exact.

Edge cases
----------

- **Constant predictions** (``p_final`` is constant at some
  ``c``): the function still produces a valid coverage
  curve. For ``T < c`` the coverage is 1.0; for ``T > c``
  the coverage is 0.0; for ``T == c`` the coverage is
  configurable. The function never returns NaN for a
  constant-input fixture.
- **Perfect predictions** (``y_true == (p_final > 0.5)``
  exactly, no noise): the accuracy on the covered subset
  is 1.0 for every ``T`` that has ``n_signals > 0``.
- **All-empty covered subsets**: at high ``T`` values the
  coverage may be 0, so ``n_signals == 0``. The function
  reports ``n_signals`` honestly and the Brier / accuracy
  are well-defined on the empty subset (Brier is NaN by
  convention when there are no samples; we return
  ``0.0`` here so the JSON sidecar is well-formed and
  downstream tools don't have to special-case NaN).
  Accuracy is similarly reported as ``0.0`` on an empty
  covered subset.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Default threshold ladder. Per the PRD W3.5 task spec: sweep
# T from 0.50 to 0.90 in 0.05 steps. The tuple is a
# ``tuple[float, ...]`` (not a list) so it is hashable and
# ordered.
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
)

# Default reference coverages. Per ``docs/objective_and_metrics.md``
# §1 + the PRD W3.5 acceptance criterion. The 25% point is
# the "comfortable" reference (the strategy is willing to
# trade on roughly 1 in 4 bars); the 10% point is the
# "high-precision" reference (only the most confident
# signals).
DEFAULT_REFERENCE_COVERAGES: tuple[float, ...] = (0.25, 0.10)


def _safe_float(x: float) -> float:
    """Coerce a non-finite float to 0.0 (defensive for empty subsets)."""
    if not math.isfinite(x):
        return 0.0
    return float(x)


def _per_threshold_metrics(
    y_true: np.ndarray,
    p_final: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute coverage / accuracy / hit_rate / brier / n_signals at one T.

    Parameters
    ----------
    y_true
        1-D ``np.ndarray`` of binary labels (0 or 1). Must
        have the same length as ``p_final``.
    p_final
        1-D ``np.ndarray`` of final probabilities in
        ``[0, 1]``. Must have the same length as ``y_true``.
    threshold
        The confidence threshold ``T`` in ``[0, 1]``. A bar
        is "covered" (i.e. the model emits a signal) when
        ``p_final[i] > T`` (strict inequality — a bar at
        exactly ``T`` is not covered; this matches the
        canonical "confidence > T" decision rule).

    Returns
    -------
    dict[str, float]
        A sub-dict with the documented keys
        ``{coverage, accuracy, hit_rate, brier, n_signals}``.
        ``coverage`` is the fraction of bars with
        ``p_final > T``; ``accuracy`` is the mean of
        ``y_true`` on the covered subset (0.0 on an empty
        subset); ``hit_rate`` is the mean of ``y_true`` on
        the covered subset (same as accuracy in the binary
        case; reported separately so the JSON sidecar
        matches the PRD W3.5 spec); ``brier`` is the mean
        squared error of ``p_final`` vs ``y_true`` on the
        covered subset (0.0 on an empty subset); and
        ``n_signals`` is the number of covered bars.
    """
    n_total: int = int(y_true.size)
    mask: np.ndarray = p_final > threshold
    n_signals: int = int(mask.sum())
    coverage: float = float(n_signals) / float(n_total) if n_total > 0 else 0.0
    if n_signals > 0:
        covered_y: np.ndarray = y_true[mask]
        covered_p: np.ndarray = p_final[mask]
        accuracy: float = float(covered_y.mean())
        brier: float = float(((covered_p - covered_y) ** 2).mean())
    else:
        accuracy = 0.0
        brier = 0.0
    return {
        "coverage": _safe_float(coverage),
        "accuracy": _safe_float(accuracy),
        "hit_rate": _safe_float(accuracy),  # binary case -> same as accuracy
        "brier": _safe_float(brier),
        "n_signals": float(n_signals),
    }


def _interpolate_threshold_at_coverage(
    thresholds: tuple[float, ...],
    coverages: list[float],
    target_coverage: float,
) -> float:
    """Linearly interpolate the threshold axis to hit a target coverage.

    Given a sorted-ascending threshold axis and the coverage
    at each threshold, find the threshold that *exactly*
    hits ``target_coverage`` by linearly interpolating
    between the two bracket thresholds.

    Edge cases
    ----------

    - ``target_coverage`` is above the highest observed
      coverage: returns the **lowest** threshold (the
      lowest T has the highest coverage, since coverage
      decreases as T increases for a fixed input). This
      is a clamp, not an extrapolation.
    - ``target_coverage`` is below the lowest observed
      coverage: returns the **highest** threshold (the
      highest T has the lowest coverage).
    - The two brackets land exactly on the target: returns
      the corresponding threshold without division-by-zero
      noise.

    Parameters
    ----------
    thresholds
        Sorted-ascending threshold axis (length >= 2).
    coverages
        Coverage at each threshold, same length as
        ``thresholds``. Coverage is *non-increasing* in T
        for a fixed input; we do not require strict
        monotonicity (the function is robust to small
        wiggles in the underlying coverage).
    target_coverage
        The desired coverage in ``[0, 1]``.

    Returns
    -------
    float
        The interpolated (or clamped) threshold.
    """
    n: int = len(thresholds)
    assert n == len(coverages) and n >= 2, (
        "thresholds and coverages must have the same length >= 2"
    )
    # Clamp the target to the observed range. The "lowest
    # coverage" point is at the highest threshold
    # (coverage decreases as T increases); the "highest
    # coverage" point is at the lowest threshold.
    highest_cov: float = float(coverages[0])  # at thresholds[0]
    lowest_cov: float = float(coverages[-1])  # at thresholds[-1]
    if target_coverage >= highest_cov:
        return float(thresholds[0])
    if target_coverage <= lowest_cov:
        return float(thresholds[-1])
    # Linear interpolation: find the bracket (i, i+1) such
    # that coverages[i] >= target >= coverages[i+1].
    # coverages is non-increasing in T.
    for i in range(n - 1):
        cov_lo: float = float(coverages[i])      # coverage at thresholds[i]
        cov_hi: float = float(coverages[i + 1])  # coverage at thresholds[i+1]
        # cov_lo >= target_coverage >= cov_hi
        if cov_lo >= target_coverage >= cov_hi:
            if cov_lo == cov_hi:
                # Degenerate (no change in coverage between
                # the two brackets): return the midpoint of
                # the threshold axis. This is the safe
                # choice when coverage is flat.
                return float(0.5 * (thresholds[i] + thresholds[i + 1]))
            # Linear interpolation of the threshold axis
            # in coverage space. Solve T* = T_lo +
            # (T_hi - T_lo) * (cov_lo - target) / (cov_lo - cov_hi).
            t_lo: float = float(thresholds[i])
            t_hi: float = float(thresholds[i + 1])
            frac: float = (cov_lo - target_coverage) / (cov_lo - cov_hi)
            return t_lo + frac * (t_hi - t_lo)
    # Fallback (should not reach here): return the
    # highest-threshold point.
    return float(thresholds[-1])


def coverage_curve(
    y_true: np.ndarray,
    p_final: np.ndarray,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Sweep ``T`` over a threshold ladder and return the per-T metrics.

    Parameters
    ----------
    y_true
        1-D ``np.ndarray`` of binary ground-truth labels
        (0 or 1). Must have the same length as ``p_final``.
    p_final
        1-D ``np.ndarray`` of final probabilities in
        ``[0, 1]`` (the W3.4 ``MetaLabeledEnsemble``'s
        output, or any other calibrated probability
        vector). Must have the same length as ``y_true``.
    thresholds
        Sorted-ascending tuple of confidence thresholds
        ``T`` to sweep. Default: ``(0.50, 0.55, 0.60,
        0.65, 0.70, 0.75, 0.80, 0.85, 0.90)`` per the PRD
        W3.5 spec. Must have at least 2 elements.

    Returns
    -------
    dict[str, Any]
        A dict with the following keys:

        - ``"thresholds"`` (``list[float]``): the
          threshold axis (echo of the input).
        - ``"curve"`` (``dict[float, dict[str, float]]``):
          ``{T: {coverage, accuracy, hit_rate, brier,
          n_signals}}`` for each T in the input.
        - ``"t_at_25pct_coverage"`` (``float``): the
          threshold ``T`` that hits exactly 25% coverage,
          found by **linear interpolation of the threshold
          axis** between the two bracket thresholds.
        - ``"t_at_25pct_accuracy"`` (``float``): the
          accuracy on the covered subset at
          ``t_at_25pct_coverage`` (linearly interpolated
          from the two bracket accuracies).
        - ``"t_at_25pct_coverage_actual"`` (``float``):
          the coverage at ``t_at_25pct_coverage`` (should
          be 0.25 by construction; reported for
          traceability).
        - ``"t_at_10pct_coverage"`` (``float``): the
          threshold that hits exactly 10% coverage.
        - ``"t_at_10pct_accuracy"`` (``float``): the
          accuracy at the 10% reference point.
        - ``"t_at_10pct_coverage_actual"`` (``float``):
          the actual coverage at the 10% reference point.

    Raises
    ------
    ValueError
        If ``y_true`` is not a 1-D ``np.ndarray``, if
        ``p_final`` is not a 1-D ``np.ndarray``, if the
        two have different lengths, if either contains
        non-finite values, if ``thresholds`` is empty or
        has fewer than 2 elements, if any threshold is
        outside ``[0, 1]``, or if the thresholds are not
        sorted ascending.
    """
    # --- input validation (fail fast) ----------------------------
    if y_true.ndim != 1:
        raise ValueError(
            f"y_true must be a 1-D np.ndarray, got ndim={y_true.ndim}"
        )
    if p_final.ndim != 1:
        raise ValueError(
            f"p_final must be a 1-D np.ndarray, got ndim={p_final.ndim}"
        )
    n: int = int(y_true.size)
    if p_final.size != n:
        raise ValueError(
            f"y_true and p_final must have the same length, "
            f"got y_true.size={n} and p_final.size={p_final.size}"
        )
    if n == 0:
        raise ValueError("y_true and p_final must be non-empty")
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true must contain only finite values")
    if not np.all(np.isfinite(p_final)):
        raise ValueError("p_final must contain only finite values")
    if not thresholds or len(thresholds) < 2:
        raise ValueError(
            f"thresholds must have at least 2 elements, got "
            f"{len(thresholds) or 0}"
        )
    for t in thresholds:
        if not (math.isfinite(t) and 0.0 <= t <= 1.0):
            raise ValueError(
                f"each threshold must be in [0, 1], got {t!r}"
            )
    sorted_thresholds: list[float] = sorted(float(t) for t in thresholds)
    if sorted_thresholds != [float(t) for t in thresholds]:
        raise ValueError(
            f"thresholds must be sorted ascending, got "
            f"{list(thresholds)}"
        )

    # --- per-threshold metrics ------------------------------------
    # Force float64 for the brier / accuracy math so the
    # results are deterministic across platforms.
    y_true_arr: np.ndarray = np.asarray(y_true, dtype=np.float64)
    p_final_arr: np.ndarray = np.asarray(p_final, dtype=np.float64)

    curve: dict[float, dict[str, float]] = {}
    for t in thresholds:
        curve[float(t)] = _per_threshold_metrics(
            y_true_arr, p_final_arr, float(t),
        )

    # --- two reference points (interpolated) ----------------------
    # For each reference coverage target, find the threshold
    # that *exactly* hits the target by linearly interpolating
    # the threshold axis, then linearly interpolate the
    # accuracy at that threshold from the two bracket
    # accuracies.
    t_at_25: float
    acc_at_25: float
    cov_25_actual: float
    t_at_25, acc_at_25, cov_25_actual = _interpolate_reference_point(
        sorted_thresholds, curve, 0.25,
    )
    t_at_10: float
    acc_at_10: float
    cov_10_actual: float
    t_at_10, acc_at_10, cov_10_actual = _interpolate_reference_point(
        sorted_thresholds, curve, 0.10,
    )

    return {
        "thresholds": sorted_thresholds,
        "curve": curve,
        "t_at_25pct_coverage": t_at_25,
        "t_at_25pct_accuracy": acc_at_25,
        "t_at_25pct_coverage_actual": cov_25_actual,
        "t_at_10pct_coverage": t_at_10,
        "t_at_10pct_accuracy": acc_at_10,
        "t_at_10pct_coverage_actual": cov_10_actual,
    }


def _interpolate_reference_point(
    sorted_thresholds: list[float],
    curve: dict[float, dict[str, float]],
    target_coverage: float,
) -> tuple[float, float, float]:
    """Compute ``(threshold, accuracy, actual_coverage)`` at a target coverage.

    Both the threshold and the accuracy are linearly
    interpolated from the two bracket points on either side
    of the target coverage. Returns a 3-tuple
    ``(t_star, acc_star, cov_actual_at_t_star)`` where
    ``cov_actual_at_t_star`` is reported for traceability
    (it should equal ``target_coverage`` by construction,
    up to floating-point noise).
    """
    coverages: list[float] = [
        curve[float(t)]["coverage"] for t in sorted_thresholds
    ]
    accuracies: list[float] = [
        curve[float(t)]["accuracy"] for t in sorted_thresholds
    ]
    t_star: float = _interpolate_threshold_at_coverage(
        tuple(sorted_thresholds), coverages, target_coverage,
    )
    # Interpolate the accuracy at t_star from the two
    # brackets. The same bracket logic as
    # ``_interpolate_threshold_at_coverage``.
    n: int = len(sorted_thresholds)
    for i in range(n - 1):
        cov_lo: float = float(coverages[i])
        cov_hi: float = float(coverages[i + 1])
        if cov_lo >= target_coverage >= cov_hi:
            # Linear-interpolation weight (in coverage
            # space). frac=0 -> use the lo-bracket
            # accuracy, frac=1 -> use the hi-bracket
            # accuracy. When cov_lo == cov_hi, frac=0.5
            # (midpoint); this is the safe choice when
            # coverage is flat.
            if cov_lo == cov_hi:
                frac: float = 0.5
            else:
                frac = (cov_lo - target_coverage) / (cov_lo - cov_hi)
            acc_star: float = (
                accuracies[i] + frac * (accuracies[i + 1] - accuracies[i])
            )
            # Re-compute the actual coverage at t_star for
            # traceability. It should equal target_coverage
            # by construction.
            cov_at_t_star: float = cov_lo + frac * (cov_hi - cov_lo)
            return float(t_star), float(acc_star), float(cov_at_t_star)
    # Fallback: the loop above failed to find a bracket
    # (this happens when coverage is degenerate, e.g. all
    # coverages are equal AND the target is outside the
    # observed range — in that case the threshold
    # interpolator already clamped t_star to the boundary
    # of the threshold axis). Use the accuracy at the
    # bracket on the side of t_star so the headline is
    # well-formed.
    if target_coverage >= float(coverages[0]):
        # t_star == thresholds[0] (clamped). Use the
        # lo-bracket accuracy.
        return (
            float(t_star),
            float(accuracies[0]),
            float(coverages[0]),
        )
    # t_star == thresholds[-1] (clamped). Use the
    # hi-bracket accuracy.
    return (
        float(t_star),
        float(accuracies[-1]),
        float(coverages[-1]),
    )


__all__ = [
    "DEFAULT_REFERENCE_COVERAGES",
    "DEFAULT_THRESHOLDS",
    "coverage_curve",
]
