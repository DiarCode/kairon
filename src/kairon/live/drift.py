"""Feature drift detection.

The model's accuracy decays in production because the *feature
distribution* drifts away from the training distribution. We track
drift with two complementary statistics:

- **Population Stability Index (PSI)** — measures the shift in a
  *binned* feature distribution. Bins come from the *reference* (train)
  sample; we count how many of the *live* sample's points fall into
  each bin and compare the proportions.
- **Kolmogorov–Smirnov (KS) two-sample** — measures the maximum
  vertical distance between two empirical CDFs. Distribution-free,
  good for continuous features.

Both return a numeric score and a *severity* band. A score of 0 means
"identical to reference"; larger means "more drift".

Reference
---------
- Yurdakul, B. (2018). *Population Stability Index in Credit Scoring*
- Berger, T. (2017). *Drift Detection in Financial Time Series*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class DriftScore:
    """The output of a single drift check."""

    feature: str
    method: str  # "psi" | "ks"
    score: float
    p_value: float | None  # KS p-value (None for PSI)
    severity: str  # "ok" | "warning" | "critical"
    n_ref: int
    n_live: int
    extras: dict[str, Any] = ...  # type: ignore[assignment]


def severity_for_psi(score: float) -> str:
    """Standard PSI bands: <0.1 ok, 0.1-0.2 warning, >0.2 critical."""
    if score < 0.1:
        return "ok"
    if score < 0.2:
        return "warning"
    return "critical"


def severity_for_ks(score: float) -> str:
    """KS bands: <0.05 ok, 0.05-0.1 warning, >0.1 critical."""
    if score < 0.05:
        return "ok"
    if score < 0.10:
        return "warning"
    return "critical"


def _build_bins(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Return n_bins+1 edges based on the reference distribution."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, qs)
    # Ensure unique edges
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([reference.min() - 1e-9, reference.max() + 1e-9])
    return edges


def population_stability_index(
    reference: np.ndarray,
    live: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute PSI = sum((live_pct - ref_pct) * ln(live_pct / ref_pct)).

    Returns the score and the (ref_pct, live_pct) per-bin arrays.
    Empty bins (ref_pct == 0) are skipped to avoid log(0).
    """
    if reference.ndim != 1 or live.ndim != 1:
        raise ValueError("PSI expects 1-D arrays")
    if reference.size < 2 or live.size < 2:
        raise ValueError("PSI needs at least 2 points in each array")
    edges = _build_bins(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)
    ref_pct = ref_counts / max(int(ref_counts.sum()), 1)
    live_pct = live_counts / max(int(live_counts.sum()), 1)
    # Add a tiny epsilon where both are zero to avoid NaN.
    eps = 1e-12
    ref_safe = np.where(ref_pct == 0, eps, ref_pct)
    live_safe = np.where(live_pct == 0, eps, live_pct)
    contrib = (live_pct - ref_pct) * np.log(live_safe / ref_safe)
    score = float(contrib.sum())
    return score, ref_pct, live_pct


def ks_two_sample(
    reference: np.ndarray,
    live: np.ndarray,
) -> tuple[float, float]:
    """Two-sided KS statistic and p-value."""
    if reference.ndim != 1 or live.ndim != 1:
        raise ValueError("KS expects 1-D arrays")
    if reference.size < 2 or live.size < 2:
        raise ValueError("KS needs at least 2 points in each array")
    res = stats.ks_2samp(reference, live)
    # scipy's ks_2samp returns a _BaseResult; narrow via Any to access .statistic
    result: Any = res
    return float(result.statistic), float(result.pvalue)


def check_drift(
    feature_name: str,
    reference: np.ndarray,
    live: np.ndarray,
    *,
    method: str = "psi",
    n_bins: int = 10,
) -> DriftScore:
    """Run a single drift check and return a :class:`DriftScore`."""
    if method == "psi":
        score, _, _ = population_stability_index(reference, live, n_bins=n_bins)
        return DriftScore(
            feature=feature_name,
            method="psi",
            score=score,
            p_value=None,
            severity=severity_for_psi(score),
            n_ref=int(reference.size),
            n_live=int(live.size),
            extras={},
        )
    if method == "ks":
        stat, p = ks_two_sample(reference, live)
        return DriftScore(
            feature=feature_name,
            method="ks",
            score=stat,
            p_value=p,
            severity=severity_for_ks(stat),
            n_ref=int(reference.size),
            n_live=int(live.size),
            extras={},
        )
    raise ValueError(f"unknown method {method!r}")


def check_drift_table(
    reference: np.ndarray,
    live: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    method: str = "psi",
    n_bins: int = 10,
) -> tuple[DriftScore, ...]:
    """Run :func:`check_drift` per column of a 2-D matrix."""
    if reference.ndim != 2 or live.ndim != 2:
        raise ValueError("drift_table expects 2-D arrays")
    if reference.shape[1] != live.shape[1]:
        raise ValueError(
            f"reference has {reference.shape[1]} features, live has {live.shape[1]}"
        )
    if reference.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries, "
            f"data has {reference.shape[1]} columns"
        )
    return tuple(
        check_drift(name, reference[:, i], live[:, i], method=method, n_bins=n_bins)
        for i, name in enumerate(feature_names)
    )


__all__ = [
    "DriftScore",
    "check_drift",
    "check_drift_table",
    "ks_two_sample",
    "population_stability_index",
    "severity_for_ks",
    "severity_for_psi",
]
