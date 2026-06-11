"""Combinatorial Purged Cross-Validation (CPCV).

Used *only* to estimate the Probability of Backtest Overfitting (PBO),
not for model selection. See Bailey, Borwein, Lopez de Prado & Zhu,
"The Probability of Backtest Overfitting" (2015), and the
``mlfinlab``/``git>`` re-implementations.

CPCV takes N folds and produces ``C(N, k)`` backtest paths. The
**logit-transformed** OOS Sharpe of each path is computed; the
distribution's left tail (e.g. percentile 5) is the *Probability of
Backtest Overfitting*: P(Sharpe of best path is from overfit noise).

This is a thin scaffold: it constructs the paths but defers the
numerical computation to :mod:`kairon.evaluation.pbo` (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Final


@dataclass(frozen=True, slots=True)
class CPCVPath:
    """One backtest path through the CPCV design."""

    path_id: int
    test_fold_ids: tuple[int, ...]
    train_fold_ids: tuple[int, ...]


def cpcv_paths(n_folds: int, k_test: int = 2) -> tuple[CPCVPath, ...]:
    """Enumerate the ``C(N, k)`` backtest paths.

    Parameters
    ----------
    n_folds:
        Number of folds in the CPCV design (typically 16 in production).
    k_test:
        Number of folds held out for the test set in each path (typically 2).
    """
    if n_folds < k_test:
        raise ValueError(f"need n_folds >= k_test, got {n_folds} and {k_test}")
    if k_test < 1:
        raise ValueError(f"k_test must be >= 1, got {k_test}")
    n_paths = comb(n_folds, k_test)
    if n_paths > 100_000:  # safety valve
        raise ValueError(f"too many paths ({n_paths}); reduce n_folds or k_test")
    paths: list[CPCVPath] = []
    fold_ids = list(range(n_folds))
    for i, test_combo in enumerate(combinations(fold_ids, k_test)):
        train_combo = tuple(f for f in fold_ids if f not in test_combo)
        paths.append(CPCVPath(path_id=i, test_fold_ids=test_combo, train_fold_ids=train_combo))
    return tuple(paths)


# Reference constants used in our default CPCV design
DEFAULT_N_FOLDS: Final[int] = 16
DEFAULT_K_TEST: Final[int] = 2
DEFAULT_PBO_THRESHOLD: Final[float] = 0.10  # max acceptable PBO

__all__ = [
    "DEFAULT_K_TEST",
    "DEFAULT_N_FOLDS",
    "DEFAULT_PBO_THRESHOLD",
    "CPCVPath",
    "cpcv_paths",
]
