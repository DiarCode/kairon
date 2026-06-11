"""Splits: walk-forward, purged, embargo, CPCV."""

from __future__ import annotations

from kairon.splits.cpcv import (
    DEFAULT_K_TEST,
    DEFAULT_N_FOLDS,
    DEFAULT_PBO_THRESHOLD,
    CPCVPath,
    cpcv_paths,
)
from kairon.splits.embargo import EmbargoSpec, embargo_train_indices
from kairon.splits.purged import PurgeSpec, purge_train_indices
from kairon.splits.walkforward import (
    DEFAULT_SPLIT_1D,
    DEFAULT_SPLIT_1H,
    DEFAULT_SPLIT_5M,
    DEFAULT_SPLIT_BY_HORIZON,
    Fold,
    SplitSpec,
    walkforward,
)

__all__ = [
    "DEFAULT_K_TEST",
    "DEFAULT_N_FOLDS",
    "DEFAULT_PBO_THRESHOLD",
    "DEFAULT_SPLIT_1D",
    "DEFAULT_SPLIT_1H",
    "DEFAULT_SPLIT_5M",
    "DEFAULT_SPLIT_BY_HORIZON",
    "CPCVPath",
    "EmbargoSpec",
    "Fold",
    "PurgeSpec",
    "SplitSpec",
    "cpcv_paths",
    "embargo_train_indices",
    "purge_train_indices",
    "walkforward",
]
