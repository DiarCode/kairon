"""Purging: remove train samples whose label overlap with test labels.

Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class PurgeSpec:
    """How aggressive to be when purging."""

    label_overlap_seconds: int = 0  # 0 == no overlap; 300 == 5min etc.
    min_gap_bars: int = 0  # extra bars to remove on either side

    def __post_init__(self) -> None:
        if self.label_overlap_seconds < 0:
            raise ValueError(
                f"label_overlap_seconds must be >= 0, got {self.label_overlap_seconds}"
            )
        if self.min_gap_bars < 0:
            raise ValueError(f"min_gap_bars must be >= 0, got {self.min_gap_bars}")


def purge_train_indices(
    *,
    train_ts: Sequence[datetime],
    test_ts: Sequence[datetime],
    spec: PurgeSpec,
) -> set[int]:
    """Return the set of train indices to drop because they overlap with test.

    A train sample at ``train_ts[i]`` is purged if its label window
    ``[train_ts[i], train_ts[i] + label_overlap_seconds]`` intersects
    any test window ``[test_ts[j], test_ts[j] + label_overlap_seconds]``.
    """
    if not test_ts or not train_ts:
        return set()
    overlap = spec.label_overlap_seconds
    drop: set[int] = set()
    for i, t in enumerate(train_ts):
        window_start = t
        window_end = t
        if overlap:
            window_end = t + timedelta(seconds=overlap)
        for u in test_ts:
            u_end = u + timedelta(seconds=overlap) if overlap else u
            if window_start <= u_end and u <= window_end:
                drop.add(i)
                break
    return drop
