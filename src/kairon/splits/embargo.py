"""Embargo: an additional gap after a test set before train resumes.

Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7.

The embargo is computed as the maximum serial-correlation horizon
across all features, expressed in bars; we provide a sane default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class EmbargoSpec:
    """Configuration for the post-test embargo."""

    embargo_bars: int = 0
    # Or expressed as a time delta; the larger of the two is used.
    embargo_seconds: int = 0


def embargo_train_indices(
    *,
    train_ts: Sequence[datetime],
    test_end_ts: datetime,
    spec: EmbargoSpec,
) -> set[int]:
    """Return the set of train indices to drop because they are within
    the embargo window after the test set.

    A train sample at ``train_ts[i]`` is embargoed if
    ``train_ts[i] - test_end_ts < max(embargo_bars * median_dt, embargo_seconds)``.
    """
    if not train_ts or (spec.embargo_bars == 0 and spec.embargo_seconds == 0):
        return set()
    # Compute the time-based embargo: max(seconds, bars * median_dt)
    embargo_td = timedelta(seconds=spec.embargo_seconds)
    if spec.embargo_bars > 0 and len(train_ts) >= 2:
        diffs = [
            (train_ts[i + 1] - train_ts[i]).total_seconds() for i in range(len(train_ts) - 1)
        ]
        if diffs:
            median_dt = sorted(diffs)[len(diffs) // 2]
            bar_td = timedelta(seconds=spec.embargo_bars * median_dt)
            embargo_td = max(embargo_td, bar_td)
    drop: set[int] = set()
    for i, t in enumerate(train_ts):
        if t - test_end_ts < embargo_td:
            drop.add(i)
    return drop
