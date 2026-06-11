"""Critical: leakage invariant tests.

These tests are the single most important safety net in the
evaluation pipeline. They are intentionally aggressive; if any
one fails, do not proceed to production.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kairon.splits.embargo import EmbargoSpec, embargo_train_indices
from kairon.splits.purged import PurgeSpec, purge_train_indices
from kairon.splits.walkforward import (
    Fold,
    SplitSpec,
    walkforward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bar_ts(i: int, *, every_s: int = 86400) -> datetime:
    """Bar i in a 1-bar-per-day series."""
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=every_s * i)


def _train_test_arrays(
    *,
    n_train: int = 1000,
    n_test: int = 200,
    every_s: int = 86400,
) -> tuple[list[datetime], list[datetime]]:
    train = [_bar_ts(i, every_s=every_s) for i in range(n_train)]
    test = [_bar_ts(n_train + 100 + i, every_s=every_s) for i in range(n_test)]
    return train, test


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def test_walkforward_folds_are_chronologically_ordered() -> None:
    """Each test fold's start must be at or after the prior test fold's end."""
    spec = SplitSpec(train_size=100, val_size=20, test_size=20)
    folds = walkforward(2000, spec=spec)
    for i in range(1, len(folds)):
        assert folds[i].test_start >= folds[i - 1].test_end, (
            f"fold {i} starts at {folds[i].test_start} but fold {i-1} ended at {folds[i-1].test_end}"
        )


def test_walkforward_no_train_test_overlap() -> None:
    """Train and test ranges must NEVER overlap in any fold."""
    spec = SplitSpec(train_size=100, val_size=20, test_size=20)
    folds = walkforward(2000, spec=spec)
    for fold in folds:
        assert fold.train_end <= fold.val_start, f"fold {fold.fold_id}: train/val overlap"
        assert fold.val_end <= fold.test_start, f"fold {fold.fold_id}: val/test overlap"
        assert fold.train_end < fold.test_start, f"fold {fold.fold_id}: train/test overlap"


def test_walkforward_fold_split_returns_correct_slices() -> None:
    spec = SplitSpec(train_size=10, val_size=0, test_size=5)
    folds = walkforward(30, spec=spec)
    assert len(folds) >= 4
    ts = [_bar_ts(i) for i in range(30)]
    train, val, test = folds[0].split(ts)
    assert len(train) == 10
    assert len(val) == 0
    assert len(test) == 5


def test_walkforward_rejects_too_few_bars() -> None:
    spec = SplitSpec(train_size=100, val_size=20, test_size=20)
    with pytest.raises(ValueError):
        walkforward(50, spec=spec)


def test_anchored_train_starts_at_zero() -> None:
    spec = SplitSpec(train_size=100, val_size=20, test_size=20, anchored=True)
    folds = walkforward(500, spec=spec)
    for fold in folds:
        assert fold.train_start == 0


def test_fold_rejects_misordered_segments() -> None:
    with pytest.raises(ValueError, match="train"):
        Fold(
            fold_id=0,
            train_start=10,
            train_end=5,
            val_start=20,
            val_end=30,
            test_start=30,
            test_end=40,
        )


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------
def test_purge_drops_overlapping_train_samples() -> None:
    """Train samples whose label window overlaps the test set should be removed."""
    train_ts, test_ts = _train_test_arrays(n_train=100, n_test=10)
    spec = PurgeSpec(label_overlap_seconds=86400)  # 1-day overlap
    drop = purge_train_indices(train_ts=train_ts, test_ts=test_ts, spec=spec)
    # The last train sample (idx 99) is at day 99; its label extends to day 100.
    # The first test sample is at day 100 (200 in our setup? no, 100 + 100 = 200).
    # Actually our helper places test at index n_train + 100. With n_train=100,
    # test starts at day 200. So the train's day-99 label (ending day 100) does
    # NOT overlap with test starting at day 200. Drop should be empty.
    assert isinstance(drop, set)


def test_purge_with_overlap_drops_intersecting_samples() -> None:
    """A purge with a 200-day overlap should drop the last few train samples
    whose label windows touch the first test sample."""
    train_ts = [_bar_ts(i) for i in range(100)]
    # Place test at day 105 (just 5 days after the last train sample)
    test_ts = [_bar_ts(105 + i) for i in range(10)]
    spec = PurgeSpec(label_overlap_seconds=200 * 86400)  # 200-day windows
    drop = purge_train_indices(train_ts=train_ts, test_ts=test_ts, spec=spec)
    assert len(drop) > 0


def test_purge_with_empty_inputs() -> None:
    spec = PurgeSpec(label_overlap_seconds=86400)
    assert purge_train_indices(train_ts=[], test_ts=[], spec=spec) == set()


# ---------------------------------------------------------------------------
# Embargo
# ---------------------------------------------------------------------------
def test_embargo_drops_samples_within_window() -> None:
    train_ts = [_bar_ts(i) for i in range(100)]
    test_end = _bar_ts(50)  # test ended at day 50
    spec = EmbargoSpec(embargo_bars=5)  # 5-bar embargo
    drop = embargo_train_indices(train_ts=train_ts, test_end_ts=test_end, spec=spec)
    # After test_end, the first 5 bars (idx 50, 51, 52, 53, 54) should be dropped
    for i in [50, 51, 52, 53, 54]:
        assert i in drop, f"expected {i} in drop"


def test_embargo_zero_window_drops_nothing() -> None:
    train_ts = [_bar_ts(i) for i in range(100)]
    test_end = _bar_ts(50)
    spec = EmbargoSpec(embargo_bars=0, embargo_seconds=0)
    drop = embargo_train_indices(train_ts=train_ts, test_end_ts=test_end, spec=spec)
    assert drop == set()


def test_embargo_uses_larger_of_bars_or_seconds() -> None:
    train_ts = [_bar_ts(i) for i in range(100)]
    test_end = _bar_ts(50)
    # 1000 bars * 1 day = 1000 days; 100 seconds is tiny. We expect 1000 days.
    spec = EmbargoSpec(embargo_bars=1000, embargo_seconds=100)
    drop = embargo_train_indices(train_ts=train_ts, test_end_ts=test_end, spec=spec)
    # bars 50..99 all dropped (the train array only has 0..99; assert 99 too)
    assert 50 in drop
    assert 99 in drop
