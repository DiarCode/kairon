"""Tests for purge and embargo helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kairon.splits.embargo import EmbargoSpec
from kairon.splits.purged import PurgeSpec, purge_train_indices


def _ts(i: int, every_s: int = 86400) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=every_s * i)


def test_purge_spec_rejects_negative() -> None:
    with pytest.raises(ValueError):
        PurgeSpec(label_overlap_seconds=-1)


def test_purge_zero_overlap() -> None:
    train = [_ts(i) for i in range(10)]
    test = [_ts(20 + i) for i in range(5)]
    drop = purge_train_indices(train_ts=train, test_ts=test, spec=PurgeSpec(label_overlap_seconds=0))
    assert drop == set()


def test_embargo_spec_default() -> None:
    s = EmbargoSpec()
    assert s.embargo_bars == 0
    assert s.embargo_seconds == 0
