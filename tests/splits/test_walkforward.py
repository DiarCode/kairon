"""Tests for walk-forward + splits."""

from __future__ import annotations

import pytest

from kairon.splits.walkforward import (
    DEFAULT_SPLIT_1D,
    DEFAULT_SPLIT_1H,
    DEFAULT_SPLIT_5M,
    DEFAULT_SPLIT_BY_HORIZON,
    SplitSpec,
    walkforward,
)


def test_default_split_by_horizon() -> None:
    assert DEFAULT_SPLIT_BY_HORIZON["5m"] is DEFAULT_SPLIT_5M
    assert DEFAULT_SPLIT_BY_HORIZON["1h"] is DEFAULT_SPLIT_1H
    assert DEFAULT_SPLIT_BY_HORIZON["1d"] is DEFAULT_SPLIT_1D


def test_split_spec_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        SplitSpec(train_size=0, val_size=0, test_size=10)
    with pytest.raises(ValueError):
        SplitSpec(train_size=10, val_size=-1, test_size=10)
    with pytest.raises(ValueError):
        SplitSpec(train_size=10, val_size=0, test_size=0)
    with pytest.raises(ValueError):
        SplitSpec(train_size=10, val_size=0, test_size=10, purge_bars=-1)


def test_walkforward_uses_default_when_none() -> None:
    folds = walkforward(5000)  # uses DEFAULT_SPLIT_1D
    assert all(f.test_end <= 5000 for f in folds)


def test_walkforward_increments_fold_id() -> None:
    spec = SplitSpec(train_size=50, val_size=0, test_size=10)
    folds = walkforward(200, spec=spec)
    assert [f.fold_id for f in folds] == list(range(len(folds)))


def test_walkforward_fold_split_zero_val() -> None:
    spec = SplitSpec(train_size=10, val_size=0, test_size=5)
    folds = walkforward(40, spec=spec)
    _train, val, _test = folds[0].split([None] * 40)  # type: ignore[list-item]
    assert len(val) == 0
