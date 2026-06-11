"""Tests for CPCV enumeration."""

from __future__ import annotations

from math import comb

import pytest

from kairon.splits.cpcv import (
    DEFAULT_K_TEST,
    DEFAULT_N_FOLDS,
    DEFAULT_PBO_THRESHOLD,
    cpcv_paths,
)


def test_default_constants() -> None:
    assert DEFAULT_N_FOLDS == 16
    assert DEFAULT_K_TEST == 2
    assert DEFAULT_PBO_THRESHOLD == 0.10


def test_cpcv_path_count_is_comb() -> None:
    paths = cpcv_paths(n_folds=8, k_test=2)
    assert len(paths) == comb(8, 2)


def test_cpcv_path_partition_test_and_train() -> None:
    paths = cpcv_paths(n_folds=6, k_test=2)
    for p in paths:
        assert len(p.test_fold_ids) == 2
        assert len(p.train_fold_ids) == 4
        assert set(p.test_fold_ids) | set(p.train_fold_ids) == {0, 1, 2, 3, 4, 5}
        assert set(p.test_fold_ids) & set(p.train_fold_ids) == set()


def test_cpcv_path_ids_are_unique() -> None:
    paths = cpcv_paths(n_folds=6, k_test=2)
    assert len({p.path_id for p in paths}) == len(paths)


def test_cpcv_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        cpcv_paths(n_folds=2, k_test=3)
    with pytest.raises(ValueError):
        cpcv_paths(n_folds=4, k_test=0)


def test_cpcv_safety_valve_rejects_huge_combos() -> None:
    with pytest.raises(ValueError, match="too many paths"):
        cpcv_paths(n_folds=50, k_test=25)
