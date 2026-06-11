"""Tests for model contracts (FeatureMatrix, ensure_feature_matrix, etc)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from kairon.models.contracts import (
    FeatureMatrix,
    ensure_feature_matrix,
    feature_diff,
    is_classification,
    is_regression,
)


def test_feature_matrix_validates_shape() -> None:
    with pytest.raises(ValueError, match="values must be 2-D"):
        FeatureMatrix(values=np.zeros(10), feature_names=("a",))
    with pytest.raises(ValueError, match="values has"):
        FeatureMatrix(
            values=np.zeros((10, 3)),
            feature_names=("a", "b"),  # 2 vs 3
        )
    with pytest.raises(ValueError, match="ts has"):
        FeatureMatrix(
            values=np.zeros((10, 2)),
            feature_names=("a", "b"),
            ts=np.zeros(5),  # wrong length
        )


def test_feature_matrix_select() -> None:
    fm = FeatureMatrix(
        values=np.arange(12, dtype=np.float64).reshape(3, 4),
        feature_names=("a", "b", "c", "d"),
    )
    sub = fm.select(("c", "a"))
    assert sub.feature_names == ("c", "a")
    assert sub.values.shape == (3, 2)
    assert sub.values[0, 0] == 2  # c
    assert sub.values[0, 1] == 0  # a


def test_feature_matrix_select_rejects_duplicates() -> None:
    fm = FeatureMatrix(
        values=np.zeros((3, 2)),
        feature_names=("a", "b"),
    )
    with pytest.raises(ValueError, match="duplicate names"):
        fm.select(("a", "a"))


def test_feature_matrix_select_rejects_unknown() -> None:
    fm = FeatureMatrix(
        values=np.zeros((3, 2)),
        feature_names=("a", "b"),
    )
    with pytest.raises(ValueError, match="unknown feature names"):
        fm.select(("a", "zzz"))


def test_ensure_feature_matrix_from_pa_table() -> None:
    table = pa.table(
        {
            "ts": pa.array([1, 2, 3], type=pa.timestamp("ns")),
            "a": pa.array([1.0, 2.0, 3.0]),
            "b": pa.array([4, 5, 6], type=pa.int64()),
            "label": pa.array(["x", "y", "z"]),  # non-numeric, should be dropped
        }
    )
    fm = ensure_feature_matrix(table)
    assert fm.feature_names == ("a", "b")
    assert fm.ts is not None
    assert fm.ts.shape == (3,)


def test_ensure_feature_matrix_required_features() -> None:
    table = pa.table({"a": pa.array([1.0, 2.0]), "b": pa.array([3.0, 4.0])})
    fm = ensure_feature_matrix(table, required_features=("b", "a"))
    assert fm.feature_names == ("b", "a")
    assert fm.values[0, 0] == 3.0


def test_ensure_feature_matrix_required_missing() -> None:
    table = pa.table({"a": pa.array([1.0])})
    with pytest.raises(ValueError, match="missing required features"):
        ensure_feature_matrix(table, required_features=("a", "z"))


def test_ensure_feature_matrix_no_features() -> None:
    table = pa.table({"label": pa.array(["x"])})
    with pytest.raises(ValueError, match="no usable numeric features"):
        ensure_feature_matrix(table)


def test_ensure_feature_matrix_from_existing() -> None:
    fm = FeatureMatrix(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        feature_names=("a", "b"),
    )
    out = ensure_feature_matrix(fm, required_features=("b",))
    assert out.feature_names == ("b",)
    assert out.values[0, 0] == 2.0


def test_ensure_feature_matrix_from_dict() -> None:
    payload = {
        "a": np.array([1.0, 2.0]),
        "b": np.array([3, 4], dtype=np.int64),
    }
    fm = ensure_feature_matrix(payload)
    assert fm.feature_names == ("a", "b")
    assert fm.values.shape == (2, 2)


def test_ensure_feature_matrix_rejects_bad_type() -> None:
    with pytest.raises(TypeError, match="unsupported source type"):
        ensure_feature_matrix("not a table")  # type: ignore[arg-type]


def test_feature_diff() -> None:
    a = ("x", "y", "z")
    b = ("y", "w")
    diff = feature_diff(a, b)
    assert set(diff["only_in_a"]) == {"x", "z"}
    assert set(diff["only_in_b"]) == {"w"}
    assert set(diff["common"]) == {"y"}


def test_is_classification_and_regression() -> None:
    assert is_classification((0, 1, 2))
    assert not is_classification(None)
    assert is_regression(None)
    assert not is_regression((0, 1))
