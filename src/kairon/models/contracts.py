"""Feature matrix contract.

The model layer speaks ``FeatureMatrix``; the rest of the codebase speaks
``pa.Table``. This adapter keeps both worlds typed without leaking pyarrow
into the model APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A typed, in-memory feature matrix.

    ``values`` is a 2-D ``float64`` array. ``feature_names`` records the
    column order. ``ts`` is an optional 1-D array of timestamps (any
    sortable type) — used by regime-aware models and live inference but
    ignored by tabular backends.
    """

    values: np.ndarray  # shape (n_rows, n_features), float64
    feature_names: tuple[str, ...]
    ts: np.ndarray | None = None  # shape (n_rows,) — datetime64[ns] or None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {self.values.shape}")
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"values has {self.values.shape[1]} columns, "
                f"feature_names has {len(self.feature_names)}"
            )
        if self.ts is not None and self.ts.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"ts has {self.ts.shape[0]} rows, "
                f"values has {self.values.shape[0]}"
            )

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def select(self, names: tuple[str, ...] | list[str]) -> FeatureMatrix:
        """Return a new ``FeatureMatrix`` with only the given columns."""
        name_set = set(names)
        if len(name_set) != len(names):
            raise ValueError(f"duplicate names in selection: {names}")
        missing = [n for n in names if n not in self.feature_names]
        if missing:
            raise ValueError(f"unknown feature names: {missing}")
        idxs = [self.feature_names.index(n) for n in names]
        return FeatureMatrix(
            values=self.values[:, idxs],
            feature_names=tuple(names),
            ts=self.ts,
        )

    def to_pandas(self) -> PandasDataFrame:
        import pandas as pd

        return pd.DataFrame(self.values, columns=list(self.feature_names))  # type: ignore[arg-type]


def ensure_feature_matrix(
    source: pa.Table | FeatureMatrix | dict[str, np.ndarray],
    *,
    required_features: tuple[str, ...] | None = None,
    drop_non_float: bool = True,
) -> FeatureMatrix:
    """Convert any of the supported sources to a ``FeatureMatrix``.

    Raises ``ValueError`` if a required feature is missing or a column
    cannot be coerced to ``float64``. Non-float columns are dropped
    unless ``drop_non_float=False``.
    """
    if isinstance(source, FeatureMatrix):
        if required_features is not None:
            return source.select(required_features)
        return source

    pa_table: pa.Table | None = source if isinstance(source, pa.Table) else None
    if pa_table is not None:
        names, arrays = _arrow_to_arrays(pa_table, drop_non_float=drop_non_float)
    elif isinstance(source, dict):
        names, arrays = _dict_to_arrays(source, drop_non_float=drop_non_float)
    else:
        raise TypeError(f"unsupported source type: {type(source).__name__}")

    if not names:
        raise ValueError("no usable numeric features in source")

    values = np.stack(arrays, axis=1).astype(np.float64, copy=False)
    if required_features is not None:
        missing = [n for n in required_features if n not in names]
        if missing:
            raise ValueError(f"missing required features: {missing}")
        idxs = [names.index(n) for n in required_features]
        values = values[:, idxs]
        names = list(required_features)

    ts_arr: np.ndarray | None = None
    if pa_table is not None and "ts" in pa_table.column_names:
        ts_arr = pa_table.column("ts").to_numpy(zero_copy_only=False)

    return FeatureMatrix(
        values=values,
        feature_names=tuple(names),
        ts=ts_arr,
    )


def _arrow_to_arrays(
    table: pa.Table,
    *,
    drop_non_float: bool,
) -> tuple[list[str], list[np.ndarray]]:
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for col in table.column_names:
        if col == "ts":
            continue
        arr = table.column(col)
        if _is_float_like(arr.type):
            names.append(col)
            arrays.append(arr.to_numpy(zero_copy_only=False))
        elif not drop_non_float:
            names.append(col)
            arrays.append(arr.to_numpy(zero_copy_only=False).astype(np.float64))
    return names, arrays


def _dict_to_arrays(
    payload: dict[str, np.ndarray],
    *,
    drop_non_float: bool,
) -> tuple[list[str], list[np.ndarray]]:
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for k, v in payload.items():
        if k == "ts":
            continue
        arr = np.asarray(v)
        if arr.dtype.kind == "f" or arr.dtype.kind == "i":
            names.append(k)
            arrays.append(arr)
        elif not drop_non_float:
            names.append(k)
            arrays.append(arr.astype(np.float64))
    return names, arrays


def _is_float_like(t: pa.DataType) -> bool:
    if pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_boolean(t):
        return True
    return False


def is_classification(classes: tuple[int, ...] | None) -> bool:
    return classes is not None


def is_regression(classes: tuple[int, ...] | None) -> bool:
    return classes is None


def feature_diff(
    a: tuple[str, ...],
    b: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Return ``{"only_in_a", "only_in_b", "common"}`` between two name tuples."""
    sa, sb = set(a), set(b)
    return {
        "only_in_a": tuple(sorted(sa - sb)),
        "only_in_b": tuple(sorted(sb - sa)),
        "common": tuple(sorted(sa & sb)),
    }


__all__ = [
    "FeatureMatrix",
    "ensure_feature_matrix",
    "feature_diff",
    "is_classification",
    "is_regression",
]


# ---------------------------------------------------------------------------
# W6.4: prediction wire-format documentation
# ---------------------------------------------------------------------------
# The :class:`kairon.models.base.Prediction` dataclass (defined in
# :mod:`kairon.models.base` to keep the model class hierarchy in one
# place) is extended with two optional fields per the W6.4 PRD:
#
#   - y_magnitude: np.ndarray | None = None
#       1-D float64 array of length n. The magnitude head's
#       regression output (predicted log-return for the horizon).
#       Populated by the W6.4 :class:`MultiHeadModel`; ``None`` for
#       every v1 backend.
#   - y_vol: np.ndarray | None = None
#       1-D float64 array of length n. The vol head's quantile
#       (pinball) prediction. With the W6.4 default alpha=0.5 this
#       is the median vol forecast. ``None`` for every v1 backend.
#
# The :class:`Prediction` dataclass in :mod:`kairon.models.base`
# applies the same 1-D / shape-aligned validation as the existing
# ``y_proba`` and ``y_score`` fields. These names are re-exported
# here so callers that import ``Prediction`` from either
# :mod:`kairon.models.contracts` (the wire-format module) or
# :mod:`kairon.models.base` (the class-hierarchy module) see the
# same dataclass instance.
PREDICTION_W64_FIELDS_DOC: str = (
    "The Prediction dataclass exposes two additive W6.4 fields: "
    "y_magnitude (np.ndarray | None) is the magnitude head's "
    "regression output, and y_vol (np.ndarray | None) is the vol "
    "head's pinball-quantile forecast. Both default to None and "
    "are only populated by the W6.4 MultiHeadModel."
)
