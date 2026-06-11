"""Labels: direction, magnitude, volatility, triple-barrier, meta."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from kairon.labels.direction import make_direction_labels
from kairon.labels.magnitude import make_magnitude_labels
from kairon.labels.metalabel import make_metalabel_labels
from kairon.labels.schema import (
    DirectionClass,
    LabeledBar,
    LabeledFrame,
    LabelKind,
    LabelSpec,
)
from kairon.labels.triple_barrier import make_triple_barrier_labels
from kairon.labels.volatility import make_volatility_labels

if TYPE_CHECKING:
    from kairon.backtest.cost import CostModel  # noqa: F401


def make_labels(
    table: pa.Table,
    *,
    spec: LabelSpec,
    symbol: str,
    **kwargs: object,
) -> LabeledFrame:
    """Dispatch to the right label maker based on ``spec.kind``."""
    if spec.kind is LabelKind.DIRECTION:
        return make_direction_labels(
            table, spec=spec, symbol=symbol,
            flat_threshold_pct=kwargs.get("flat_threshold_pct", 0.0005),  # type: ignore[arg-type]
        )
    if spec.kind is LabelKind.MAGNITUDE:
        return make_magnitude_labels(table, spec=spec, symbol=symbol)
    if spec.kind is LabelKind.VOLATILITY:
        return make_volatility_labels(table, spec=spec, symbol=symbol)
    if spec.kind is LabelKind.TRIPLE_BARRIER:
        return make_triple_barrier_labels(
            table,
            spec=spec,
            symbol=symbol,
            pt_pct=kwargs.get("pt_pct", 0.01),  # type: ignore[arg-type]
            sl_pct=kwargs.get("sl_pct", 0.01),  # type: ignore[arg-type]
            require_finite=kwargs.get("require_finite", True),  # type: ignore[arg-type]
        )
    if spec.kind is LabelKind.META:
        # Metalabel requires a CostModel. The caller passes
        # ``cost_model=...`` via kwargs; raise a clear error if it
        # is missing rather than passing ``None`` downstream.
        cost_model = kwargs.get("cost_model")
        if cost_model is None:
            raise ValueError(
                "LabelKind.META requires a cost_model= keyword argument"
            )
        return make_metalabel_labels(
            table,
            spec=spec,
            symbol=symbol,
            cost_model=cost_model,  # type: ignore[arg-type]
            pt_pct=kwargs.get("pt_pct", 0.01),  # type: ignore[arg-type]
            sl_pct=kwargs.get("sl_pct", 0.01),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown label kind: {spec.kind}")


__all__ = [
    "DirectionClass",
    "LabelKind",
    "LabelSpec",
    "LabeledBar",
    "LabeledFrame",
    "make_direction_labels",
    "make_labels",
    "make_magnitude_labels",
    "make_metalabel_labels",
    "make_triple_barrier_labels",
    "make_volatility_labels",
]
