"""Label schema: typed specs and containers.

A ``LabelSpec`` is a complete description of "what to predict": the
horizon (e.g. ``5m``, ``1h``), the kind (``direction``, ``magnitude``,
``volatility``, ``triple_barrier``), and the kind-specific parameters.

Labels themselves are computed in the per-kind submodules
(:mod:`direction`, :mod:`magnitude`, :mod:`volatility`,
:mod:`triple_barrier`) and stored in ``LabeledBars`` containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LabelKind(str, Enum):
    """The kind of target we train on."""

    DIRECTION = "direction"
    MAGNITUDE = "magnitude"
    VOLATILITY = "volatility"
    TRIPLE_BARRIER = "triple_barrier"
    META = "meta"


class DirectionClass(int, Enum):
    """The 3-class direction target (used by direction & triple_barrier)."""

    DOWN = -1
    FLAT = 0
    UP = 1


class LabelSpec(BaseModel):
    """A fully-typed label specification.

    The same ``LabelSpec`` is what the trainer reads to know what to
    build, what the policy uses to know what to predict, and what the
    evaluator uses to know how to score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: LabelKind
    horizon: str = Field(pattern=r"^[0-9]+(m|h|d|w)$")  # e.g. "5m", "1h", "1d"
    params: dict[str, float] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _validate_params(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if val != val:  # NaN guard
                raise ValueError(f"LabelSpec params[{key!r}] is NaN")
        return v

    @property
    def horizon_seconds(self) -> int:
        """Number of seconds in the horizon (5m=300, 1h=3600, 1d=86400)."""
        from kairon.data.diagnostics import timeframe_to_timedelta

        return timeframe_to_timedelta(self.horizon)


@dataclass(frozen=True, slots=True)
class LabeledBar:
    """A single bar plus its label(s).

    The label is intentionally a union type: ``int`` for direction,
    ``float`` for magnitude/volatility, or a richer dataclass for
    triple-barrier. Callers narrow the type by the ``LabelSpec.kind``.
    """

    symbol: str
    ts: datetime  # must be UTC, tz-aware
    horizon: str
    kind: LabelKind
    y: int | float
    y_class: int | None = None
    meta: dict[str, float | str | int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LabeledFrame:
    """A batch of labeled bars plus the spec that produced them."""

    spec: LabelSpec
    symbol: str
    bars: tuple[LabeledBar, ...]
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def ys(self) -> tuple[int | float, ...]:
        return tuple(b.y for b in self.bars)

    @property
    def y_classes(self) -> tuple[int | None, ...]:
        return tuple(b.y_class for b in self.bars)

    @property
    def tss(self) -> tuple[datetime, ...]:
        return tuple(b.ts for b in self.bars)


__all__ = [
    "DirectionClass",
    "LabelKind",
    "LabelSpec",
    "LabeledBar",
    "LabeledFrame",
]
