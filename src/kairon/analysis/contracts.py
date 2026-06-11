"""Core-domain contracts for the Kairon web app.

These are the canonical pydantic models for analysis output. They live in
``kairon.analysis.contracts`` (not under ``kairon.ui.web``) because they
are core-domain entities: any consumer of the engine — CLI, web, future
notebooks — should depend on the same types.

Constraints (per ``AGENTS.md``):
- pydantic v2, ``frozen=True``, ``extra='forbid'``, ``strict=True``
- All timestamps UTC tz-aware
- Provenance on every output (per project invariant #7)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HorizonName = Literal["day", "swing", "long"]
ModelName = Literal["trend", "mean_reversion", "volatility", "ensemble"]
VerificationStatus = Literal["pending", "hit", "missed"]


class ProvenanceBlock(BaseModel):
    """Reproducibility metadata for an analysis output (project invariant #7)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    config_hash: str
    data_hash: str
    model_version: str
    seed: int


class ModelTile(BaseModel):
    """A single model tile in the Result bento.

    ``name`` is the spec-facing name (one of four). The engine internally
    emits two raw model heads (``lr`` and ``tree``); the rebadging is
    performed by :func:`kairon.analysis.engine.build_run_result`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: ModelName
    chart_png_path: str
    predicted_pct: float
    stop_loss: float
    ideal_entry: float
    ideal_exit: float
    confidence: float = Field(ge=0.0, le=1.0)


class RunResult(BaseModel):
    """A complete analysis run, persisted to ``RunStore`` and rendered on Result/Track."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_id: str
    asset: str
    horizon: HorizonName
    created_at_utc: datetime
    models: tuple[ModelTile, ...]
    provenance: ProvenanceBlock
    base_price: float


__all__ = [
    "HorizonName",
    "ModelName",
    "ModelTile",
    "ProvenanceBlock",
    "RunResult",
    "VerificationStatus",
]
