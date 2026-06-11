"""Web-only view-models for the Kairon web app.

Core-domain models (``RunResult``, ``ModelTile``, ``ProvenanceBlock``) live
in :mod:`kairon.analysis.contracts` and are re-exported here. The models
in this module are strictly *view models* — DTOs the templates render.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from kairon.analysis.contracts import ModelTile, ProvenanceBlock, RunResult

__all__ = [
    "ModelTile",
    "ProvenanceBlock",
    "RunResult",
    "TrackRow",
    "UploadedCSV",
]


class TrackRow(BaseModel):
    """A single row in the Track screen's 7-column table.

    Derived from a :class:`RunResult` + the store's verification columns.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_id: str
    asset: str
    horizon: str
    date: datetime
    predicted_pct: float
    actual_pct: float | None
    delta_pct: float | None
    status: str


class UploadedCSV(BaseModel):
    """Server-side descriptor for an uploaded CSV file."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_id: str
    filename: str
    row_count: int
    server_path: str
