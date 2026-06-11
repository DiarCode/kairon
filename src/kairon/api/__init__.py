"""API layer: typed DTOs and the FastAPI app.

The HTTP layer is *optional* — the rest of the codebase can import
this module without FastAPI being installed. Importing
``kairon.api.app`` will raise :class:`ImportError` if FastAPI isn't
available; the DTOs and schemas (``kairon.api.dto``) are always
importable.
"""

from __future__ import annotations

from kairon.api.dto import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "TrainRequest",
    "TrainResponse",
]
