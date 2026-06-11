"""Typed DTOs for the HTTP layer.

DTOs are :class:`pydantic.BaseModel` (v2, strict) so they can serve
double duty: validate HTTP request bodies and serialize responses.
The DTOs are kept decoupled from the internal types (e.g.
:class:`kairon.backtest.engine.BacktestResult`) so the HTTP contract
can evolve without forcing a downstream change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """``GET /healthz`` payload."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    status: Literal["ok", "degraded", "down"]
    version: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """``POST /v1/models/train`` payload."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbol: str = Field(min_length=1, max_length=64)
    model_backend: str = Field(min_length=1, max_length=64)
    model_config_dict: dict[str, float | int | str | bool] = Field(default_factory=dict)
    n_folds: int = Field(default=5, ge=1, le=64)
    train_size: int = Field(default=500, ge=10)
    test_size: int = Field(default=100, ge=1)
    label_kind: Literal["direction", "magnitude", "volatility", "triple_barrier"] = "direction"
    horizon: str = Field(default="1h", pattern=r"^[0-9]+(m|h|d|w)$")
    run_name: str | None = None

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        if "/" in v:
            base, quote = v.split("/", maxsplit=1)
            if not base or not quote:
                raise ValueError(f"symbol {v!r} malformed: expected BASE/QUOTE")
        return v


class FoldMetricDTO(BaseModel):
    """A single fold's metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fold_id: int
    train_rows: int
    test_rows: int
    metrics: dict[str, float]


class TrainResponse(BaseModel):
    """``POST /v1/models/train`` response."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    backend: str
    n_folds: int
    mean_test_acc: float
    mean_test_logloss: float
    train_seconds: float
    folds: list[FoldMetricDTO]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """``POST /v1/models/predict`` payload.

    The model must already have been trained via :class:`TrainRequest`
    and the artifact registered under the same ``run_name``. We
    deliberately *do not* accept feature data inline — the model
    fetches its own feature matrix from the registry. This keeps the
    request small and the schema stable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_name: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=64)
    n_rows: int = Field(default=1, ge=1, le=10_000)


class PredictResponse(BaseModel):
    """``POST /v1/models/predict`` response."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_name: str
    backend: str
    ts: list[datetime]
    y_class: list[int]
    y_proba: list[float] | None = None


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    """``POST /v1/backtest`` payload."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_name: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=64)
    horizon: str = Field(default="1h", pattern=r"^[0-9]+(m|h|d|w)$")
    n_trials: int = Field(default=1, ge=1, le=10_000)
    initial_equity: float = Field(default=10_000.0, gt=0)
    commission_bps: float = Field(default=10.0, ge=0, le=1000)
    slippage_bps: float = Field(default=2.0, ge=0, le=1000)
    half_spread_bps: float = Field(default=2.0, ge=0, le=1000)


class BacktestResponse(BaseModel):
    """``POST /v1/backtest`` response."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbol: str
    n_trades: int
    total_pnl: float
    win_rate: float
    final_equity: float
    sharpe: float
    sortino: float
    max_drawdown: float
    dsr: float
    dsr_p_value: float
    sr_star: float
    extras: dict[str, float] = Field(default_factory=dict)


__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "FoldMetricDTO",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "TrainRequest",
    "TrainResponse",
]
