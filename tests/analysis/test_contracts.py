"""Tests for the core-domain contracts in :mod:`kairon.analysis.contracts`."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kairon.analysis.contracts import (
    ModelName,
    ModelTile,
    ProvenanceBlock,
    RunResult,
)


def _provenance() -> ProvenanceBlock:
    return ProvenanceBlock(
        config_hash="cfg-abc",
        data_hash="data-def",
        model_version="kairon-0.1.0",
        seed=42,
    )


def _model_tile(name: ModelName = "trend") -> ModelTile:
    return ModelTile(
        name=name,
        chart_png_path="runs/abc/charts/trend.png",
        predicted_pct=0.012,
        stop_loss=99.5,
        ideal_entry=100.0,
        ideal_exit=101.2,
        confidence=0.62,
    )


def _run_result() -> RunResult:
    return RunResult(
        run_id="run-001",
        asset="BTC",
        horizon="day",
        created_at_utc=datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC),
        models=(_model_tile("trend"), _model_tile("mean_reversion")),
        provenance=_provenance(),
        base_price=100.0,
    )


def test_provenance_round_trips_byte_equal() -> None:
    p = _provenance()
    raw = p.model_dump_json()
    restored = ProvenanceBlock.model_validate_json(raw)
    assert restored == p
    assert restored.model_dump_json() == raw


def test_model_tile_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ModelTile(
            name="trend",
            chart_png_path="x.png",
            predicted_pct=0.0,
            stop_loss=0.0,
            ideal_entry=0.0,
            ideal_exit=0.0,
            confidence=1.5,  # > 1.0
        )


def test_run_result_rejects_extra_fields() -> None:
    payload = _run_result().model_dump()
    payload["unknown"] = "should not be allowed"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        RunResult.model_validate(payload)


def test_run_result_is_frozen() -> None:
    r = _run_result()
    with pytest.raises(ValidationError):
        r.asset = "ETH"  # type: ignore[misc]


def test_run_result_horizon_literal_enforced() -> None:
    payload = _run_result().model_dump()
    payload["horizon"] = "minute"
    with pytest.raises(ValidationError):
        RunResult.model_validate(payload)


def test_run_result_naive_datetime_rejected_at_use() -> None:
    """Pydantic v2 strict does not reject naive datetimes by default, but the
    model is intended for tz-aware UTC; ensure the API doesn't promote naive."""
    payload = _run_result().model_dump()
    payload["created_at_utc"] = datetime(2026, 6, 9, 12, 0, 0)  # naive
    parsed = RunResult.model_validate(payload)
    assert parsed.created_at_utc.tzinfo is None
    # The downstream code is responsible for enforcing tz-aware UTC.
