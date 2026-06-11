"""Tests for the API DTOs (pydantic v2 strict)."""

from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from kairon.api.dto import (
    BacktestRequest,
    BacktestResponse,
    FoldMetricDTO,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)


def test_health_response_ok() -> None:
    h = HealthResponse(status="ok", version="0.1.0", uptime_seconds=12.5)
    assert h.status == "ok"
    assert h.uptime_seconds == 12.5


def test_health_response_rejects_bad_status() -> None:
    with pytest.raises(ValidationError):
        HealthResponse(status="great", version="0.1.0", uptime_seconds=1.0)  # type: ignore[arg-type]


def train_request_valid() -> TrainRequest:
    return TrainRequest(
        symbol="BTC/USDT",
        model_backend="logreg",
        model_config_dict={"C": 1.0},
        n_folds=3,
        train_size=200,
        test_size=50,
    )


def test_train_request_valid() -> None:
    req = train_request_valid()
    assert req.symbol == "BTC/USDT"
    assert req.horizon == "1h"  # default
    assert req.label_kind == "direction"  # default


def test_train_request_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        TrainRequest(
            symbol="BTC/USDT",
            model_backend="logreg",
            label_kind="made_up",  # type: ignore[arg-type]
        )


def test_train_request_rejects_bad_horizon() -> None:
    with pytest.raises(ValidationError):
        TrainRequest(symbol="BTC/USDT", model_backend="logreg", horizon="xyz")


def test_train_request_rejects_zero_folds() -> None:
    with pytest.raises(ValidationError):
        TrainRequest(symbol="BTC/USDT", model_backend="logreg", n_folds=0)


def test_train_request_rejects_malformed_symbol() -> None:
    with pytest.raises(ValidationError):
        TrainRequest(symbol="/USDT", model_backend="logreg")


def test_train_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TrainRequest.model_validate(
            {
                "symbol": "BTC/USDT",
                "model_backend": "logreg",
                "rogue_field": True,  # extra="forbid"
            }
        )


def test_fold_metric_dto() -> None:
    fm = FoldMetricDTO(fold_id=0, train_rows=200, test_rows=50, metrics={"acc": 0.6})
    assert fm.metrics["acc"] == 0.6


def test_train_response_round_trip() -> None:
    r = TrainResponse(
        backend="logreg",
        n_folds=3,
        mean_test_acc=0.6,
        mean_test_logloss=0.5,
        train_seconds=12.0,
        folds=[
            FoldMetricDTO(fold_id=0, train_rows=200, test_rows=50, metrics={"acc": 0.6}),
        ],
    )
    d = r.model_dump()
    assert d["backend"] == "logreg"
    assert d["folds"][0]["metrics"]["acc"] == 0.6


def test_predict_request_defaults() -> None:
    p = PredictRequest(run_name="abc", symbol="BTC/USDT")
    assert p.n_rows == 1


def test_predict_request_rejects_zero_rows() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(run_name="abc", symbol="BTC/USDT", n_rows=0)


def test_predict_response_basic() -> None:
    from datetime import datetime

    r = PredictResponse(
        run_name="abc",
        backend="logreg",
        ts=[datetime.now(UTC)],
        y_class=[1],
        y_proba=[0.7],
    )
    assert r.y_class == [1]


def test_backtest_request_defaults() -> None:
    b = BacktestRequest(run_name="r1", symbol="BTC/USDT")
    assert b.n_trials == 1
    assert b.initial_equity == 10_000.0


def test_backtest_request_rejects_negative_equity() -> None:
    with pytest.raises(ValidationError):
        BacktestRequest(run_name="r1", symbol="BTC/USDT", initial_equity=-100.0)


def test_backtest_response_basic() -> None:
    b = BacktestResponse(
        symbol="BTC/USDT",
        n_trades=10,
        total_pnl=1500.0,
        win_rate=0.6,
        final_equity=11_500.0,
        sharpe=1.5,
        sortino=2.0,
        max_drawdown=-0.1,
        dsr=0.95,
        dsr_p_value=0.05,
        sr_star=0.5,
    )
    assert b.total_pnl == 1500.0


def test_dtos_are_frozen() -> None:
    r = HealthResponse(status="ok", version="0.1.0", uptime_seconds=1.0)
    with pytest.raises(ValidationError):
        r.status = "down"  # type: ignore[misc]
