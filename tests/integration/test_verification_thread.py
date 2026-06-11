"""Tests for the in-process verifier (US-004)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kairon.analysis.contracts import (
    ModelTile,
    ProvenanceBlock,
    RunResult,
)
from kairon.store.runs import RunStore
from kairon.store.verifier import HIT_TOLERANCE_PCT, run_once


def _run(
    *,
    run_id: str = "run-1",
    asset: str = "BTCUSDT",
    horizon: str = "day",
    created_at: datetime | None = None,
    base_price: float = 100.0,
    predicted_pct: float = 0.01,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        asset=asset,
        horizon=horizon,  # type: ignore[arg-type]
        created_at_utc=created_at or datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC),
        models=(
            ModelTile(
                name="trend",
                chart_png_path=f"runs/{run_id}/charts/trend.png",
                predicted_pct=predicted_pct,
                stop_loss=99.0,
                ideal_entry=100.0,
                ideal_exit=101.0,
                confidence=0.6,
            ),
        ),
        provenance=ProvenanceBlock(
            config_hash="c", data_hash="d", model_version="kairon-0.1.0", seed=42
        ),
        base_price=base_price,
    )


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    s = RunStore(tmp_path / "runs.db")
    try:
        yield s
    finally:
        s.close()


def test_run_once_verifies_a_due_run(store: RunStore, tmp_path: Path) -> None:
    now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    r = _run(
        run_id="day-old",
        horizon="day",
        created_at=now - timedelta(hours=25),
        base_price=100.0,
        predicted_pct=0.01,
    )
    store.create(r, tmp_path / "input.csv")

    # fetch returns 101 -> actual_pct = 0.01, delta = 0.0 -> hit
    n = run_once(
        store,
        fetch_price_fn=lambda asset, venue: 101.0,
        now_utc=now,
        base_price_reader=lambda run, path: 100.0,
    )
    assert n == 1
    v = store.get_verification("day-old")
    assert v is not None
    actual, delta, status, ts = v
    assert abs(actual - 0.01) < 1e-9
    assert abs(delta) < 1e-9
    assert status == "hit"
    assert ts == now


def test_run_once_marks_missed_when_delta_exceeds_tolerance(
    store: RunStore, tmp_path: Path
) -> None:
    now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    r = _run(
        run_id="day-old",
        horizon="day",
        created_at=now - timedelta(hours=25),
        base_price=100.0,
        predicted_pct=0.01,
    )
    store.create(r, tmp_path / "input.csv")

    # fetch returns 110 -> actual_pct = 0.10, delta = 0.09 > HIT_TOLERANCE_PCT -> missed
    n = run_once(
        store,
        fetch_price_fn=lambda asset, venue: 110.0,
        now_utc=now,
        base_price_reader=lambda run, path: 100.0,
    )
    assert n == 1
    v = store.get_verification("day-old")
    assert v is not None
    assert v[2] == "missed"


def test_run_once_skips_fresh_runs(store: RunStore, tmp_path: Path) -> None:
    now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    fresh = _run(
        run_id="day-fresh",
        horizon="day",
        created_at=now - timedelta(hours=23),  # 1h short of Day window
    )
    store.create(fresh, tmp_path / "fresh.csv")
    n = run_once(
        store,
        fetch_price_fn=lambda asset, venue: 100.0,
        now_utc=now,
        base_price_reader=lambda run, path: 100.0,
    )
    assert n == 0
    assert store.get_verification("day-fresh") is None


def test_run_once_rejects_naive_now(store: RunStore) -> None:
    naive_now = datetime(2026, 6, 10, 12, 0, 0)  # no tz
    with pytest.raises(ValueError, match="tz-aware UTC"):
        run_once(
            store,
            fetch_price_fn=lambda asset, venue: 100.0,
            now_utc=naive_now,
        )


def test_run_once_returns_zero_when_nothing_due(store: RunStore, tmp_path: Path) -> None:
    now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    n = run_once(
        store,
        fetch_price_fn=lambda asset, venue: 100.0,
        now_utc=now,
    )
    assert n == 0


def test_hit_tolerance_is_a_finite_positive_constant() -> None:
    assert 0.0 < HIT_TOLERANCE_PCT < 1.0
