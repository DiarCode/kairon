"""Tests for the :class:`kairon.store.runs.RunStore` sqlite-backed run store."""
from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kairon.analysis.contracts import (
    ModelTile,
    ProvenanceBlock,
    RunResult,
)
from kairon.store.runs import RunStore


def _run(
    *,
    run_id: str = "run-001",
    asset: str = "BTC",
    horizon: str = "day",
    created_at: datetime | None = None,
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
                predicted_pct=0.01,
                stop_loss=99.0,
                ideal_entry=100.0,
                ideal_exit=101.0,
                confidence=0.6,
            ),
        ),
        provenance=ProvenanceBlock(
            config_hash="cfg",
            data_hash="dat",
            model_version="kairon-0.1.0",
            seed=42,
        ),
        base_price=100.0,
    )


def _store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs.db")


def test_create_and_get_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        r = _run()
        store.create(r, csv_path=tmp_path / "input.csv")
        restored = store.get(r.run_id)
        assert restored == r
    finally:
        store.close()


def test_list_runs_newest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        a = _run(run_id="a", created_at=datetime(2026, 6, 9, 10, 0, 0, tzinfo=UTC))
        b = _run(run_id="b", created_at=datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC))
        store.create(a, tmp_path / "a.csv")
        store.create(b, tmp_path / "b.csv")
        ids = [r.run_id for r in store.list_runs()]
        assert ids == ["b", "a"]
    finally:
        store.close()


def test_update_verification_writes_back(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        r = _run()
        store.create(r, tmp_path / "input.csv")
        now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
        store.update_verification(r.run_id, actual_pct=0.02, delta_pct=0.01, verified_at_utc=now, status="hit")
        v = store.get_verification(r.run_id)
        assert v is not None
        assert v[0] == 0.02
        assert v[1] == 0.01
        assert v[2] == "hit"
        assert v[3] == now
    finally:
        store.close()


def test_mark_due_respects_horizon_windows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
        # Day run, created 25h ago: due
        day_old = _run(run_id="day-old", horizon="day", created_at=now - timedelta(hours=25))
        # Day run, created 23h ago: NOT due
        day_fresh = _run(run_id="day-fresh", horizon="day", created_at=now - timedelta(hours=23))
        # Swing run, created 4d ago: NOT due (swing is 5d)
        swing_fresh = _run(run_id="swing-fresh", horizon="swing", created_at=now - timedelta(days=4))
        # Long run, created 31d ago: due
        long_due = _run(run_id="long-due", horizon="long", created_at=now - timedelta(days=31))
        for r in (day_old, day_fresh, swing_fresh, long_due):
            store.create(r, tmp_path / f"{r.run_id}.csv")
        due = store.mark_due(now)
        assert set(due) == {"day-old", "long-due"}
    finally:
        store.close()


def test_set_pinned_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        r = _run()
        store.create(r, tmp_path / "input.csv")
        store.set_pinned(r.run_id, True)
        store.set_pinned(r.run_id, True)  # idempotent
        store.set_pinned(r.run_id, False)
        # No exception, no row corruption.
        assert store.get(r.run_id) is not None
    finally:
        store.close()


def test_get_missing_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        assert store.get("does-not-exist") is None
        assert store.get_verification("does-not-exist") is None
    finally:
        store.close()


def test_concurrent_inserts_are_serialized(tmp_path: Path) -> None:
    """10 threads x 100 inserts each, all unique run_ids, all must persist."""
    store = _store(tmp_path)
    try:
        def worker(tid: int) -> None:
            for j in range(100):
                rid = f"t{tid}-{j}"
                store.create(_run(run_id=rid), tmp_path / f"{rid}.csv")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        listed = store.list_runs()
        assert len(listed) == 1000
    finally:
        store.close()
