"""Run store: persist :class:`kairon.analysis.contracts.RunResult` to sqlite.

Why a new store: the existing :class:`kairon.store.ModelStore` is a
filesystem store for :class:`TrainedModel` artifacts. Web app runs are a
different domain — small, queryable, transactional — so they get their
own sqlite-backed store. The two coexist; ``ModelStore`` is untouched.

Implementation notes:
- stdlib ``sqlite3`` only (zero new dep)
- One writer at a time, enforced by a ``threading.Lock``
- ``check_same_thread=False`` so the FastAPI verifier thread can read
- Schema migrates on first open (``CREATE TABLE IF NOT EXISTS``)
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

from kairon.analysis.contracts import (
    HorizonName,
    ModelName,
    ModelTile,
    ProvenanceBlock,
    RunResult,
    VerificationStatus,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    asset           TEXT NOT NULL,
    horizon         TEXT NOT NULL,
    created_at_utc  TEXT NOT NULL,
    csv_path        TEXT NOT NULL,
    run_result_json TEXT NOT NULL,
    actual_pct      REAL,
    delta_pct       REAL,
    status          TEXT NOT NULL DEFAULT 'pending',
    verified_at_utc TEXT,
    pinned          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at_utc);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


def _parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a UTC tz-aware datetime.

    Accepts any tz-aware ISO timestamp (UTC, +00:00, +05:30, etc.) and
    returns it unchanged (preserves the original offset, since UTC-aware
    comparison works without normalization).
    """
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {s!r} is not tz-aware; expected UTC")
    return dt


class RunStore:
    """SQLite-backed store of :class:`RunResult` rows.

    A single :class:`RunStore` instance wraps a single sqlite file. Use one
    instance per process; share via a global or DI as appropriate.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.executescript(_SCHEMA)

    # --- writes ---------------------------------------------------------

    def create(self, run: RunResult, csv_path: Path) -> None:
        """Persist a new run. Overwrites if ``run_id`` already exists."""
        if run.created_at_utc.tzinfo is None:
            raise ValueError("RunResult.created_at_utc must be tz-aware UTC")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs ("
                "run_id, asset, horizon, created_at_utc, csv_path, run_result_json, "
                "actual_pct, delta_pct, status, verified_at_utc, pinned"
                ") VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 'pending', NULL, 0)",
                (
                    run.run_id,
                    run.asset,
                    run.horizon,
                    run.created_at_utc.isoformat(),
                    str(csv_path),
                    run.model_dump_json(),
                ),
            )

    def update_verification(
        self,
        run_id: str,
        actual_pct: float,
        delta_pct: float,
        verified_at_utc: datetime,
        status: VerificationStatus,
    ) -> None:
        """Write back the verification result for a run."""
        if verified_at_utc.tzinfo is None:
            raise ValueError("verified_at_utc must be tz-aware UTC")
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET actual_pct = ?, delta_pct = ?, status = ?, "
                "verified_at_utc = ? WHERE run_id = ?",
                (actual_pct, delta_pct, status, verified_at_utc.isoformat(), run_id),
            )

    def set_pinned(self, run_id: str, pinned: bool) -> None:
        """Pin/unpin a run. Idempotent."""
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET pinned = ? WHERE run_id = ?",
                (1 if pinned else 0, run_id),
            )

    # --- reads ----------------------------------------------------------

    def get(self, run_id: str) -> RunResult | None:
        """Return the :class:`RunResult` for ``run_id``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT run_result_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return RunResult.model_validate_json(row[0])

    def get_csv_path(self, run_id: str) -> Path | None:
        """Return the server-side CSV path for a run, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT csv_path FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return Path(row[0])

    def list_runs(self) -> tuple[RunResult, ...]:
        """Return all runs, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_result_json FROM runs ORDER BY created_at_utc DESC"
            ).fetchall()
        return tuple(RunResult.model_validate_json(r[0]) for r in rows)

    def mark_due(self, now_utc: datetime) -> tuple[str, ...]:
        """Return run_ids whose verification is due as of ``now_utc``.

        A run is due when ``now_utc - created_at_utc >= horizon_duration``
        and ``status == 'pending'``. Horizon durations are hardcoded here
        to match ``HORIZON_PROFILES`` in ``kairon.analysis.engine``.
        """
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be tz-aware UTC")
        durations: dict[HorizonName, timedelta] = {
            "day": timedelta(hours=24),
            "swing": timedelta(days=5),
            "long": timedelta(days=30),
        }
        runs = self.list_runs()
        due: list[str] = []
        for r in runs:
            if r.created_at_utc.tzinfo is None:
                continue
            if now_utc - r.created_at_utc >= durations[r.horizon]:
                due.append(r.run_id)
        return tuple(due)

    def get_verification(
        self, run_id: str
    ) -> tuple[float, float, VerificationStatus, datetime | None] | None:
        """Return ``(actual_pct, delta_pct, status, verified_at_utc)`` for a run, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT actual_pct, delta_pct, status, verified_at_utc "
                "FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        actual_pct = float(row[0])
        delta_pct = float(row[1])
        status = row[2]
        if status not in ("pending", "hit", "missed"):
            raise ValueError(f"unknown verification status {status!r}")
        verified_at = _parse_iso_utc(row[3]) if row[3] is not None else None
        return actual_pct, delta_pct, status, verified_at

    def close(self) -> None:
        """Close the underlying sqlite connection. Idempotent."""
        with self._lock:
            self._conn.close()

    # --- helpers --------------------------------------------------------

    @staticmethod
    def iter_models(run: RunResult) -> Iterable[tuple[ModelName, ModelTile]]:
        """Iterate (name, tile) pairs over a run's models."""
        return ((m.name, m) for m in run.models)

    @staticmethod
    def make_provenance(
        *, config_hash: str, data_hash: str, model_version: str, seed: int
    ) -> ProvenanceBlock:
        """Convenience constructor for tests + callers."""
        return ProvenanceBlock(
            config_hash=config_hash,
            data_hash=data_hash,
            model_version=model_version,
            seed=seed,
        )


__all__ = ["RunStore"]
