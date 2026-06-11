"""Model store: persist and load :class:`TrainedModel` artifacts.

A *trained model* lives in two pieces:

- ``meta.json`` — name, feature order, classes, target kind, created_at,
  train metrics, etc. Small, JSON-serializable, fast to inspect.
- ``state.pkl`` (or ``state.pt`` for torch backends) — the backend's
  fitted artifact, written via joblib/pickle or torch.save.

The store keys artifacts by ``run_name`` (a user-supplied string) and
``fold_id`` (0 for the all-data run, 1..N for walk-forward folds).
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kairon.models.base import TrainedModel


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Pointer to a persisted :class:`TrainedModel` artifact."""

    run_name: str
    backend: str
    root: Path
    meta_path: Path
    state_path: Path
    created_at: datetime
    extras: dict[str, Any] = ...  # type: ignore[assignment]


class ModelStore:
    """Filesystem-backed model store.

    Layout::

        {root}/
          {run_name}/
            meta.json
            state.pkl

    The store is intentionally simple: no concurrency, no checksums.
    A production deployment would back this with S3 or a similar
    object store, but the *interface* (save/load/list) stays the same.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_name: str) -> Path:
        d = self.root / run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, trained: TrainedModel, run_name: str) -> StoredArtifact:
        d = self._run_dir(run_name)
        meta = {
            "backend": trained.backend,
            "feature_names": list(trained.feature_names),
            "classes": list(trained.classes) if trained.classes is not None else None,
            "target_kind": trained.target_kind,
            "created_at_ns": trained.created_at_ns,
            "created_at_iso": datetime.fromtimestamp(
                trained.created_at_ns / 1e9, tz=UTC
            ).isoformat() if trained.created_at_ns else None,
            "n_train_rows": trained.spec.n_train_rows,
            "n_features": trained.spec.n_features,
            "metrics": trained.spec.metrics,
            "train_seconds": trained.spec.train_seconds,
        }
        meta_path = d / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        state_path = d / "state.pkl"
        with state_path.open("wb") as f:
            pickle.dump(trained.state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return StoredArtifact(
            run_name=run_name,
            backend=trained.backend,
            root=self.root,
            meta_path=meta_path,
            state_path=state_path,
            created_at=datetime.now(UTC),
            extras={},
        )

    def load(self, run_name: str, *, backend: str | None = None) -> TrainedModel:
        d = self.root / run_name
        if not d.exists():
            raise FileNotFoundError(f"no run {run_name!r} in {self.root}")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if backend is not None and meta["backend"] != backend:
            raise ValueError(
                f"run {run_name!r} is backend {meta['backend']!r}, not {backend!r}"
            )
        state = pickle.loads((d / "state.pkl").read_bytes())  # noqa: S301
        from kairon.models.base import TrainResult

        return TrainedModel(
            backend=meta["backend"],
            spec=TrainResult(
                backend=meta["backend"],
                n_train_rows=int(meta["n_train_rows"]),
                n_features=int(meta["n_features"]),
                feature_names=tuple(meta["feature_names"]),
                train_seconds=float(meta.get("train_seconds", 0.0)),
                metrics=dict(meta.get("metrics", {})),
            ),
            state=state,
            feature_names=tuple(meta["feature_names"]),
            target_kind=meta["target_kind"],
            classes=tuple(meta["classes"]) if meta.get("classes") is not None else None,
            created_at_ns=int(meta.get("created_at_ns", 0)),
        )

    def exists(self, run_name: str) -> bool:
        return (self.root / run_name / "meta.json").exists()

    def list_runs(self) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            sorted(p.name for p in self.root.iterdir() if (p / "meta.json").exists())
        )

    def delete(self, run_name: str) -> bool:
        d = self.root / run_name
        if not d.exists():
            return False
        shutil.rmtree(d)
        return True


__all__ = ["ModelStore", "StoredArtifact"]
