"""Training orchestrator: fit + persist + track one (or several) models.

The :class:`Trainer` is the only place in the codebase that calls
``model.fit``. It is responsible for:

- iterating walk-forward folds,
- running the model on each fold,
- collecting per-fold metrics (accuracy, log-loss, Brier),
- persisting the ``TrainedModel`` artifacts (metadata only at this stage),
- emitting an mlflow run with the right params and metrics.

It is intentionally backend-agnostic: any :class:`Model` works.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kairon.models.base import (
    LossFnName,
    Model,
    ModelError,
    TrainedModel,
    _validate_loss_fn,
)
from kairon.models.contracts import FeatureMatrix
from kairon.splits.walkforward import Fold


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    """Per-fold evaluation summary."""

    fold_id: int
    train_rows: int
    val_rows: int
    test_rows: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_uri: str | None = None


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Aggregate result of training a single model across folds."""

    backend: str
    folds: tuple[FoldMetrics, ...]
    train_seconds: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_test_acc(self) -> float:
        vals = [f.metrics.get("test_acc", float("nan")) for f in self.folds]
        return float(np.nanmean(vals)) if vals else float("nan")

    @property
    def mean_test_logloss(self) -> float:
        vals = [f.metrics.get("test_logloss", float("nan")) for f in self.folds]
        return float(np.nanmean(vals)) if vals else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_folds": len(self.folds),
            "mean_test_acc": self.mean_test_acc,
            "mean_test_logloss": self.mean_test_logloss,
            "train_seconds": self.train_seconds,
        }


class Trainer:
    """Walk-forward trainer.

    Parameters
    ----------
    model
        The :class:`Model` to fit.
    artifact_root
        Directory where per-fold ``TrainedModel`` artifacts are persisted.
        Each fold is written to ``{artifact_root}/{backend}/fold_{i:03d}/``.
    tracker
        Optional :class:`Tracker` to log params/metrics to. If ``None``,
        a no-op tracker is used.
    """

    def __init__(
        self,
        model: Model[Any],
        *,
        artifact_root: Path | str | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        self.model = model
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.tracker = tracker or NoOpTracker()

    def fit_walkforward(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        folds: list[Fold],
        *,
        run_name: str | None = None,
        loss_fn: LossFnName = "cross_entropy",
    ) -> TrainResult:
        if not folds:
            raise ModelError("fit_walkforward needs at least one fold")
        if features.n_rows != y.shape[0]:
            raise ModelError(
                f"features has {features.n_rows} rows, y has {y.shape[0]}"
            )
        # Defence-in-depth: the Literal type is the static contract,
        # but a non-type-checked caller (CLI / YAML / mlflow) could
        # still pass an unknown name. _validate_loss_fn raises
        # ValueError immediately, before any model.fit is called.
        loss_fn = _validate_loss_fn(loss_fn)

        run_name = run_name or f"{self.model.name}-{int(time.time())}"
        # Report the loss_fn choice in the tracker params; this is
        # the load-bearing report channel the W5.1 spec requires.
        self.tracker.start_run(
            run_name=run_name,
            params={"backend": self.model.name, "loss_fn": loss_fn},
        )
        t0 = time.perf_counter()
        per_fold: list[FoldMetrics] = []
        for f in folds:
            train_slice = features.values[f.train_start : f.train_end]
            test_slice = features.values[f.test_start : f.test_end]
            train_y = y[f.train_start : f.train_end]
            test_y = y[f.test_start : f.test_end]
            if train_slice.shape[0] < 2 or test_slice.shape[0] < 1:
                continue
            train_fm = FeatureMatrix(
                values=train_slice,
                feature_names=features.feature_names,
                ts=features.ts[f.train_start : f.train_end] if features.ts is not None else None,
            )
            test_fm = FeatureMatrix(
                values=test_slice,
                feature_names=features.feature_names,
                ts=features.ts[f.test_start : f.test_end] if features.ts is not None else None,
            )
            trained = self.model.fit(train_fm, train_y, loss_fn=loss_fn)
            pred = self.model.predict(trained, test_fm)
            metrics = _classification_metrics(pred.y_class, pred.y_proba, test_y)
            artifact_uri = self._persist(trained, f.fold_id)
            self.tracker.log_metrics(
                {f"fold{f.fold_id}_" + k: v for k, v in metrics.items()}
            )
            per_fold.append(
                FoldMetrics(
                    fold_id=f.fold_id,
                    train_rows=int(train_slice.shape[0]),
                    val_rows=f.n_val(),
                    test_rows=int(test_slice.shape[0]),
                    metrics=metrics,
                    artifact_uri=artifact_uri,
                )
            )
        elapsed = time.perf_counter() - t0
        agg = TrainResult(
            backend=self.model.name,
            folds=tuple(per_fold),
            train_seconds=elapsed,
            extras={"loss_fn": loss_fn},
        )
        self.tracker.log_metrics(
            {
                "mean_test_acc": agg.mean_test_acc,
                "mean_test_logloss": agg.mean_test_logloss,
                "train_seconds": elapsed,
            }
        )
        self.tracker.end_run()
        return agg

    def _persist(self, trained: TrainedModel, fold_id: int) -> str | None:
        if self.artifact_root is None:
            return None
        fold_dir = self.artifact_root / trained.backend / f"fold_{fold_id:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        import json

        meta = {
            "backend": trained.backend,
            "feature_names": list(trained.feature_names),
            "classes": list(trained.classes) if trained.classes is not None else None,
            "target_kind": trained.target_kind,
            "created_at_ns": trained.created_at_ns,
            "n_train_rows": trained.spec.n_train_rows,
            "n_features": trained.spec.n_features,
            "metrics": trained.spec.metrics,
            "train_seconds": trained.spec.train_seconds,
        }
        (fold_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        return str(fold_dir.resolve())


def _classification_metrics(
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    y_true: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if y_true.size == 0:
        return out
    out["test_acc"] = float((y_pred == y_true).mean())
    if y_proba is not None:
        eps = 1e-15
        if y_proba.ndim == 1:
            p = np.clip(y_proba, eps, 1.0 - eps)
            ll = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        else:
            n_classes = y_proba.shape[1]
            oh = np.eye(n_classes, dtype=np.float64)[y_true.astype(np.int64)]
            p = np.clip(y_proba, eps, 1.0 - eps)
            ll = -(oh * np.log(p)).sum(axis=1)
        out["test_logloss"] = float(ll.mean())
        if y_proba.ndim == 1:
            out["test_brier"] = float(np.mean((y_proba - y_true) ** 2))
        else:
            # oh is always defined here because y_proba.ndim == 2
            out["test_brier"] = float(np.mean((y_proba - oh) ** 2))  # type: ignore[possibly-unbound]
    return out


# ---------------------------------------------------------------------------
# Tracker interface
# ---------------------------------------------------------------------------
class Tracker:
    """Minimal mlflow-shaped tracker.

    A real implementation forwards to ``mlflow.log_*``. We keep the
    interface tiny and provide a no-op default so the trainer can run
    in unit tests without mlflow.
    """

    def start_run(self, *, run_name: str, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def end_run(self) -> None: ...


class NoOpTracker(Tracker):
    def start_run(self, *, run_name: str, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float]) -> None:
        return None

    def end_run(self) -> None:
        return None


__all__ = [
    "FoldMetrics",
    "NoOpTracker",
    "Tracker",
    "TrainResult",
    "Trainer",
]
