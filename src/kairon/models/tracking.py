"""MLflow tracker adapter.

This module is the only place in the codebase that imports ``mlflow``.
If mlflow is not installed, :class:`MlflowTracker` falls back to a
no-op implementation so the trainer can still run (Phase 4 ships the
contract; Phase 11 wires the actual UI on top of these runs).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from kairon.models.trainer import Tracker


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Where to log runs."""

    tracking_uri: str | None = None  # e.g. "file:./mlruns" or "http://localhost:5000"
    experiment_name: str = "kairon"
    run_tags: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _has_mlflow() -> bool:
    return importlib.util.find_spec("mlflow") is not None  # type: ignore[attr-defined]


class MlflowTracker(Tracker):
    """Thin wrapper over the mlflow tracking client.

    If mlflow is not installed, all methods are no-ops. This is
    intentional: trainer/evaluator code should run in tests without
    mlflow present, and the model layer must not take a hard dependency
    on the experiment-tracking package.
    """

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._mlflow: Any = None
        self._active = False
        if _has_mlflow():
            self._mlflow = importlib.import_module("mlflow")
            if self.config.tracking_uri:
                self._mlflow.set_tracking_uri(self.config.tracking_uri)
            self._mlflow.set_experiment(self.config.experiment_name)

    def start_run(self, *, run_name: str, params: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        self._mlflow.start_run(run_name=run_name, tags=self.config.run_tags)
        self._mlflow.log_params(params)
        self._active = True

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._mlflow is None or not self._active:
            return
        clean = {k: float(v) for k, v in metrics.items() if _is_finite(v)}
        if clean:
            self._mlflow.log_metrics(clean)

    def end_run(self) -> None:
        if self._mlflow is None or not self._active:
            return
        self._mlflow.end_run()
        self._active = False


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


__all__ = ["MlflowTracker", "TrackingConfig"]
