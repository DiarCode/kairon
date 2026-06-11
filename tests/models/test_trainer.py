"""Tests for the Trainer + tracker integration."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.models.contracts import FeatureMatrix
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.trainer import (
    FoldMetrics,
    NoOpTracker,
    Tracker,
    Trainer,
    TrainResult,
)
from kairon.splits.walkforward import SplitSpec, walkforward


class _CollectTracker(Tracker):
    def __init__(self) -> None:
        self.runs: list[str] = []
        self.metrics: list[dict[str, float]] = []
        self.starts = 0
        self.ends = 0

    def start_run(self, *, run_name: str, params: dict) -> None:
        self.runs.append(run_name)
        self.starts += 1

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.append(metrics)

    def end_run(self) -> None:
        self.ends += 1


def _toy(n: int = 600) -> tuple[FeatureMatrix, np.ndarray]:
    rng = np.random.default_rng(5)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 + 0.2 * rng.normal(size=n) > 0).astype(np.int64)
    fm = FeatureMatrix(
        values=np.column_stack([x1, x2]).astype(np.float64),
        feature_names=("x1", "x2"),
        ts=np.array([np.datetime64(0, "ns") + i for i in range(n)], dtype="datetime64[ns]"),
    )
    return fm, y


def test_trainer_walkforward_no_folds_raises() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    t = Trainer(m)
    with pytest.raises(Exception):
        t.fit_walkforward(fm, y, [])


def test_trainer_walkforward_row_mismatch_raises() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel()
    t = Trainer(m)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    with pytest.raises(Exception):
        t.fit_walkforward(fm, y[:100], folds)


def test_trainer_walkforward_smoke() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    trk = _CollectTracker()
    t = Trainer(m, tracker=trk)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    result = t.fit_walkforward(fm, y, folds, run_name="smoke")
    assert isinstance(result, TrainResult)
    assert result.backend == "logreg"
    assert result.folds
    assert 0.0 <= result.mean_test_acc <= 1.0
    assert trk.starts == 1
    assert trk.ends == 1
    assert any("mean_test_acc" in m for m in trk.metrics)


def test_trainer_persists_artifacts(tmp_path) -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    t = Trainer(m, artifact_root=tmp_path)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    result = t.fit_walkforward(fm, y, folds)
    # At least one fold has an artifact_uri
    uris = [f.artifact_uri for f in result.folds if f.artifact_uri]
    assert uris
    # meta.json exists in the first fold dir
    import json
    from pathlib import Path

    p = Path(uris[0]) / "meta.json"
    assert p.exists()
    meta = json.loads(p.read_text(encoding="utf-8"))
    assert meta["backend"] == "logreg"
    assert meta["feature_names"] == ["x1", "x2"]


def test_trainer_no_artifact_root() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    t = Trainer(m)  # no artifact_root
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    result = t.fit_walkforward(fm, y, folds)
    for f in result.folds:
        assert f.artifact_uri is None


def test_noop_tracker_methods_return_none() -> None:
    t = NoOpTracker()
    assert t.start_run(run_name="x", params={}) is None
    assert t.log_metrics({"a": 1.0}) is None
    assert t.end_run() is None


def test_train_result_to_dict() -> None:
    fm, y = _toy()
    m = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    t = Trainer(m)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    result = t.fit_walkforward(fm, y, folds)
    d = result.to_dict()
    assert d["backend"] == "logreg"
    assert d["n_folds"] == len(result.folds)
    assert d["mean_test_acc"] == result.mean_test_acc


def test_trainer_skips_short_folds() -> None:
    """If a fold has only 1 train row, the trainer should skip it gracefully."""
    fm = FeatureMatrix(
        values=np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [1.5, 1.5]]),
        feature_names=("a", "b"),
    )
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    m = LogisticRegressionModel(LinearConfig(C=1.0, random_state=0))
    t = Trainer(m)
    # Manually build a degenerate fold with 1 train row
    from kairon.splits.walkforward import Fold
    folds = [Fold(0, 0, 1, 1, 1, 1, 2)]
    result = t.fit_walkforward(fm, y, folds)
    # Skipped, so no folds recorded
    assert result.folds == ()


def test_fold_metrics_dataclass() -> None:
    fm = FoldMetrics(fold_id=0, train_rows=10, val_rows=2, test_rows=5, metrics={"acc": 0.6})
    assert fm.metrics["acc"] == 0.6


# ---------------------------------------------------------------------------
# W5.1 — Trainer.loss_fn parameter
# ---------------------------------------------------------------------------
class _SpyModel(LogisticRegressionModel):
    """A logistic regression that records every fit() call's kwargs.

    Used by the W5.1 tests to pin that loss_fn is threaded from
    Trainer.fit_walkforward into model.fit. The spy overrides the
    public ``fit`` method (not ``_fit_core``) so the existing v1 fit
    shim runs to completion and the trainer's downstream ``predict``
    call still works.
    """

    def __init__(self) -> None:
        super().__init__(LinearConfig(C=1.0, random_state=0))
        self.fit_calls: list[dict] = []

    def fit(self, features, y, *, sample_weight=None, loss_fn="cross_entropy"):  # type: ignore[override]
        self.fit_calls.append({"loss_fn": loss_fn, "sample_weight": sample_weight})
        return super().fit(
            features, y, sample_weight=sample_weight, loss_fn=loss_fn
        )


def test_loss_fn_required() -> None:
    """Default loss_fn is cross_entropy (per Critic round 1 minor #5).

    Calling fit_walkforward WITHOUT an explicit loss_fn is the
    backward-compat behaviour pinned by the spec; the resolved
    loss_fn is reported in the TrainResult.extras dict.
    """
    fm, y = _toy()
    m = _SpyModel()
    t = Trainer(m)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    result = t.fit_walkforward(fm, y, folds, run_name="loss_default")
    assert result.extras["loss_fn"] == "cross_entropy"
    # And it threaded to the model.fit call (default is cross_entropy).
    assert all(call["loss_fn"] == "cross_entropy" for call in m.fit_calls)


def test_loss_fn_rejects_unknown() -> None:
    """An unknown loss_fn name raises ValueError immediately.

    The Literal type is the static contract; the runtime check in
    :func:`kairon.models.base._validate_loss_fn` is the defence for
    non-type-checked callers. The validation happens BEFORE any
    model.fit is called, so a bad loss_fn never reaches the model.
    """
    fm, y = _toy()
    m = _SpyModel()
    t = Trainer(m)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    with pytest.raises(ValueError, match="unknown loss_fn"):
        # The Literal type rejects "invalid_name" statically; the
        # `# type: ignore[arg-type]` confirms the runtime guard is
        # the load-bearing rejection path the test exercises.
        t.fit_walkforward(
            fm,
            y,
            folds,
            run_name="loss_bad",
            loss_fn="invalid_name",  # type: ignore[arg-type]
        )
    # And no model.fit was called — validation fails fast.
    assert m.fit_calls == []


def test_loss_fn_threads_to_model_fit() -> None:
    """The loss_fn kwarg is forwarded verbatim to every model.fit call.

    The W5.2 / W5.3 release wires ``"sharpe"`` and ``"cost_focal"`` to
    their torch implementations; for W5.1 the contract is that the
    string is threaded through and the v1 backends treat it as
    advisory metadata (they keep fitting cross-entropy). The spy
    records the kwargs it received and we pin the value here.
    """
    fm, y = _toy()
    m = _SpyModel()
    t = Trainer(m)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    result = t.fit_walkforward(
        fm, y, folds, run_name="loss_thread", loss_fn="cost_focal"
    )
    # Every fold's model.fit saw loss_fn="cost_focal".
    assert m.fit_calls, "model.fit was never called"
    assert all(call["loss_fn"] == "cost_focal" for call in m.fit_calls)
    # And the TrainResult reports the same value.
    assert result.extras["loss_fn"] == "cost_focal"


class _ParamsTracker(Tracker):
    """Tracker double that records the params dict passed to start_run.

    Used to verify that the loss_fn choice is reported in the mlflow
    run's params (the PRD W5.1 spec, item 3 of the deliverable list).
    """

    def __init__(self) -> None:
        self.runs: list[str] = []
        self.params: list[dict] = []
        self.starts = 0
        self.ends = 0
        self.metric_calls = 0

    def start_run(self, *, run_name: str, params: dict) -> None:
        self.runs.append(run_name)
        self.params.append(dict(params))
        self.starts += 1

    def log_metrics(self, metrics: dict) -> None:
        self.metric_calls += 1

    def end_run(self) -> None:
        self.ends += 1


def test_loss_fn_in_tracker() -> None:
    """The loss_fn is logged as an mlflow run param at start_run time.

    The tracker is the load-bearing report channel the W5.1 spec
    requires. After a fit_walkforward with loss_fn='cost_focal', the
    params dict MUST include loss_fn='cost_focal' so an operator can
    pivot dashboards on it.
    """
    fm, y = _toy()
    m = _SpyModel()
    trk = _ParamsTracker()
    t = Trainer(m, tracker=trk)
    folds = walkforward(600, spec=SplitSpec(train_size=200, val_size=0, test_size=100))
    t.fit_walkforward(fm, y, folds, run_name="loss_param", loss_fn="cost_focal")
    assert trk.starts == 1
    assert trk.params[0]["loss_fn"] == "cost_focal"
    assert trk.params[0]["backend"] == "logreg"
    # And the run completed cleanly.
    assert trk.ends == 1
