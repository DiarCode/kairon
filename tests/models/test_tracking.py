"""Tests for the mlflow tracker adapter (no-ops if mlflow isn't installed)."""

from __future__ import annotations

import pytest

from kairon.models.tracking import MlflowTracker, TrackingConfig, _has_mlflow


def test_tracker_no_mlflow_is_noop() -> None:
    """If mlflow is absent, all methods must be no-ops (no exceptions)."""
    if _has_mlflow():
        pytest.skip("mlflow installed; can only test no-op path when absent")
    t = MlflowTracker(TrackingConfig())
    t.start_run(run_name="x", params={"a": 1.0})  # no error
    t.log_metrics({"acc": 0.5})  # no error
    t.end_run()  # no error


def test_tracker_with_mlflow_does_not_throw() -> None:
    """If mlflow is installed, the tracker should work end-to-end with a temp URI."""
    if not _has_mlflow():
        pytest.skip("mlflow not installed")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        t = MlflowTracker(
            TrackingConfig(tracking_uri=f"file:{tmp}/mlruns", experiment_name="kairon-test")
        )
        t.start_run(run_name="smoke", params={"backend": "logreg"})
        t.log_metrics({"acc": 0.5, "loss": float("inf")})  # inf should be filtered
        t.end_run()


def test_is_finite_helper() -> None:
    from kairon.models.tracking import _is_finite

    assert _is_finite(0.5) is True
    assert _is_finite(0.0) is True
    assert _is_finite(float("inf")) is False
    assert _is_finite(float("-inf")) is False
    assert _is_finite(float("nan")) is False
