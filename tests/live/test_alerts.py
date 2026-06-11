"""Tests for the live (drift + alerting + predictor) module."""

from __future__ import annotations

import numpy as np
import pytest

from kairon.live.alerts import (
    Alert,
    AlertEngine,
    Channel,
    DriftSeverityRule,
    InMemoryChannel,
    LoggingChannel,
    Severity,
    ThresholdRule,
)
from kairon.live.drift import (
    DriftScore,
    check_drift,
    check_drift_table,
    ks_two_sample,
    population_stability_index,
    severity_for_ks,
    severity_for_psi,
)


def _samples(seed: int = 0, n: int = 2000, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, std, n)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------
def test_psi_identical_distributions_is_near_zero() -> None:
    ref = _samples(seed=0, n=2000)
    live = _samples(seed=1, n=2000)  # same distribution, different sample
    score, ref_pct, live_pct = population_stability_index(ref, live)
    assert score < 0.1
    assert ref_pct.shape == live_pct.shape


def test_psi_shifted_distribution_is_high() -> None:
    ref = _samples(seed=0, n=2000, mean=0.0)
    live = _samples(seed=1, n=2000, mean=3.0)  # big mean shift
    score, _, _ = population_stability_index(ref, live)
    assert score > 0.2


def test_psi_rejects_2d() -> None:
    with pytest.raises(ValueError):
        population_stability_index(np.zeros((2, 2)), np.zeros((2, 2)))


def test_psi_rejects_too_small() -> None:
    with pytest.raises(ValueError):
        population_stability_index(np.array([1.0]), np.array([1.0]))


def test_severity_for_psi_bands() -> None:
    assert severity_for_psi(0.05) == "ok"
    assert severity_for_psi(0.15) == "warning"
    assert severity_for_psi(0.5) == "critical"


# ---------------------------------------------------------------------------
# KS
# ---------------------------------------------------------------------------
def test_ks_identical_distribution() -> None:
    ref = _samples(seed=0, n=300)
    live = _samples(seed=1, n=300)
    stat, p = ks_two_sample(ref, live)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= p <= 1.0


def test_ks_shifted_distribution() -> None:
    ref = _samples(seed=0, n=300, mean=0.0)
    live = _samples(seed=1, n=300, mean=5.0)
    stat, _ = ks_two_sample(ref, live)
    assert stat > 0.5


def test_severity_for_ks_bands() -> None:
    assert severity_for_ks(0.02) == "ok"
    assert severity_for_ks(0.07) == "warning"
    assert severity_for_ks(0.5) == "critical"


# ---------------------------------------------------------------------------
# check_drift / check_drift_table
# ---------------------------------------------------------------------------
def test_check_drift_psi() -> None:
    score = check_drift("x", _samples(seed=0), _samples(seed=1), method="psi")
    assert isinstance(score, DriftScore)
    assert score.method == "psi"
    assert score.severity == "ok"
    assert score.p_value is None


def test_check_drift_ks() -> None:
    score = check_drift("x", _samples(seed=0), _samples(seed=1), method="ks")
    assert score.method == "ks"
    assert score.p_value is not None


def test_check_drift_unknown_method() -> None:
    with pytest.raises(ValueError):
        check_drift("x", _samples(seed=0), _samples(seed=1), method="nope")


def test_check_drift_table() -> None:
    ref = np.column_stack([_samples(seed=0, n=300), _samples(seed=1, n=300)])
    live = np.column_stack([_samples(seed=2, n=300), _samples(seed=3, n=300)])
    scores = check_drift_table(ref, live, ("a", "b"), method="psi")
    assert len(scores) == 2
    assert scores[0].feature == "a"
    assert scores[1].feature == "b"


def test_check_drift_table_shape_mismatch() -> None:
    ref = np.zeros((10, 3))
    live = np.zeros((10, 4))
    with pytest.raises(ValueError):
        check_drift_table(ref, live, ("a", "b", "c", "d"))


def test_check_drift_table_feature_count_mismatch() -> None:
    ref = np.zeros((10, 3))
    live = np.zeros((10, 3))
    with pytest.raises(ValueError):
        check_drift_table(ref, live, ("a", "b"))


# ---------------------------------------------------------------------------
# Alert rules + engine
# ---------------------------------------------------------------------------
def test_drift_severity_rule_matches_critical() -> None:
    rule = DriftSeverityRule()
    score = DriftScore(
        feature="f1", method="psi", score=0.5, p_value=None, severity="critical",
        n_ref=100, n_live=100, extras={},
    )
    a = rule.matches(score)
    assert a is not None
    assert a.severity == Severity.CRITICAL
    assert "f1" in a.message


def test_drift_severity_rule_ignores_ok() -> None:
    rule = DriftSeverityRule()
    score = DriftScore(
        feature="f1", method="psi", score=0.05, p_value=None, severity="ok",
        n_ref=100, n_live=100, extras={},
    )
    assert rule.matches(score) is None


def test_drift_severity_rule_ignores_unknown_features() -> None:
    rule = DriftSeverityRule(features=("a",))
    score = DriftScore(
        feature="b", method="psi", score=0.5, p_value=None, severity="critical",
        n_ref=100, n_live=100, extras={},
    )
    assert rule.matches(score) is None


def test_drift_severity_rule_ignores_non_drift_facts() -> None:
    rule = DriftSeverityRule()
    assert rule.matches(("foo", 1.0)) is None
    assert rule.matches("hello") is None


def test_threshold_rule_above() -> None:
    rule = ThresholdRule(name="t1", source="foo", threshold=0.5, direction="above")
    assert rule.matches(("foo", 0.6)) is not None
    assert rule.matches(("foo", 0.4)) is None
    assert rule.matches(("bar", 0.6)) is None


def test_threshold_rule_below() -> None:
    rule = ThresholdRule(name="t1", source="foo", threshold=0.5, direction="below")
    a = rule.matches(("foo", 0.1))
    assert a is not None
    assert rule.matches(("foo", 0.6)) is None


def test_threshold_rule_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        ThresholdRule(name="t", source="x", threshold=0.5, direction="sideways")


def test_alert_engine_collects_alerts() -> None:
    ch = InMemoryChannel()
    eng = AlertEngine(
        rules=[DriftSeverityRule()],
        channels=[ch],
    )
    score = DriftScore(
        feature="f", method="psi", score=0.5, p_value=None, severity="critical",
        n_ref=100, n_live=100, extras={},
    )
    alerts = eng.evaluate(score)
    assert len(alerts) == 1
    assert ch.alerts == alerts


def test_alert_engine_skips_non_matching_facts() -> None:
    ch = InMemoryChannel()
    eng = AlertEngine(rules=[DriftSeverityRule()], channels=[ch])
    eng.evaluate(("foo", 1.0))
    assert ch.alerts == ()


def test_alert_engine_suppresses_exploding_channel() -> None:
    class ExplodingChannel(Channel):
        def send(self, alert: Alert) -> None:
            raise RuntimeError("boom")

    eng = AlertEngine(
        rules=[DriftSeverityRule()],
        channels=[ExplodingChannel(), InMemoryChannel()],
    )
    score = DriftScore(
        feature="f", method="psi", score=0.5, p_value=None, severity="critical",
        n_ref=100, n_live=100, extras={},
    )
    alerts = eng.evaluate(score)
    assert len(alerts) == 1  # engine still returns alerts even if a channel dies


def test_logging_channel_does_not_explode(caplog) -> None:  # type: ignore[no-untyped-def]
    import logging
    ch = LoggingChannel()
    a = Alert(
        rule_name="r", severity=Severity.WARNING, message="hi", source="x",
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    with caplog.at_level(logging.WARNING, logger="kairon.alerts"):
        ch.send(a)
    assert "hi" in caplog.text  # type: ignore[attr-defined]


def test_alert_engine_add_rule_and_channel() -> None:
    eng = AlertEngine()
    eng.add_rule(DriftSeverityRule())
    ch = InMemoryChannel()
    eng.add_channel(ch)
    score = DriftScore(
        feature="f", method="psi", score=0.5, p_value=None, severity="warning",
        n_ref=100, n_live=100, extras={},
    )
    eng.evaluate(score)
    assert len(ch.alerts) == 1
