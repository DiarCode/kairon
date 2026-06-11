"""Alerting: rules + dispatcher.

A simple rule engine that takes a stream of "facts" (drift scores,
metrics, or arbitrary events) and emits alerts. The rules are
configurable; the dispatcher just routes alerts to a list of
*channels*. The default channel is a no-op so the rule engine can
run in unit tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kairon.live.drift import DriftScore


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Alert:
    """A single alert."""

    rule_name: str
    severity: Severity
    message: str
    source: str
    created_at: datetime
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
class Rule:
    """Base class for alert rules — subclass and implement :meth:`matches`."""

    def __init__(self, name: str) -> None:
        self.name = name

    def matches(self, fact: Any) -> Alert | None:  # pragma: no cover - abstract
        raise NotImplementedError


class DriftSeverityRule(Rule):
    """Fire when a :class:`DriftScore` reaches a given severity."""

    def __init__(
        self,
        *,
        name: str = "drift_severity",
        features: tuple[str, ...] | None = None,
        levels: tuple[str, ...] = ("warning", "critical"),
    ) -> None:
        super().__init__(name)
        self.features = features
        self.levels = levels

    def matches(self, fact: Any) -> Alert | None:
        if not isinstance(fact, DriftScore):
            return None
        if self.features is not None and fact.feature not in self.features:
            return None
        if fact.severity not in self.levels:
            return None
        severity = Severity.CRITICAL if fact.severity == "critical" else Severity.WARNING
        return Alert(
            rule_name=self.name,
            severity=severity,
            message=(
                f"Drift {fact.method.upper()}={fact.score:.4f} on {fact.feature} "
                f"({fact.severity})"
            ),
            source=f"drift:{fact.feature}",
            created_at=datetime.now(UTC),
            extras={"score": fact.score, "method": fact.method},
        )


class ThresholdRule(Rule):
    """Fire when a numeric fact crosses a threshold."""

    def __init__(
        self,
        *,
        name: str,
        source: str,
        threshold: float,
        direction: str = "above",  # "above" | "below"
        severity: Severity = Severity.WARNING,
    ) -> None:
        if direction not in {"above", "below"}:
            raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
        super().__init__(name)
        self.source = source
        self.threshold = float(threshold)
        self.direction = direction
        self.severity = severity

    def matches(self, fact: Any) -> Alert | None:
        if not isinstance(fact, tuple) or len(fact) != 2:
            return None
        source, value = fact
        if source != self.source:
            return None
        if not isinstance(value, (int, float)):
            return None
        if self.direction == "above" and value > self.threshold:
            return Alert(
                rule_name=self.name,
                severity=self.severity,
                message=f"{self.source}={value:.4f} > {self.threshold:.4f}",
                source=self.source,
                created_at=datetime.now(UTC),
                extras={"value": float(value), "threshold": self.threshold},
            )
        if self.direction == "below" and value < self.threshold:
            return Alert(
                rule_name=self.name,
                severity=self.severity,
                message=f"{self.source}={value:.4f} < {self.threshold:.4f}",
                source=self.source,
                created_at=datetime.now(UTC),
                extras={"value": float(value), "threshold": self.threshold},
            )
        return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
class Channel:
    """A sink for alerts. Default is no-op."""

    def send(self, alert: Alert) -> None:
        return None


class InMemoryChannel(Channel):
    """Collects alerts in a list — useful for tests and replays."""

    def __init__(self) -> None:
        self._alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self._alerts.append(alert)

    @property
    def alerts(self) -> tuple[Alert, ...]:
        return tuple(self._alerts)

    def clear(self) -> None:
        self._alerts = []


class LoggingChannel(Channel):
    """Prints alerts via the standard library logger."""

    def __init__(self, name: str = "kairon.alerts") -> None:
        import logging
        self._log = logging.getLogger(name)

    def send(self, alert: Alert) -> None:
        if alert.severity == Severity.CRITICAL:
            self._log.critical(alert.message)
        elif alert.severity == Severity.WARNING:
            self._log.warning(alert.message)
        else:
            self._log.info(alert.message)


class AlertEngine:
    """The rule + channel glue."""

    def __init__(
        self,
        rules: list[Any] | None = None,
        channels: list[Channel] | None = None,
    ) -> None:
        self._rules: list[Any] = list(rules or [])
        self._channels: list[Channel] = list(channels or [])

    @property
    def n_alerts(self) -> int:
        return sum(
            len(c._alerts)  # type: ignore[attr-defined]
            for c in self._channels
            if isinstance(c, InMemoryChannel)
        )

    def add_rule(self, rule: Any) -> None:
        self._rules.append(rule)

    def add_channel(self, channel: Channel) -> None:
        self._channels.append(channel)

    def evaluate(self, fact: Any) -> tuple[Alert, ...]:
        alerts: list[Alert] = []
        for rule in self._rules:
            try:
                a = rule.matches(fact)
            except Exception:
                continue
            if a is not None:
                alerts.append(a)
        for a in alerts:
            for ch in self._channels:
                try:
                    ch.send(a)
                except Exception:
                    continue
        return tuple(alerts)


__all__ = [
    "Alert",
    "AlertEngine",
    "Channel",
    "DriftSeverityRule",
    "InMemoryChannel",
    "LoggingChannel",
    "Rule",
    "Severity",
    "ThresholdRule",
]
