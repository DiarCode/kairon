"""Tests for the label schema."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kairon.labels.schema import (
    DirectionClass,
    LabeledBar,
    LabeledFrame,
    LabelKind,
    LabelSpec,
)


def test_label_spec_rejects_bad_horizon() -> None:
    with pytest.raises(ValidationError):
        LabelSpec(kind=LabelKind.DIRECTION, horizon="5x")  # type: ignore[arg-type]


def test_label_spec_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LabelSpec(
            kind=LabelKind.DIRECTION,
            horizon="1h",
            unknown_field="x",  # type: ignore[call-arg]
        )


def test_label_spec_horizon_seconds() -> None:
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    assert spec.horizon_seconds == 3600
    spec = LabelSpec(kind=LabelKind.MAGNITUDE, horizon="5m")
    assert spec.horizon_seconds == 300


def test_label_spec_rejects_nan_in_params() -> None:
    with pytest.raises(ValidationError):
        LabelSpec(
            kind=LabelKind.TRIPLE_BARRIER,
            horizon="1h",
            params={"pt": float("nan")},
        )


def test_label_spec_is_frozen() -> None:
    spec = LabelSpec(kind=LabelKind.DIRECTION, horizon="1h")
    with pytest.raises(ValidationError):
        spec.kind = LabelKind.MAGNITUDE  # type: ignore[misc]


def test_labeled_bar_dataclass() -> None:
    bar = LabeledBar(
        symbol="BTC-USDT",
        ts=datetime(2024, 1, 1, tzinfo=UTC),
        horizon="1h",
        kind=LabelKind.DIRECTION,
        y=1,
        y_class=1,
    )
    assert bar.meta == {}
    assert bar.symbol == "BTC-USDT"


def test_labeled_frame_aggregates() -> None:
    bar = LabeledBar(
        symbol="BTC-USDT",
        ts=datetime(2024, 1, 1, tzinfo=UTC),
        horizon="1h",
        kind=LabelKind.DIRECTION,
        y=1,
    )
    frame = LabeledFrame(
        spec=LabelSpec(kind=LabelKind.DIRECTION, horizon="1h"),
        symbol="BTC-USDT",
        bars=(bar, bar, bar),
    )
    assert len(frame) == 3
    assert frame.ys == (1, 1, 1)
    assert frame.y_classes == (None, None, None)


def test_direction_class_values() -> None:
    assert int(DirectionClass.DOWN) == -1
    assert int(DirectionClass.FLAT) == 0
    assert int(DirectionClass.UP) == 1


def test_metalabel_kind() -> None:
    """W3.1: LabelKind.META is constructable; consumers are unaffected.

    Per plan risk #7, the META addition is a schema migration; the
    Literal extension is additive so existing ``LabelSpec(kind=...)``
    callers keep working. This test pins both halves of the contract:
    (a) ``LabelSpec(kind=LabelKind.META, horizon='1h')`` constructs
    without error, and (b) the value is the canonical string ``"meta"``
    so a serialised LabelSpec round-trips through JSON / YAML.
    """
    spec = LabelSpec(kind=LabelKind.META, horizon="1h")
    assert spec.kind is LabelKind.META
    assert spec.kind.value == "meta"
    # Existing kinds are unaffected (additive Literal extension).
    assert LabelSpec(kind=LabelKind.DIRECTION, horizon="1h").kind is LabelKind.DIRECTION
    assert LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon="1h").kind is LabelKind.TRIPLE_BARRIER
