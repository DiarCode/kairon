"""Tests for :mod:`kairon.evaluation.w2_5_gate`.

The five tests pin the W2.5 GO/NO-GO gate acceptance
criteria:

1. ``test_proceed_when_all_viable`` — synthetic sidecar
   with max break-even=0.55, all rows viable=True
   -> decision='PROCEED'.
2. ``test_escalate_when_any_row_above_threshold`` —
   synthetic sidecar with one row at 0.85 and the
   remaining rows viable -> decision='ESCALATE'
   (the 0.85 row exceeds the 0.80 escalate threshold
   even though the others are viable).
3. ``test_halt_when_all_above_threshold`` — synthetic
   sidecar with max break-even=0.90, all rows
   viable=False -> decision='HALT'.
4. ``test_decision_writes_artifact`` — calling the gate
   on the real ``artifacts/break_even_w2.json`` produces
   a valid JSON at ``artifacts/w2_5_decision.json`` with
   the expected keys.
5. ``test_gate_includes_cost_sensitivity_flag`` — when
   ``cost_sensitivity_path`` is provided AND the file
   exists, ``cost_sensitivity_present=True``; otherwise
   False. The W2.3 artifact is not yet produced, so the
   default for the real sidecar is False.

The synthetic sidecars are written to ``tmp_path`` so the
test never mutates the canonical ``artifacts/`` directory.
Test 4 is the one that exercises the real sidecar and the
real output path; it uses ``monkeypatch.chdir`` to redirect
the decision artifact to ``tmp_path/artifacts/`` so the
canonical ``artifacts/w2_5_decision.json`` is not touched
by the test run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kairon.evaluation.w2_5_gate import (
    DECISION_ESCALATE,
    DECISION_HALT,
    DECISION_PROCEED,
    w2_5_go_no_go,
)


# Canonical project paths. Test 4 reads from these and
# redirects the OUTPUT path to tmp_path.
_PROJECT_BE_SIDECAR: Path = (
    Path(__file__).resolve().parents[2] / "artifacts" / "break_even_w2.json"
)


def _write_sidecar(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    n_assets: int = 3,
    n_horizons: int = 4,
) -> Path:
    """Write a minimal W2.2-shaped sidecar to ``path``.

    The shape matches the W2.2 schema (see
    ``artifacts/break_even_w2.json``): the gate only reads
    the ``rows`` field, so we keep the sidecar minimal
    (no headline, no cost_models). The caller passes
    ``rows`` directly with whatever ``break_even_pct`` /
    ``viable`` combination the test wants to exercise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar: dict[str, Any] = {
        "schema_version": "1",
        "story_id": "W2.2",
        "decided_at_iso": "2026-06-07",
        "n_assets": n_assets,
        "n_horizons": n_horizons,
        "rows": rows,
    }
    path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_proceed_when_all_viable(tmp_path: Path) -> None:
    """All rows viable and max break-even below 0.80 -> PROCEED.

    The PRD W2.5 acceptance criterion says: "if any
    single break_even > halt_threshold AND all rows are
    non-viable -> HALT; elif any single break_even >
    escalate_threshold -> ESCALATE; else -> PROCEED."

    With all 12 rows below 0.80 and ``viable=True``,
    the gate must return decision='PROCEED'.
    """
    sidecar: Path = tmp_path / "be.json"
    rows: list[dict[str, Any]] = [
        # 3 assets x 4 horizons = 12 rows; max break_even=0.55
        {
            "asset": f"A{i // 4}",
            "horizon": f"H{i % 4}",
            "break_even_pct": 0.50 + 0.01 * (i % 6),
            "viable": True,
        }
        for i in range(12)
    ]
    # Sanity: max is 0.55.
    assert max(r["break_even_pct"] for r in rows) == 0.55
    _write_sidecar(sidecar, rows)

    record: dict[str, Any] = w2_5_go_no_go(break_even_path=sidecar)
    assert record["decision"] == DECISION_PROCEED
    assert record["decision"] == "PROCEED"
    assert record["max_break_even_pct"] == pytest.approx(0.55, abs=1e-9)
    assert record["n_above_halt"] == 0
    assert record["n_above_escalate"] == 0
    assert record["n_viable"] == 12
    assert record["n_assets"] == 3
    assert record["n_horizons"] == 4
    assert record["cost_sensitivity_present"] is False
    assert "PROCEED" in record["rationale"]


def test_escalate_when_any_row_above_threshold(
    tmp_path: Path,
) -> None:
    """One row above 0.80 with other rows viable -> ESCALATE.

    Per the PRD: a single row crossing the escalate
    threshold is enough to trigger ESCALATE, even when
    the other rows are viable. The 0.85 row exceeds
    0.80, so the gate must return decision='ESCALATE'.
    """
    sidecar: Path = tmp_path / "be.json"
    # 11 rows at 0.55 (viable), 1 row at 0.85 (viable=True
    # so the HALT branch — which requires all rows
    # non-viable — is NOT taken).
    rows: list[dict[str, Any]] = [
        {
            "asset": "BTCUSDT",
            "horizon": "5m",
            "break_even_pct": 0.85,
            "viable": True,
        },
    ]
    for i in range(11):
        rows.append({
            "asset": f"A{i // 4}",
            "horizon": f"H{i % 4}",
            "break_even_pct": 0.55,
            "viable": True,
        })
    _write_sidecar(sidecar, rows)

    record: dict[str, Any] = w2_5_go_no_go(break_even_path=sidecar)
    assert record["decision"] == DECISION_ESCALATE
    assert record["decision"] == "ESCALATE"
    assert record["max_break_even_pct"] == pytest.approx(0.85, abs=1e-9)
    assert record["n_above_halt"] == 1
    assert record["n_above_escalate"] == 1
    assert record["n_viable"] == 12
    assert "ESCALATE" in record["rationale"]


def test_halt_when_all_above_threshold(tmp_path: Path) -> None:
    """Max break-even 0.90 and all rows non-viable -> HALT.

    The PRD: HALT fires when at least one row crosses
    the halt threshold AND every row is non-viable.
    This is the plan's pre-mortem scenario 1 (30%
    probability per the plan); the test pins the
    decision logic for that scenario.
    """
    sidecar: Path = tmp_path / "be.json"
    # 11 rows at 0.85 (non-viable), 1 row at 0.90 (non-viable).
    rows: list[dict[str, Any]] = []
    for i in range(11):
        rows.append({
            "asset": f"A{i // 4}",
            "horizon": f"H{i % 4}",
            "break_even_pct": 0.85,
            "viable": False,
        })
    rows.append({
        "asset": "BTCUSDT",
        "horizon": "1d",
        "break_even_pct": 0.90,
        "viable": False,
    })
    _write_sidecar(sidecar, rows)

    record: dict[str, Any] = w2_5_go_no_go(break_even_path=sidecar)
    assert record["decision"] == DECISION_HALT
    assert record["decision"] == "HALT"
    assert record["max_break_even_pct"] == pytest.approx(0.90, abs=1e-9)
    assert record["n_above_halt"] == 12
    assert record["n_above_escalate"] == 12
    assert record["n_viable"] == 0
    assert "HALT" in record["rationale"]


def test_decision_writes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate writes a valid JSON artifact at the canonical path.

    Calls the gate on the real ``artifacts/break_even_w2.json``
    and verifies that ``artifacts/w2_5_decision.json`` is
    written with the expected keys. The output path is
    redirected to ``tmp_path`` via ``monkeypatch.chdir`` so
    the test does not pollute the project's canonical
    ``artifacts/`` directory.
    """
    if not _PROJECT_BE_SIDECAR.exists():
        pytest.skip(
            f"real W2.2 sidecar not found at {_PROJECT_BE_SIDECAR}; "
            f"run scripts/run_break_even.py first"
        )

    # Redirect CWD to tmp_path so artifacts/w2_5_decision.json
    # is written under tmp_path and the canonical artifacts/
    # directory is untouched. The gate uses Path("artifacts")
    # relative to CWD, so chdir is the cleanest way to redirect
    # without exposing an output_path argument.
    monkeypatch.chdir(tmp_path)

    record: dict[str, Any] = w2_5_go_no_go(
        break_even_path=_PROJECT_BE_SIDECAR,
    )

    # The artifact must exist on disk at the canonical relative
    # path (now resolved under tmp_path).
    out_path: Path = tmp_path / "artifacts" / "w2_5_decision.json"
    assert out_path.exists(), (
        f"gate did not write decision artifact at {out_path}"
    )

    # The file is valid JSON with the expected keys.
    on_disk: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    expected_keys: set[str] = {
        "max_break_even_pct",
        "n_assets",
        "n_horizons",
        "n_viable",
        "n_above_halt",
        "n_above_escalate",
        "decision",
        "decided_at_iso",
        "rationale",
        "report_path",
        "cost_sensitivity_present",
    }
    assert set(on_disk.keys()) == expected_keys, (
        f"artifact keys {set(on_disk.keys())} do not match expected "
        f"{expected_keys}"
    )

    # The on-disk content matches the in-memory return value.
    assert on_disk == record

    # The real W2.2 data has max(break_even_pct)=0.5473 with
    # all 12 rows viable; the gate must say PROCEED.
    assert on_disk["decision"] == "PROCEED"
    assert on_disk["max_break_even_pct"] == pytest.approx(
        0.5473, abs=1e-3
    )
    assert on_disk["n_above_halt"] == 0
    assert on_disk["n_above_escalate"] == 0
    assert on_disk["n_viable"] == 12
    assert on_disk["n_assets"] == 3
    assert on_disk["n_horizons"] == 4
    assert on_disk["cost_sensitivity_present"] is False


def test_gate_includes_cost_sensitivity_flag(
    tmp_path: Path,
) -> None:
    """``cost_sensitivity_present`` reflects whether the file exists.

    The W2.5 acceptance criterion: when
    ``cost_sensitivity_path`` is provided AND the file
    exists, the returned record's
    ``cost_sensitivity_present`` is True; otherwise False.
    The W2.3 artifact is not yet produced, so the default
    for the real sidecar is False. This test pins both
    branches (file present -> True; file absent -> False)
    and confirms the path with no cost_sensitivity_path
    argument at all is also False.
    """
    sidecar: Path = tmp_path / "be.json"
    rows: list[dict[str, Any]] = [
        {
            "asset": "BTCUSDT",
            "horizon": "5m",
            "break_even_pct": 0.55,
            "viable": True,
        },
    ]
    _write_sidecar(sidecar, rows)

    # Branch 1: cost_sensitivity_path=None -> False.
    record_none: dict[str, Any] = w2_5_go_no_go(break_even_path=sidecar)
    assert record_none["cost_sensitivity_present"] is False

    # Branch 2: cost_sensitivity_path provided, file
    # does NOT exist -> False.
    missing_cs: Path = tmp_path / "cost_sens_does_not_exist.json"
    record_missing: dict[str, Any] = w2_5_go_no_go(
        break_even_path=sidecar,
        cost_sensitivity_path=missing_cs,
    )
    assert record_missing["cost_sensitivity_present"] is False

    # Branch 3: cost_sensitivity_path provided, file
    # DOES exist -> True.
    present_cs: Path = tmp_path / "cost_sens.json"
    present_cs.write_text("{}\n", encoding="utf-8")
    record_present: dict[str, Any] = w2_5_go_no_go(
        break_even_path=sidecar,
        cost_sensitivity_path=present_cs,
    )
    assert record_present["cost_sensitivity_present"] is True

    # The decision is NOT altered by the cost-sensitivity
    # flag in any of the three branches.
    assert record_none["decision"] == record_missing["decision"]
    assert record_none["decision"] == record_present["decision"]
    assert record_none["decision"] == "PROCEED"
