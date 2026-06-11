"""Tests for the W10.1 final report schema.

Per the W10.1 PRD acceptance criterion #1, the final report must
include: break-even table, ceiling table, coverage-accuracy Pareto,
CAS at cost shocks, regime breakdown, ablation JSON, 30-day paper
trade summary (W7 sim), W8 decision-fork outcome.

Per the W10.1 PRD acceptance criterion #2, the report is
schema-validated by tests/reports/test_report_schema.py (this file).

The W10.1 final report ASSEMBLES (does NOT regenerate) the W3-W9
artifacts. This test file pins the report's section structure: the
table of contents (with source-artifact citations) and the load-
bearing honest-rubric verdict.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# Canonical project paths
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_FINAL_REPORT: Path = _PROJECT_ROOT / "reports" / "w10_final_report.md"


# ---------------------------------------------------------------------------
# Required sections (per W10.1 PRD acceptance criterion #1)
# ---------------------------------------------------------------------------
_REQUIRED_SECTIONS: tuple[str, ...] = (
    # W10 metadata
    "What this project is and isn't",
    "Result Label",
    "Table of contents",
    # W2-W9 assemblies
    "Break-even accuracy table",
    "Ceiling accuracy table",
    "Coverage-accuracy Pareto",
    "CAS at cost shocks",
    "Regime breakdown",
    "Ablation JSON",
    "30-day paper trade summary",
    "W8 decision-fork outcome",
    "W8.3 honest report",
    "W9 regime fix",
    "W0 BTC-only fallback",
    "How to reproduce",
    "W10 honest rubric verdict",
    "Notes for the W10.3 2nd-human reviewer",
    # Status inventory
    "Status file inventory",
    "Source-of-truth summary",
)


# Required source-artifact citations in the table of contents
# (each section that ASSEMBLES from a W3-W9 artifact must cite the path)
_REQUIRED_SOURCE_CITATIONS: tuple[str, ...] = (
    "reports/break_even_w2.md",
    "artifacts/break_even_w2.json",
    "docs/objective_and_metrics.md",
    "reports/coverage_pareto_w4.json",
    "reports/cost_sensitivity_w2.md",
    "artifacts/cost_sensitivity_w2.json",
    "artifacts/w8_1_status.json",
    "artifacts/w8_2_status.json",
    "artifacts/w7_simulator.json",
    "artifacts/w8_decision.json",
    "reports/w8_honest_report.md",
    "reports/w9_regime_fix.md",
    "artifacts/w9_state.json",
    "reports/w0_fallback.md",
)


# ---------------------------------------------------------------------------
# Helper: read the W10 final report
# ---------------------------------------------------------------------------
def _read_report() -> str:
    """Read the W10 final report. Fails the test if the file is missing."""
    if not _FINAL_REPORT.exists():
        pytest.fail(
            f"W10 final report not found at {_FINAL_REPORT}. "
            "The W10.1 story ASSEMBLES the W3-W9 artifacts into this file."
        )
    return _FINAL_REPORT.read_text(encoding="utf-8")


def _section_headers(report: str) -> list[str]:
    """Return all markdown headers (lines starting with '#') in the report.

    Strips the leading '#' and surrounding whitespace.
    """
    headers: list[str] = []
    for line in report.splitlines():
        stripped: str = line.strip()
        if stripped.startswith("#"):
            # Strip the leading '#' characters and whitespace
            headers.append(stripped.lstrip("#").strip())
    return headers


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_final_report_exists() -> None:
    """W10.1 PRD acceptance criterion #1: reports/w10_final_report.md exists."""
    assert _FINAL_REPORT.exists(), (
        f"W10 final report missing at {_FINAL_REPORT}; the W10.1 story "
        f"ASSEMBLES the W3-W9 artifacts into this file."
    )


@pytest.mark.parametrize("section", _REQUIRED_SECTIONS)
def test_required_section_present(section: str) -> None:
    """W10.1 PRD acceptance criterion #1: the report must include
    break-even + ceiling + coverage Pareto + CAS at cost shocks +
    regime breakdown + ablation + 30-day paper trade summary +
    W8 decision-fork outcome, plus the W10 metadata sections.
    """
    report: str = _read_report()
    headers: list[str] = _section_headers(report)
    # Case-insensitive substring match against any header
    section_lower: str = section.lower()
    found: bool = any(section_lower in h.lower() for h in headers)
    assert found, (
        f"W10 final report is missing required section {section!r}. "
        f"Found headers: {headers}"
    )


@pytest.mark.parametrize("citation", _REQUIRED_SOURCE_CITATIONS)
def test_source_artifact_cited(citation: str) -> None:
    """W10.1 PRD: the report ASSEMBLES (does NOT regenerate) the W3-W9
    artifacts. Every section that ASSEMBLES from a W3-W9 artifact must
    cite the source path.
    """
    report: str = _read_report()
    assert citation in report, (
        f"W10 final report is missing the source-artifact citation "
        f"{citation!r}. The W10.1 report ASSEMBLES the W3-W9 artifacts "
        f"and must cite each source by path."
    )


def test_disclaimer_includes_btc_only_fallback() -> None:
    """W10.1 PRD: the disclaimer at the top must mention that the
    BTC-only fallback was active.
    """
    report: str = _read_report()
    # Check the disclaimer section
    assert "BTC-only fallback" in report or "W0" in report, (
        "W10 final report disclaimer must mention the BTC-only fallback "
        "(W0) as ACTIVE; the v1 deliverables use synthetic data per the W0 "
        "contingency."
    )


def test_disclaimer_includes_w8_5_extend() -> None:
    """W10.1 PRD: the disclaimer at the top must mention the W8.5
    EXTEND branch outcome.
    """
    report: str = _read_report()
    assert "EXTEND" in report, (
        "W10 final report disclaimer must mention the W8.5 EXTEND branch "
        "as the DECISION-FORK outcome; see artifacts/w8_decision.json."
    )


def test_disclaimer_includes_90pct_reframe() -> None:
    """W10.1 PRD: the disclaimer at the top must mention the 90%
    accuracy target reframe (DSR>=0.95 + CAS>=0.7 sustained across 3
    consecutive walk-forward folds in 2+ regimes).
    """
    report: str = _read_report()
    assert "DSR" in report and "0.95" in report, (
        "W10 final report disclaimer must mention the 90% accuracy target "
        "reframe (DSR>=0.95 + CAS>=0.7) per docs/objective_and_metrics.md §2."
    )


def test_how_to_reproduce_includes_commands() -> None:
    """W10.1 PRD: the 'How to reproduce' section must list the scripts
    to run (W2.2, W2.3, W3.5, W8.1, W8.2, W7, W9 CI gates).
    """
    report: str = _read_report()
    # The section should include the runnable script invocations
    reproduction_keywords: tuple[str, ...] = (
        "scripts.run_break_even",
        "scripts.run_cost_sensitivity",
        "scripts.run_coverage_curve",
        "scripts.run_e2e",
        "uv run pyright",
        "uv run pytest",
    )
    for keyword in reproduction_keywords:
        assert keyword in report, (
            f"W10 final report 'How to reproduce' section is missing the "
            f"command {keyword!r}; the section must list the scripts to run "
            f"to reproduce the W2-W9 deliverables."
        )


def test_report_is_assembled_not_regenerated() -> None:
    """W10.1 PRD: the report ASSEMBLES the W3-W9 artifacts (does NOT
    regenerate the W8 metrics or the W2 break-even table). The report
    must explicitly state this.
    """
    report: str = _read_report()
    # The report should mention the "ASSEMBLES" / "does not regenerate" /
    # "source-of-truth" pattern at least once
    assembly_keywords: tuple[str, ...] = (
        "ASSEMBLES",
        "does NOT regenerate",
        "does not regenerate",
        "source-of-truth",
    )
    found: bool = any(keyword in report for keyword in assembly_keywords)
    assert found, (
        "W10 final report must explicitly state that it ASSEMBLES the W3-W9 "
        "artifacts (does NOT regenerate the W8 metrics). The assembly-not-"
        "regeneration pattern is the load-bearing constraint from the W10.1 "
        "PRD."
    )


def test_report_includes_honest_negative_rubric() -> None:
    """W10.2 PRD: the final report must explicitly label the outcome
    using the rubric from plan §10.2. The label is in {POSITIVE,
    MIXED, NEGATIVE}.
    """
    report: str = _read_report()
    labels: tuple[str, ...] = ("POSITIVE", "MIXED", "NEGATIVE")
    found_labels: list[str] = [label for label in labels if label in report]
    assert len(found_labels) >= 1, (
        f"W10 final report must contain at least one of the result labels "
        f"{labels!r} for the W10.2 honest-negative-result rubric."
    )
    # Pin the applied verdict
    assert "MIXED" in found_labels, (
        f"W10 final report must contain the MIXED label (the W10.2 rubric "
        f"verdict for the W8 backtests). Found labels: {found_labels}"
    )


def test_report_does_not_inflate_results() -> None:
    """W10 PRD (Critic open question #2): the honest rubric is the
    load-bearing labeling; do NOT inflate or soften the result. The
    W10 report must mention the load-bearing honest result.
    """
    report: str = _read_report()
    # The load-bearing honest result is "synthetic zero-edge; DSR<0.95
    # is the load-bearing honest result"
    assert "zero-edge" in report or "no real edge" in report, (
        "W10 final report must mention that the v1 backtests are synthetic "
        "zero-edge by construction; do not inflate the result. The honest "
        "verdict is the load-bearing labeling."
    )
    # DSR < 0.95 must be cited explicitly
    assert re.search(r"DSR\s*<\s*0\.95", report) is not None, (
        "W10 final report must explicitly cite the load-bearing honest "
        "result 'DSR < 0.95' (per the W8.3 honest report)."
    )


def test_report_includes_w8_metrics_table() -> None:
    """W10.1 PRD: the report must include the W8 headline metrics
    (Sharpe, CAS, DSR, MaxDD) — these are the load-bearing inputs to
    the W10.2 rubric.
    """
    report: str = _read_report()
    # The W8 metrics must be cited in the report. Note: the report
    # uses "Max DD" (with a space) per the W8.3 honest report
    # convention; the W10 final report preserves that convention.
    w8_metrics: tuple[str, ...] = (
        "Sharpe",  # 1.71 / 27.57
        "CAS",  # -24.58 / -359.37
        "DSR",  # 0.0069 / 0.3075
        "Max DD",  # -6.8% / -1.7% (with a space, per W8.3 convention)
        "1.71",
        "27.57",
        "0.0069",
        "0.3075",
        "-6.8%",
        "-1.7%",
    )
    for metric in w8_metrics:
        assert metric in report, (
            f"W10 final report is missing the W8 metric {metric!r}; the W8 "
            f"headline metrics are the load-bearing inputs to the W10.2 rubric."
        )
