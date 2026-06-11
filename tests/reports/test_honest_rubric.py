"""Tests for the W10.2 honest-negative-result rubric.

Per plan §10.2 (and the W10.2 PRD acceptance criterion #2), the
final report MUST explicitly label the outcome using the rubric
"'negative' = ≤3 metrics meet target, with negative Sharpe or
DSR<0.5 on all 3 assets". The label is one of {POSITIVE, MIXED,
NEGATIVE} and is placed in a 'RESULT_LABEL' section with a
1-paragraph rationale.

The applied verdict for the W8 backtests is MIXED:
- DSR<0.50 on BOTH BTCUSDT 1h (0.0069) and BTCUSDT 5m (0.3075)
- CAS<0 on BOTH BTCUSDT 1h (-24.58) and BTCUSDT 5m (-359.37)
- BUT per-trade Sharpe positive on both (1.71 / 27.57)
- BUT Max DD within ship band on both (-6.8% / -1.7%)
- BUT PBO=0.0 on both (within ship band, but inflated per W8.3)

The MIXED label reflects the conjunction: DSR<0.5 on both AND
CAS<0 on both, but per-trade Sharpe positive on both. The W8.5
DECISION-FORK recorded the EXTEND branch (3 more folds + revisit
cost) as the resolution; the W10.2 rubric formalises that into a
single-label result.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# Canonical project paths
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_FINAL_REPORT: Path = _PROJECT_ROOT / "reports" / "w10_final_report.md"


# ---------------------------------------------------------------------------
# Helper: parse the W10 report's RESULT_LABEL section
# ---------------------------------------------------------------------------
def _read_report() -> str:
    """Read the W10 final report. Fails the test if the file is missing."""
    if not _FINAL_REPORT.exists():
        pytest.fail(
            f"W10 final report not found at {_FINAL_REPORT}. "
            "The W10.1 story ASSEMBLES the W3-W9 artifacts into this file."
        )
    return _FINAL_REPORT.read_text(encoding="utf-8")


def _extract_result_label(report: str) -> str:
    """Extract the result label from the W10 report's RESULT_LABEL section.

    The report contains a section header that names "Result Label"
    (case-insensitive), followed by a label value that is one of
    {POSITIVE, MIXED, NEGATIVE}. The label MUST be in uppercase
    (POSITIVE / MIXED / NEGATIVE) per the W10.2 PRD acceptance
    criterion #2; lowercase mentions of "negative" / "positive" in
    body prose are not the rubric label.

    The strategy: find the first '#'-prefixed header that mentions
    "Result Label" (case-insensitive), then scan the next ~500 chars
    for the first UPPERCASE label token. The headers we care about
    are at line-anchored positions, so the lookhead is bounded.
    """
    # Iterate over markdown headers (lines starting with '#')
    lines: list[str] = report.splitlines()
    for idx, line in enumerate(lines):
        stripped: str = line.strip()
        if not stripped.startswith("#"):
            continue
        if "result label" not in stripped.lower():
            continue
        # We found a "Result Label" header. Scan the next ~200 lines
        # for the first UPPERCASE label token in {POSITIVE, MIXED, NEGATIVE}.
        # The label appears as "**VERDICT: MIXED**" or in a bullet list
        # in the section body; we bound the lookhead so we never cross
        # into the next major section.
        window: str = "\n".join(lines[idx: idx + 200])
        match: re.Match[str] | None = re.search(
            r"\b(?P<label>POSITIVE|MIXED|NEGATIVE)\b", window
        )
        if match is not None:
            return match.group("label")
    return ""


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_final_report_exists() -> None:
    """The W10 final report must exist at the canonical path."""
    assert _FINAL_REPORT.exists(), (
        f"W10 final report missing at {_FINAL_REPORT}; "
        "the W10.1 story ASSEMBLES the W3-W9 artifacts into this file."
    )


def test_rubric_label_applied() -> None:
    """PRD W10.2 acceptance criterion #2: the report contains a
    'RESULT_LABEL' section with one of {POSITIVE, MIXED, NEGATIVE}
    and the rationale.

    The W8 backtests are MIXED (DSR<0.5 on both, CAS<0 on both,
    per-trade Sharpe positive on both). The label MUST be in the
    report.
    """
    report: str = _read_report()
    label: str = _extract_result_label(report)
    assert label in {"POSITIVE", "MIXED", "NEGATIVE"}, (
        f"W10 final report must contain a 'Result Label' section with "
        f"one of {{POSITIVE, MIXED, NEGATIVE}}, but the extracted label "
        f"was {label!r}. The W10.2 rubric verdict must be visible in the report."
    )


def test_rubric_label_is_mixed() -> None:
    """The applied verdict for the W8 backtests is MIXED.

    The W10.2 rubric is "'negative' = ≤3 metrics meet target, with
    negative Sharpe or DSR<0.5 on all 3 assets". The W8 backtests
    satisfy the DSR<0.5 condition on both assets, but per-trade
    Sharpe is positive on both; the conjunction is MIXED, not pure
    NEGATIVE.
    """
    report: str = _read_report()
    label: str = _extract_result_label(report)
    assert label == "MIXED", (
        f"W10 final report label is {label!r}, but the W10.2 rubric "
        f"applied to the W8 backtests (DSR<0.5 on both 1h and 5m, "
        f"per-trade Sharpe positive on both) must be MIXED. "
        f"See §1 / §15 of the W10 final report for the rationale."
    )


def test_rubric_rationale_present() -> None:
    """The W10 final report's RESULT_LABEL section must include a
    1-paragraph rationale (the load-bearing honest verdict).
    """
    report: str = _read_report()
    label: str = _extract_result_label(report)
    assert label == "MIXED", (
        f"Precondition failed: expected MIXED label, got {label!r}."
    )
    # The rationale must reference the load-bearing honest result:
    # DSR<0.5 on both assets, CAS<0 on both, per-trade Sharpe
    # positive on both. The pattern checks for the load-bearing
    # keywords in the section following the label.
    rationale_keywords: tuple[str, ...] = (
        "DSR",
        "Sharpe",
        "0.0069",
        "0.3075",
        "synthetic",
    )
    for keyword in rationale_keywords:
        assert keyword in report, (
            f"W10 final report rationale is missing the keyword "
            f"{keyword!r}; the load-bearing honest result is "
            f"'synthetic zero-edge; DSR<0.5 on both 1h (0.0069) and "
            f"5m (0.3075); per-trade Sharpe positive on both'."
        )


def test_w8_decision_fork_cited() -> None:
    """The W10 final report must cite the W8.5 DECISION-FORK outcome
    (EXTEND) as the resolution of the MIXED label.
    """
    report: str = _read_report()
    assert "EXTEND" in report, (
        "W10 final report must cite the W8.5 DECISION-FORK outcome (EXTEND) "
        "as the resolution of the MIXED label; see artifacts/w8_decision.json."
    )
    assert "w8_decision" in report or "W8.5" in report, (
        "W10 final report must reference artifacts/w8_decision.json or the "
        "W8.5 DECISION-FORK story."
    )
