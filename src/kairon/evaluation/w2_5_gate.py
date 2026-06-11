"""W2.5 — GO/NO-GO gate.

Story W2.5 reads the W2.2 break-even accuracy sidecar at
``artifacts/break_even_w2.json`` and applies the plan's
PROCEED/ESCALATE/HALT decision logic. The function is
pure: it reads the JSON sidecar (and optionally a W2.3
cost-sensitivity sidecar), decides, and writes the verdict
to ``artifacts/w2_5_decision.json``. No async, no global
state, no network.

Decision logic (per the PRD W2.5 acceptance criteria and
``reports/break_even_w2.md`` note 4):

  - HALT     if any single (asset, horizon) break_even_pct
             exceeds ``halt_threshold_pct`` AND every row is
             non-viable (``viable=False`` on all).
  - ESCALATE if any single break_even_pct exceeds
             ``escalate_threshold_pct`` (i.e. on any single
             row, even if the others are viable).
  - PROCEED  otherwise.

The default thresholds are 0.80 for both halt and escalate
(per the PRD). The function returns the full decision
record (a dict) AND writes the same record to disk, so
downstream consumers (the ralph loop, the W3 entry point)
can read the verdict off the filesystem.

The cost-sensitivity flag is informational: W2.3 has not
landed yet (per the PRD the W2.3 story is still
``passes: false``), so the default cost_sensitivity_present
is False. When the W2.3 artifact lands, callers pass
``cost_sensitivity_path="artifacts/cost_sensitivity_w2.json"``
and the gate stamps ``cost_sensitivity_present=True`` on
the decision record; the decision itself is NOT altered by
the cost-sensitivity flag.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Decision constants. Exposed at module level so tests and
# downstream consumers can compare against the strings
# without a typo-magnet.
DECISION_PROCEED: str = "PROCEED"
DECISION_ESCALATE: str = "ESCALATE"
DECISION_HALT: str = "HALT"


def w2_5_go_no_go(
    *,
    break_even_path: str | Path = "artifacts/break_even_w2.json",
    halt_threshold_pct: float = 0.80,
    escalate_threshold_pct: float = 0.80,
    cost_sensitivity_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the W2.2 break-even JSON sidecar and apply the gate decision logic.

    Parameters
    ----------
    break_even_path
        Path to the W2.2 break-even JSON sidecar produced by
        ``scripts/run_break_even.py``. Default is the canonical
        path ``artifacts/break_even_w2.json``. Must exist; the
        function raises ``FileNotFoundError`` otherwise.
    halt_threshold_pct
        Break-even accuracy above which a single (asset,
        horizon) row is considered non-viable at the HALT
        level. Default ``0.80`` per the PRD.
    escalate_threshold_pct
        Break-even accuracy above which a single row triggers
        an ESCALATE decision. Default ``0.80`` per the PRD.
        The PRD sets halt and escalate to the same value; the
        two parameters are kept distinct so a future
        refinement (e.g. ESCALATE at 0.70, HALT at 0.80) is a
        one-line change.
    cost_sensitivity_path
        Optional path to the W2.3 cost-sensitivity sidecar. If
        provided AND the file exists, the returned record's
        ``cost_sensitivity_present`` is True. The decision is
        NOT altered by this flag; the W2.5 gate is a
        break-even gate per the PRD. W2.3 is independent
        (W2.3 is its own story with its own accept/reject).

    Returns
    -------
    dict[str, Any]
        The decision record with the following keys:

        - ``max_break_even_pct`` (float): max of
          ``break_even_pct`` across all rows.
        - ``n_assets`` (int): number of distinct assets in
          the sidecar.
        - ``n_horizons`` (int): number of distinct horizons.
        - ``n_viable`` (int): number of rows with
          ``viable=True``.
        - ``n_above_halt`` (int): number of rows with
          ``break_even_pct > halt_threshold_pct``.
        - ``n_above_escalate`` (int): number of rows with
          ``break_even_pct > escalate_threshold_pct``.
        - ``decision`` (str): one of ``"PROCEED"``,
          ``"ESCALATE"``, ``"HALT"``.
        - ``decided_at_iso`` (str): UTC ISO-8601 timestamp
          of the decision.
        - ``rationale`` (str): human-readable explanation
          citing the specific max, threshold, and counts.
        - ``report_path`` (str): absolute path of the W2.2
          break-even markdown report that the gate read.
        - ``cost_sensitivity_present`` (bool): True iff
          ``cost_sensitivity_path`` was provided and the file
          exists.

    Side Effects
    ------------
    Writes the decision record to
    ``artifacts/w2_5_decision.json`` (created if absent,
    overwritten if present). The path is relative to the
    current working directory, matching the W2.1/W2.2
    status-file convention.

    Raises
    ------
    FileNotFoundError
        If ``break_even_path`` does not exist.
    ValueError
        If the sidecar is not a JSON object, has no
        ``rows`` field, has zero rows, or if the threshold
        arguments are outside ``(0, 1)``.
    """
    # --- validate thresholds up front (fail fast) -----------------
    if not (0.0 < halt_threshold_pct <= 1.0):
        raise ValueError(
            f"halt_threshold_pct must be in (0, 1], got {halt_threshold_pct!r}"
        )
    if not (0.0 < escalate_threshold_pct <= 1.0):
        raise ValueError(
            f"escalate_threshold_pct must be in (0, 1], "
            f"got {escalate_threshold_pct!r}"
        )

    # --- read the W2.2 sidecar ------------------------------------
    be_path: Path = Path(break_even_path)
    if not be_path.exists():
        raise FileNotFoundError(
            f"break-even sidecar not found at {be_path}. "
            f"Run scripts/run_break_even.py first to produce it."
        )

    # The W2.2 sidecar is a JSON object written by
    # scripts/run_break_even.py. We type it as a dict of
    # object values so the per-row type checks below
    # remain the load-bearing runtime guards. JSON values
    # can be of any type, so a runtime isinstance check
    # is required even when the static type says "dict".
    sidecar_raw: str = be_path.read_text(encoding="utf-8")
    sidecar_parsed: Any = json.loads(sidecar_raw)
    if not isinstance(sidecar_parsed, dict):
        raise ValueError(
            f"break-even sidecar at {be_path} is not a JSON object, "
            f"got {type(sidecar_parsed).__name__}"
        )
    sidecar: dict[str, Any] = sidecar_parsed

    rows_raw: Any = sidecar.get("rows")
    if rows_raw is None:
        raise ValueError(
            f"break-even sidecar at {be_path} has no 'rows' field"
        )
    if not isinstance(rows_raw, list) or len(rows_raw) == 0:
        raise ValueError(
            f"break-even sidecar at {be_path} has empty or non-list 'rows'"
        )

    rows: list[Any] = rows_raw

    # --- compute the headline numbers ----------------------------
    # Every row must have a finite 'break_even_pct' (float)
    # and a boolean 'viable'. We extract in one pass so a
    # bad row fails fast with a clear message rather than
    # poisoning max() or the sum.
    be_pcts: list[float] = []
    viable_flags: list[bool] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"row #{idx} in {be_path} is not a dict, "
                f"got {type(row).__name__}"
            )
        if "break_even_pct" not in row or "viable" not in row:
            raise ValueError(
                f"row #{idx} in {be_path} is missing 'break_even_pct' "
                f"or 'viable'"
            )
        be_raw: object = row["break_even_pct"]
        viable_raw: object = row["viable"]
        if not isinstance(be_raw, (int, float)) or isinstance(be_raw, bool):
            raise ValueError(
                f"row #{idx} in {be_path} has non-numeric "
                f"break_even_pct={be_raw!r}"
            )
        if not isinstance(viable_raw, bool):
            raise ValueError(
                f"row #{idx} in {be_path} has non-boolean "
                f"viable={viable_raw!r}"
            )
        be_pcts.append(float(be_raw))
        viable_flags.append(viable_raw)

    max_be: float = max(be_pcts)
    n_viable: int = sum(1 for v in viable_flags if v)
    n_above_halt: int = sum(1 for b in be_pcts if b > halt_threshold_pct)
    n_above_escalate: int = sum(
        1 for b in be_pcts if b > escalate_threshold_pct
    )
    n_assets: int = len({
        str(r.get("asset", "")) for r in rows
        if isinstance(r.get("asset"), str)
    })
    n_horizons: int = len({
        str(r.get("horizon", "")) for r in rows
        if isinstance(r.get("horizon"), str)
    })

    # --- apply the decision logic --------------------------------
    # Branch order matters: HALT is the strictest condition
    # (all rows non-viable AND at least one row above the
    # halt threshold), ESCALATE is the next-strictest
    # (any single row above the escalate threshold), and
    # PROCEED is the catch-all.
    all_non_viable: bool = n_viable == 0
    any_above_halt: bool = n_above_halt > 0
    any_above_escalate: bool = n_above_escalate > 0

    if any_above_halt and all_non_viable:
        decision: str = DECISION_HALT
    elif any_above_escalate:
        decision = DECISION_ESCALATE
    else:
        decision = DECISION_PROCEED

    # --- resolve the cost-sensitivity flag -----------------------
    cost_sens_present: bool = False
    if cost_sensitivity_path is not None:
        cs_path: Path = Path(cost_sensitivity_path)
        cost_sens_present = cs_path.exists()

    # --- build the rationale -------------------------------------
    # The rationale is a one-or-two-sentence human-readable
    # explanation that downstream reviewers can paste into a
    # status file. It cites the specific max, the threshold,
    # and the count of rows that crossed it.
    total_rows: int = len(rows)
    rationale: str = (
        f"max(break_even_pct)={max_be:.4f} across {total_rows} rows "
        f"({n_assets} assets x {n_horizons} horizons). "
        f"Halt threshold={halt_threshold_pct:.2f} "
        f"(rows above: {n_above_halt}). "
        f"Escalate threshold={escalate_threshold_pct:.2f} "
        f"(rows above: {n_above_escalate}). "
        f"Viable rows: {n_viable}/{total_rows}. "
        f"Decision={decision}."
    )
    if decision == DECISION_HALT:
        rationale += (
            f" HALT: at least one (asset, horizon) pair crossed the "
            f"{halt_threshold_pct:.2f} halt threshold AND every row "
            f"is non-viable; the per-asset, per-horizon cost profile "
            f"is not exploitable at the current cost model."
        )
    elif decision == DECISION_ESCALATE:
        rationale += (
            f" ESCALATE: at least one (asset, horizon) pair crossed "
            f"the {escalate_threshold_pct:.2f} escalate threshold "
            f"while the remaining rows are still viable; the "
            f"recommended pivot per the plan is to 4h/1d horizons."
        )
    else:
        rationale += (
            f" PROCEED: max(break_even_pct) {max_be:.4f} is within "
            f"the viable band (break_even_pct <= 0.60) and below "
            f"the {halt_threshold_pct:.2f} halt / escalate "
            f"thresholds on every row; the W2.2 baseline supports "
            f"proceeding to W3."
        )

    # --- assemble the decision record ----------------------------
    decided_at: str = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    # Resolve the canonical markdown report path. The W2.2
    # sidecar lives at artifacts/break_even_w2.json and the
    # markdown report lives at reports/break_even_w2.md at
    # the same project root; we resolve both relative to the
    # sidecar's parent directory (artifacts/) and walk one
    # level up to find the project root.
    be_abs: Path = be_path.resolve()
    project_root: Path = be_abs.parent.parent
    report_path: str = str(project_root / "reports" / "break_even_w2.md")

    record: dict[str, Any] = {
        "max_break_even_pct": max_be,
        "n_assets": n_assets,
        "n_horizons": n_horizons,
        "n_viable": n_viable,
        "n_above_halt": n_above_halt,
        "n_above_escalate": n_above_escalate,
        "decision": decision,
        "decided_at_iso": decided_at,
        "rationale": rationale,
        "report_path": report_path,
        "cost_sensitivity_present": cost_sens_present,
    }

    # --- write the artifact --------------------------------------
    # The decision artifact is the load-bearing output the
    # ralph loop reads in the next iteration. We write
    # deterministically (indent=2, sorted keys) so a
    # byte-equal diff is stable across re-runs.
    out_path: Path = Path("artifacts") / "w2_5_decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return record


__all__ = [
    "DECISION_ESCALATE",
    "DECISION_HALT",
    "DECISION_PROCEED",
    "w2_5_go_no_go",
]
