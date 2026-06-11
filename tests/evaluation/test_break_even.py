"""Tests for :mod:`kairon.evaluation.break_even` and the W2.2 break-even table.

The five tests pin the W2.2 acceptance criteria:

1. ``test_break_even_formula`` — ``p* = 0.5 + C / (2R)`` within
   ``1e-9`` absolute tolerance, and the two known cases the
   PRD calls out: ``(R=10, C=2) -> p*=0.6`` and
   ``(R=20, C=2) -> p*=0.55``.
2. ``test_break_even_table_writes_md`` — the W2.2 runner
   script produces a valid markdown file at the canonical
   path with the right shape (3 assets x 4 horizons =
   12 rows).
3. ``test_break_even_table_writes_json`` — the JSON sidecar
   is valid JSON with the same 12-row shape and a
   ``headline.max_break_even_pct`` field.
4. ``test_break_even_viable_threshold`` — the ``viable``
   column is ``True`` when ``break_even_pct <= 0.60`` and
   ``False`` otherwise.
5. ``test_break_even_table_handles_unknown_asset`` — passing
   an asset not in ``cost_models`` raises ``ValueError``.

Tests 2 and 3 run the actual ``scripts/run_break_even.py``
entry point in a subprocess-style call to the ``main``
function (in-process import) and read the resulting files
off disk. This catches the IO + serialisation path that a
pure-function unit test would miss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.evaluation.break_even import break_even_accuracy
from kairon.evaluation.break_even_table import (
    DEFAULT_ASSETS,
    DEFAULT_HORIZONS,
    DEFAULT_VIABLE_THRESHOLD,
    build_break_even_table,
)
from scripts import run_break_even as rbe


# ---------------------------------------------------------------------------
# Synthetic per-bar sigma fixture (matches the runner script's
# _SYNTHETIC_SIGMA exactly so the table values are reproducible)
# ---------------------------------------------------------------------------
_SYNTHETIC_SIGMA: dict[tuple[str, str], float] = {
    # BTCUSDT (baseline)
    ("BTCUSDT", "5m"): 0.0008,
    ("BTCUSDT", "15m"): 0.0014,
    ("BTCUSDT", "1h"): 0.0035,
    ("BTCUSDT", "1d"): 0.012,
    # ETHUSDT (~1.3x BTC)
    ("ETHUSDT", "5m"): 0.00104,
    ("ETHUSDT", "15m"): 0.00182,
    ("ETHUSDT", "1h"): 0.00455,
    ("ETHUSDT", "1d"): 0.0156,
    # SOLUSDT (~1.8x BTC)
    ("SOLUSDT", "5m"): 0.00144,
    ("SOLUSDT", "15m"): 0.00252,
    ("SOLUSDT", "1h"): 0.0063,
    ("SOLUSDT", "1d"): 0.0216,
}

# Known tolerance: the PRD W2.2 acceptance criterion #1 says
# "p* = 0.5 + C / (2R) within 1e-9". We assert it with the
# explicit 1e-9 absolute tolerance the PRD specifies.
_ABS_TOL: float = 1e-9


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_break_even_formula() -> None:
    """``p* = 0.5 + C / (2R)`` within 1e-9, plus the two PRD-known cases.

    The PRD W2.2 acceptance criterion #1 says "p* = 0.5 +
    C / (2R) within 1e-9 absolute tolerance". We verify the
    closed-form against a hand-rolled reference and the two
    known cases the PRD calls out explicitly:
    ``(R=10, C=2) -> p*=0.6`` and ``(R=20, C=2) -> p*=0.55``.
    """
    # Generic closed-form check at 5 random (R, C) pairs.
    cases: list[tuple[float, float, float]] = [
        (10.0, 2.0, 0.5 + 2.0 / (2.0 * 10.0)),
        (20.0, 2.0, 0.5 + 2.0 / (2.0 * 20.0)),
        (50.0, 7.5, 0.5 + 7.5 / (2.0 * 50.0)),
        (200.0, 14.0, 0.5 + 14.0 / (2.0 * 200.0)),
        (1000.0, 5.0, 0.5 + 5.0 / (2.0 * 1000.0)),
    ]
    for r_bps, c_bps, expected in cases:
        p_star: float = break_even_accuracy(
            expected_move_bps=r_bps, round_trip_cost_bps=c_bps,
        )
        assert p_star == pytest.approx(expected, abs=_ABS_TOL), (
            f"p*={p_star} for R={r_bps}, C={c_bps} does not match "
            f"closed-form 0.5+C/(2R)={expected} within {_ABS_TOL}"
        )

    # The two PRD-known cases explicitly.
    p_10_2: float = break_even_accuracy(
        expected_move_bps=10.0, round_trip_cost_bps=2.0,
    )
    assert p_10_2 == pytest.approx(0.6, abs=_ABS_TOL)
    p_20_2: float = break_even_accuracy(
        expected_move_bps=20.0, round_trip_cost_bps=2.0,
    )
    assert p_20_2 == pytest.approx(0.55, abs=_ABS_TOL)


def test_break_even_table_writes_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner script produces a valid markdown file with 12 rows.

    Drives the W2.2 runner in-process via :func:`main` with
    the report + sidecar paths redirected to a tmp dir, then
    reads the markdown back and asserts:

    - the file exists at the canonical path
    - the file is non-empty
    - the markdown table has 12 data rows (3 assets x 4 horizons)
    - the file references the headline ``max(break_even_pct)``
      that the W2.5 gate reads
    """
    report_path: Path = tmp_path / "reports" / "break_even_w2.md"
    sidecar_path: Path = tmp_path / "artifacts" / "break_even_w2.json"
    # The runner script uses argparse defaults; we override
    # both via CLI args so the test does not pollute the
    # project's canonical reports/ and artifacts/ dirs.
    rc: int = rbe.main([
        "--report-path", str(report_path),
        "--sidecar-path", str(sidecar_path),
    ])
    assert rc == 0
    assert report_path.exists(), f"missing markdown report at {report_path}"
    md: str = report_path.read_text(encoding="utf-8")
    assert md, "markdown report is empty"
    # 3 assets x 4 horizons = 12 data rows. The headline
    # table has 6 columns; the "synthetic sigma baseline"
    # appendix table has 3 columns. We filter to the 6-column
    # table only so the appendix does not contaminate the
    # row count.
    asset_set: set[str] = set(DEFAULT_ASSETS)
    data_rows: list[str] = [
        line for line in md.splitlines()
        if line.startswith("| ") and any(
            line.startswith(f"| {asset} ") for asset in asset_set
        ) and line.count("|") == 7  # 6 columns -> 7 pipe chars (incl. edges)
    ]
    assert len(data_rows) == 12, (
        f"expected 12 data rows (3 assets x 4 horizons), got {len(data_rows)}: "
        f"{data_rows}"
    )
    # Headline number is the load-bearing value the W2.5
    # gate reads. Pin the regex to the W2.2 format so a
    # future headline rename catches the test.
    assert "max(break_even_pct)" in md
    assert "viable rows" in md.lower() or "viable" in md.lower()


def test_break_even_table_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The JSON sidecar is valid JSON with the same 12-row shape.

    Drives the same W2.2 runner in-process and reads the
    JSON sidecar back, asserting:

    - the sidecar exists at the canonical path
    - the JSON parses cleanly (``json.loads`` succeeds)
    - the sidecar has 12 ``rows``
    - the sidecar has a ``headline.max_break_even_pct`` field
      in (0.5, 1.0]
    - each row has the documented columns
    """
    report_path: Path = tmp_path / "reports" / "break_even_w2.md"
    sidecar_path: Path = tmp_path / "artifacts" / "break_even_w2.json"
    rc: int = rbe.main([
        "--report-path", str(report_path),
        "--sidecar-path", str(sidecar_path),
    ])
    assert rc == 0
    assert sidecar_path.exists(), f"missing JSON sidecar at {sidecar_path}"

    sidecar: dict[str, Any] = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert "rows" in sidecar
    assert len(sidecar["rows"]) == 12
    assert "headline" in sidecar
    max_be: float = sidecar["headline"]["max_break_even_pct"]
    assert 0.5 < max_be <= 1.0, f"max_break_even_pct={max_be} not in (0.5, 1.0]"

    # Every row must have the documented columns. The
    # builder's schema is the source of truth; we read the
    # first row's keys and assert all 12 rows have the same.
    expected_columns: set[str] = {
        "asset", "horizon", "expected_move_bps", "round_trip_cost_bps",
        "break_even_pct", "viable",
    }
    for row in sidecar["rows"]:
        assert set(row.keys()) == expected_columns, (
            f"row keys {set(row.keys())} do not match expected "
            f"{expected_columns}"
        )
        # Cross-check: every row must have a (asset, horizon)
        # combo in the documented asset x horizon grid.
        assert row["asset"] in DEFAULT_ASSETS
        assert row["horizon"] in DEFAULT_HORIZONS


def test_break_even_viable_threshold() -> None:
    """``viable`` is True iff ``break_even_pct <= 0.60``.

    The PRD W2.2 acceptance criterion #4 says "viable is
    true when break_even_pct <= 60%, false otherwise". We
    exercise the table builder with two fixtures: one where
    every (asset, horizon) is comfortably viable, and one
    where a small-R / high-C pair crosses the 60% threshold
    and flips to ``viable=False``.
    """
    # Fixture 1: comfortable sigma for every (asset, horizon);
    # round_trip_bps is the constant 28 bps from
    # DEFAULT_CRYPTO_COSTS. With the realistic W2.2 sigma
    # baseline, the break-even is well below 60% for every
    # row, so all 12 are ``viable=True``.
    cost_models: dict[str, CostModel] = {
        a: DEFAULT_CRYPTO_COSTS for a in DEFAULT_ASSETS
    }
    table: pa.Table = build_break_even_table(
        cost_models=cost_models, realized_sigma=_SYNTHETIC_SIGMA,
    )
    rows: list[dict[str, Any]] = table.to_pylist()
    assert len(rows) == 12
    for row in rows:
        is_viable: bool = row["viable"]
        be_pct: float = row["break_even_pct"]
        assert is_viable == (be_pct <= DEFAULT_VIABLE_THRESHOLD), (
            f"viable={is_viable} but break_even_pct={be_pct} and "
            f"threshold={DEFAULT_VIABLE_THRESHOLD}"
        )

    # Fixture 2: cross the 60% threshold. With the realistic
    # W2.2 sigma baseline the largest break-even is
    # ~0.547 (well under 60%), so the 60% threshold is not
    # crossed. To exercise the ``viable=False`` branch we
    # shrink sigma by 3x (sigma *= 1/3), which inflates the
    # break-even accuracy by 3x. The break-even is linear in
    # 1/sigma so the 0.547 -> 0.641 max, the smallest-C
    # / largest-sigma pair stays under 60% (it becomes
    # ~0.556), and the largest-C / smallest-sigma pair
    # (BTCUSDT 1d) crosses 60% to ``viable=False``. The
    # break-even does not saturate at 1.0 in this regime
    # (the max stays well below 1.0), so the threshold
    # check is the only thing that flips.
    third_sigma: dict[tuple[str, str], float] = {
        k: v / 3.0 for k, v in _SYNTHETIC_SIGMA.items()
    }
    table2: pa.Table = build_break_even_table(
        cost_models=cost_models, realized_sigma=third_sigma,
    )
    rows2: list[dict[str, Any]] = table2.to_pylist()
    any_not_viable: bool = any(not r["viable"] for r in rows2)
    assert any_not_viable, (
        "shrinking sigma by 3x should have flipped at least one row "
        "to viable=False (the realistic W2.2 baseline is calibrated "
        "so BTCUSDT 1d has the smallest per-bar sigma relative to its "
        "expected move and crosses 60% under the 3x shrink)"
    )
    max_be2: float = max(r["break_even_pct"] for r in rows2)
    assert max_be2 < 1.0, (
        f"max(break_even_pct)={max_be2} saturated at 1.0; the test "
        f"fixture should land in (0.6, 1.0) so the threshold check is "
        f"the only thing that flips (not the saturation guard)"
    )
    for row in rows2:
        is_viable = row["viable"]
        be_pct = row["break_even_pct"]
        assert is_viable == (be_pct <= DEFAULT_VIABLE_THRESHOLD), (
            f"third-sigma: viable={is_viable} but "
            f"break_even_pct={be_pct} and threshold={DEFAULT_VIABLE_THRESHOLD}"
        )


def test_break_even_table_handles_unknown_asset() -> None:
    """Passing an asset not in ``cost_models`` raises ``ValueError``.

    The PRD W2.2 acceptance criterion #5 says "passing an
    asset not in ``cost_models`` raises ``ValueError``. We
    build a table with the asset universe {BTCUSDT,
    ETHUSDT, SOLUSDT} but only the BTCUSDT cost model in
    the cost_models dict, plus a synthetic sigma entry for
    the missing assets so we cross the "missing sigma"
    check and reach the "missing cost model" check. The
    expected error message names the unknown assets.
    """
    cost_models_btc_only: dict[str, CostModel] = {
        "BTCUSDT": DEFAULT_CRYPTO_COSTS,
    }
    # Pad realized_sigma with entries for the missing
    # assets so the "missing sigma" guard is satisfied and
    # the "missing cost model" guard is the one that fires.
    realized_sigma_padded: dict[tuple[str, str], float] = {
        k: v for k, v in _SYNTHETIC_SIGMA.items()
    }
    # Sanity: ensure the missing-cost-model assets are
    # actually missing in the cost_models dict.
    for asset in ("ETHUSDT", "SOLUSDT"):
        assert asset not in cost_models_btc_only

    with pytest.raises(ValueError, match="assets not in cost_models"):
        build_break_even_table(
            assets=DEFAULT_ASSETS,  # the full 3-asset universe
            horizons=DEFAULT_HORIZONS,
            cost_models=cost_models_btc_only,  # BTCUSDT only
            realized_sigma=realized_sigma_padded,  # all 12 entries present
        )

    # The error message should also name the missing assets
    # so the engineer can see at a glance which cost model
    # is missing. We re-raise and assert the message.
    raised: bool = False
    try:
        build_break_even_table(
            assets=DEFAULT_ASSETS,
            horizons=DEFAULT_HORIZONS,
            cost_models=cost_models_btc_only,
            realized_sigma=realized_sigma_padded,
        )
    except ValueError as e:
        raised = True
        msg: str = str(e)
        assert "ETHUSDT" in msg
        assert "SOLUSDT" in msg
    assert raised, "ValueError was not raised for missing cost models"
