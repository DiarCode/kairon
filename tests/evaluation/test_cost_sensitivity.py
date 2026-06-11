"""Tests for :mod:`kairon.evaluation.cost_sensitivity` and the W2.3 runner.

The four tests pin the W2.3 acceptance criteria:

1. ``test_cost_sensitivity_reduces_sharpe_with_higher_cost`` —
   on a positive-Sharpe synthetic equity curve + trade_pnl,
   the Sharpe at multiplier=5.0 is strictly lower than at
   multiplier=1.0.
2. ``test_cost_sensitivity_preserves_zero_cost_baseline`` —
   at multiplier=0.0 (or very small) the Sharpe equals the
   no-cost baseline; at multiplier=1.0 it is strictly lower.
3. ``test_cost_sensitivity_handles_missing_trade_pnl`` —
   when trade_pnl=None, the function uses the equity_curve
   only and still returns 4 PerformanceReport objects.
4. ``test_cost_sensitivity_table_writes_md_and_json`` — the
   runner script produces both files with the expected
   shape (4 rows, valid JSON, valid markdown with a table).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS
from kairon.evaluation.cost_sensitivity import (
    DEFAULT_BASE_ROUND_TRIP_BPS,
    DEFAULT_MULTIPLIERS,
    cost_sensitivity_curve,
)
from scripts import run_cost_sensitivity as rcs


# ---------------------------------------------------------------------------
# Synthetic positive-Sharpe fixtures
# ---------------------------------------------------------------------------
# 720 hourly bars = 30 days. The W2.3 task spec says 1mo
# BTCUSDT 1h synthetic equity curve; 30 days x 24h = 720 bars.
# Per-bar mean = 0.0003 (3 bps), per-bar std-dev = 0.0035
# (matches the W2.2 sigma_1h = 0.0035 used in the break-even
# table; the mean is set to give a positive but small Sharpe
# so a 28 bps cost shock erodes the edge in a way that the
# tests can pin directionally).
_N_BARS: int = 720
_MU: float = 0.0003  # 3 bps per bar
_SIGMA: float = 0.0035  # 35 bps per bar
_SEED: int = 20260607
_INITIAL_EQUITY: float = 100_000.0


def _make_equity_curve(
    n: int = _N_BARS,
    mu: float = _MU,
    sigma: float = _SIGMA,
    seed: int = _SEED,
    initial_equity: float = _INITIAL_EQUITY,
) -> np.ndarray:
    """Build a deterministic positive-Sharpe equity curve."""
    rng: np.random.Generator = np.random.default_rng(seed)
    rets: np.ndarray = rng.normal(loc=mu, scale=sigma, size=n)
    equity: np.ndarray = np.empty(n + 1, dtype=np.float64)
    equity[0] = initial_equity
    equity[1:] = initial_equity * np.cumprod(1.0 + rets)
    return equity


def _make_trade_pnl(
    n: int = _N_BARS,
    mean_bps: float = 0.3,
    std_bps: float = 1.0,
    seed: int = _SEED,
) -> np.ndarray:
    """Build a deterministic per-trade pnl vector.

    Mean is 0.3 bps (positive edge), std is 1 bps — represents
    a strategy that has a small edge in this synthetic scenario
    (per the W2.3 task spec).
    """
    rng: np.random.Generator = np.random.default_rng(seed + 1)
    return rng.normal(loc=mean_bps / 1e4, scale=std_bps / 1e4, size=n)


# ---------------------------------------------------------------------------
# Acceptance-criterion tests
# ---------------------------------------------------------------------------
def test_cost_sensitivity_reduces_sharpe_with_higher_cost() -> None:
    """Sharpe at multiplier=5.0 is strictly lower than at multiplier=1.0.

    On a positive-Sharpe synthetic equity curve + trade_pnl,
    the cost shock at 5x must produce a strictly lower Sharpe
    than the 1x baseline. This is the load-bearing direction
    the W2.3 evaluation framework requires.
    """
    equity: np.ndarray = _make_equity_curve()
    trade_pnl: np.ndarray = _make_trade_pnl()
    base_cost_bps: float = DEFAULT_BASE_ROUND_TRIP_BPS  # 28.0

    reports: dict[float, Any] = cost_sensitivity_curve(
        equity,
        base_round_trip_bps=base_cost_bps,
        trade_pnl=trade_pnl,
    )

    # All four default multipliers must be present.
    assert 0.5 in reports
    assert 1.0 in reports
    assert 2.0 in reports
    assert 5.0 in reports

    sharpe_1x: float = float(reports[1.0].sharpe)
    sharpe_5x: float = float(reports[5.0].sharpe)

    assert sharpe_5x < sharpe_1x, (
        f"5x Sharpe ({sharpe_5x:.4f}) must be strictly less than "
        f"1x Sharpe ({sharpe_1x:.4f}); higher cost must reduce "
        f"the risk-adjusted return"
    )

    # Sanity: the 0.5x tail should have a *higher* Sharpe
    # than the 1x baseline (cheaper fees -> better
    # risk-adjusted return). This is the directional check
    # the W2.3 evaluation framework requires.
    sharpe_0p5x: float = float(reports[0.5].sharpe)
    assert sharpe_0p5x > sharpe_1x, (
        f"0.5x Sharpe ({sharpe_0p5x:.4f}) must be strictly greater "
        f"than 1x Sharpe ({sharpe_1x:.4f}); lower cost must "
        f"improve the risk-adjusted return"
    )


def test_cost_sensitivity_preserves_zero_cost_baseline() -> None:
    """At very small multiplier, the Sharpe matches the no-cost baseline.

    At multiplier=0.0 the cost shock is zero, so the
    per-trade pnl is unchanged and the Sharpe equals the
    no-cost baseline. At multiplier=1.0 the cost shock is
    non-zero, so the Sharpe is strictly lower.
    """
    equity: np.ndarray = _make_equity_curve()
    trade_pnl: np.ndarray = _make_trade_pnl()
    base_cost_bps: float = DEFAULT_BASE_ROUND_TRIP_BPS  # 28.0

    # Sweep at 0.0 and 1.0 only; we don't need the full
    # default ladder for this test.
    reports: dict[float, Any] = cost_sensitivity_curve(
        equity,
        base_round_trip_bps=base_cost_bps,
        trade_pnl=trade_pnl,
        multipliers=(0.0, 1.0),
    )

    assert 0.0 in reports
    assert 1.0 in reports

    # Compute the no-cost baseline Sharpe directly from
    # the trade_pnl vector. At multiplier=0.0 the function
    # uses the per-trade pnl branch: the per-bar return is
    # shocked_pnl[i] = trade_pnl[i] - 0 = trade_pnl[i]. The
    # per-bar return series IS the (unshocked) trade_pnl
    # vector, so the no-cost baseline is summarize of an
    # equity curve where each per-bar return equals
    # trade_pnl[i]. The cleanest way to express this is to
    # pass trade_pnl as the per-bar return series directly
    # via the equity-curve-only branch (multiply by
    # something that gives trade_pnl as the diff returns).
    from kairon.backtest.metrics import (
        BARS_PER_YEAR_1H,
        sharpe_ratio,
    )
    baseline_sharpe: float = float(
        sharpe_ratio(trade_pnl, bars_per_year=BARS_PER_YEAR_1H)
    )

    sharpe_0x: float = float(reports[0.0].sharpe)
    assert sharpe_0x == pytest.approx(baseline_sharpe, abs=1e-9), (
        f"0x Sharpe ({sharpe_0x:.4f}) must match the no-cost "
        f"baseline ({baseline_sharpe:.4f}) to machine precision"
    )

    # At multiplier=1.0 the cost shock is non-zero, so
    # the Sharpe is strictly lower than the no-cost
    # baseline.
    sharpe_1x: float = float(reports[1.0].sharpe)
    assert sharpe_1x < sharpe_0x, (
        f"1x Sharpe ({sharpe_1x:.4f}) must be strictly less than "
        f"the no-cost baseline ({sharpe_0x:.4f}); a positive cost "
        f"shock must reduce the risk-adjusted return"
    )


def test_cost_sensitivity_handles_missing_trade_pnl() -> None:
    """When trade_pnl=None, the function uses the equity_curve only.

    The function falls back to the v1 mean-scaling branch
    and still returns 4 PerformanceReport objects (one per
    default multiplier). The reports are finite for a
    well-formed equity curve.
    """
    equity: np.ndarray = _make_equity_curve()
    base_cost_bps: float = DEFAULT_BASE_ROUND_TRIP_BPS  # 28.0

    reports: dict[float, Any] = cost_sensitivity_curve(
        equity,
        base_round_trip_bps=base_cost_bps,
        trade_pnl=None,  # explicit None -> equity-only branch
    )

    # All four default multipliers must be present.
    assert len(reports) == 4
    for m in DEFAULT_MULTIPLIERS:
        assert m in reports
        rep = reports[m]
        # Sharpe / Sortino / max_dd are finite for a
        # well-formed equity curve.
        assert np.isfinite(rep.sharpe), (
            f"multiplier={m} produced non-finite Sharpe={rep.sharpe}"
        )
        assert np.isfinite(rep.sortino), (
            f"multiplier={m} produced non-finite Sortino={rep.sortino}"
        )
        assert np.isfinite(rep.max_drawdown), (
            f"multiplier={m} produced non-finite max_dd={rep.max_drawdown}"
        )

    # And the cost-sensitivity direction is preserved:
    # 5x must have strictly lower Sharpe than 1x.
    assert reports[5.0].sharpe < reports[1.0].sharpe


def test_cost_sensitivity_table_writes_md_and_json(
    tmp_path: Path,
) -> None:
    """The runner produces both files with the expected shape.

    Drives :func:`rcs.main` in-process with the report +
    sidecar paths redirected to a tmp dir, then reads the
    files back and asserts:

    - the markdown file exists, is non-empty, has a
      4-row table (one per multiplier) with the
      documented columns {multiplier, sharpe, sortino,
      max_dd, total_return, n_trades}, and references
      the headline numbers;
    - the JSON sidecar parses cleanly, has 4 ``rows``,
      and every row has the documented columns.
    """
    report_path: Path = tmp_path / "reports" / "cost_sensitivity_w2.md"
    sidecar_path: Path = tmp_path / "artifacts" / "cost_sensitivity_w2.json"

    rc: int = rcs.main([
        "--report-path", str(report_path),
        "--sidecar-path", str(sidecar_path),
    ])
    assert rc == 0

    # --- markdown checks ----------------------------------------
    assert report_path.exists(), f"missing markdown report at {report_path}"
    md: str = report_path.read_text(encoding="utf-8")
    assert md, "markdown report is empty"

    # Find the main data table. The PRD W2.3 spec says the
    # columns are {multiplier, sharpe, sortino, max_dd,
    # total_return, n_trades} -> 6 columns -> 7 pipe chars
    # (incl. edges). Filter to those rows. We expect exactly
    # 4 data rows (one per multiplier) after the header and
    # separator lines.
    header_idx: int = -1
    for i, line in enumerate(md.splitlines()):
        if line.startswith("|") and "multiplier" in line.lower():
            header_idx = i
            break
    assert header_idx >= 0, "could not find table header in markdown"
    table_lines: list[str] = md.splitlines()[header_idx + 1:]
    data_only_clean: list[str] = [
        line for line in table_lines
        if line.startswith("|") and line.count("|") == 7
        and not line.startswith("|---")
        and not line.startswith("|------")
    ]
    assert len(data_only_clean) == 4, (
        f"expected 4 data rows (one per multiplier), got "
        f"{len(data_only_clean)}: {data_only_clean}"
    )

    # Required headline numbers are referenced.
    assert "sharpe" in md.lower()
    assert "sortino" in md.lower()
    assert "max_dd" in md.lower() or "max dd" in md.lower()
    assert "total_return" in md.lower() or "total return" in md.lower()
    assert "n_trades" in md.lower() or "n trades" in md.lower()

    # --- JSON sidecar checks ------------------------------------
    assert sidecar_path.exists(), f"missing JSON sidecar at {sidecar_path}"
    sidecar: dict[str, Any] = json.loads(sidecar_path.read_text(encoding="utf-8"))

    # 4 rows, one per default multiplier.
    assert "rows" in sidecar
    assert len(sidecar["rows"]) == 4

    # Every row has the documented columns.
    expected_columns: set[str] = {
        "multiplier", "sharpe", "sortino", "max_dd",
        "total_return", "n_trades",
    }
    for row in sidecar["rows"]:
        assert set(row.keys()) == expected_columns, (
            f"row keys {set(row.keys())} do not match expected "
            f"{expected_columns}"
        )
        assert row["multiplier"] in DEFAULT_MULTIPLIERS

    # Headline sanity: the sidecar exposes the canonical
    # fields the W2.5 gate reads.
    assert "schema_version" in sidecar
    assert "story_id" in sidecar
    assert "base_round_trip_bps" in sidecar
    # The base cost matches DEFAULT_CRYPTO_COSTS.
    assert sidecar["base_round_trip_bps"] == pytest.approx(
        DEFAULT_CRYPTO_COSTS.round_trip_bps, abs=1e-9,
    )
