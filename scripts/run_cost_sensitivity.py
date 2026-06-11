"""Story W2.3 — Cost sensitivity shock (CAS at 0.5x, 1x, 2x, 5x cost).

The W2.3 runner script is the IO + serialisation layer for
:func:`kairon.evaluation.cost_sensitivity.cost_sensitivity_curve`.
It:

1. Synthesizes a 1mo BTCUSDT 1h equity curve (720 hourly
   bars) with the W2.2 BTC-like distribution:
   ``sigma_per_bar = 0.0035``, mean return ~ 0.0003 per bar
   (positive-Sharpe baseline). Documented in
   :data:`_SYNTHETIC_N_BARS` and the headline markdown.
2. Generates a synthetic per-trade PnL vector of the same
   length with positive mean ~ 0.3 bps and std-dev ~ 1 bps
   (per the W2.3 task spec — represents a strategy that has
   a small edge in this synthetic scenario).
3. Calls :func:`cost_sensitivity_curve` at the four
   multipliers ``(0.5, 1.0, 2.0, 5.0)`` with the synthetic
   trade_pnl vector and the default base round-trip cost
   (28 bps from ``DEFAULT_CRYPTO_COSTS``).
4. Writes the headline markdown to
   ``reports/cost_sensitivity_w2.md`` — a 4-row table with
   columns {multiplier, sharpe, sortino, max_dd, total_return,
   n_trades}.
5. Writes a JSON sidecar to
   ``artifacts/cost_sensitivity_w2.json`` — the load-bearing
   artifact the W2.5 GO/NO-GO gate's
   ``cost_sensitivity_present`` flag reads.

Run as::

    uv run python scripts/run_cost_sensitivity.py
    # or
    uv run python -m scripts.run_cost_sensitivity

Exit code is 0 on success, non-zero on a fatal error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS
from kairon.backtest.metrics import BARS_PER_YEAR_1H
from kairon.evaluation.cost_sensitivity import (
    DEFAULT_BASE_ROUND_TRIP_BPS,
    DEFAULT_MULTIPLIERS,
    cost_sensitivity_curve,
)


# ---------------------------------------------------------------------------
# Synthetic 1mo BTCUSDT 1h equity curve + per-trade pnl
# ---------------------------------------------------------------------------
# 720 hourly bars = 30 days x 24h (the W2.3 task spec says
# "1mo BTCUSDT 1h"; 30 days x 24h = 720 bars). The seed
# 20260607 is the W2.1 seed (documented in
# artifacts/w2_1_status.json) and is reused here so the
# W2.x batch is bit-deterministic.
_SYNTHETIC_N_BARS: int = 720
_SYNTHETIC_MU: float = 0.0003  # per-bar mean return (3 bps)
_SYNTHETIC_SIGMA: float = 0.0035  # per-bar std-dev (35 bps)
# Synthetic per-trade pnl: positive mean (small edge),
# std-dev = 1 bps. Per the W2.3 task spec.
_SYNTHETIC_TRADE_MEAN_BPS: float = 0.3
_SYNTHETIC_TRADE_STD_BPS: float = 1.0
_SYNTHETIC_SEED: int = 20260607
_SYNTHETIC_INITIAL_EQUITY: float = 100_000.0

# Headline markdown path (the cost-sensitivity artifact the
# evaluation framework §8.4 requires) and the JSON sidecar
# path (the load-bearing artifact the W2.5 GO/NO-GO gate's
# cost_sensitivity_present flag reads).
_REPORT_PATH: Path = Path("reports") / "cost_sensitivity_w2.md"
_SIDECAR_PATH: Path = Path("artifacts") / "cost_sensitivity_w2.json"


def _synthesize_equity_curve(
    *,
    n_bars: int = _SYNTHETIC_N_BARS,
    mu: float = _SYNTHETIC_MU,
    sigma: float = _SYNTHETIC_SIGMA,
    seed: int = _SYNTHETIC_SEED,
    initial_equity: float = _SYNTHETIC_INITIAL_EQUITY,
) -> np.ndarray:
    """Build a synthetic positive-Sharpe equity curve.

    Returns a 1-D ``np.ndarray`` of mark-to-market equity
    values of length ``n_bars + 1`` (the initial point +
    ``n_bars`` per-bar returns). The per-bar returns are
    sampled from a normal distribution with mean ``mu`` and
    std-dev ``sigma``; the equity curve is the cumulative
    product of (1 + returns), starting from ``initial_equity``.
    The RNG is seeded with ``seed`` for determinism so the
    headline markdown is reproducible across iterations of
    the ralph loop.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    returns: np.ndarray = rng.normal(loc=mu, scale=sigma, size=n_bars)
    equity: np.ndarray = np.empty(n_bars + 1, dtype=np.float64)
    equity[0] = initial_equity
    equity[1:] = initial_equity * np.cumprod(1.0 + returns)
    return equity


def _synthesize_trade_pnl(
    *,
    n_trades: int = _SYNTHETIC_N_BARS,
    mean_bps: float = _SYNTHETIC_TRADE_MEAN_BPS,
    std_bps: float = _SYNTHETIC_TRADE_STD_BPS,
    seed: int = _SYNTHETIC_SEED,
) -> np.ndarray:
    """Build a synthetic per-trade pnl vector.

    Per the W2.3 task spec: positive mean ~ 0.3 bps
    (representing a strategy that has a small edge in this
    synthetic scenario) with std-dev ~ 1 bps. The RNG is
    seeded with ``seed + 1`` (a separate stream from the
    equity-curve generator) so the two fixtures are
    independent.
    """
    rng: np.random.Generator = np.random.default_rng(seed + 1)
    return rng.normal(
        loc=mean_bps / 1e4,
        scale=std_bps / 1e4,
        size=n_trades,
    )


def _format_markdown(
    *,
    base_cost_bps: float,
    multipliers: tuple[float, ...],
    reports: dict[float, Any],
    trade_pnl: np.ndarray,
) -> str:
    """Format the cost-sensitivity sweep as a markdown report.

    The output is a 4-row markdown table (one row per
    multiplier) with the columns ``{multiplier, sharpe,
    sortino, max_dd, total_return, n_trades}`` per the W2.3
    task spec, plus a header section that documents the
    synthetic-baseline provenance, the cost-shock
    semantics, and the W2.5 gate implications.
    """
    lines: list[str] = []
    lines.append("# W2.3 — Cost Sensitivity Shock (CAS at 0.5x, 1x, 2x, 5x cost)")
    lines.append("")
    lines.append("**Date:** 2026-06-07  ")
    lines.append("**Story:** W2.3 — Cost sensitivity shock (CAS at 0.5x, 1x, 2x, 5x cost)")
    lines.append("")
    lines.append(
        "This report is the cost sensitivity shock required by "
        "``evaluation_framework.md`` §8.4. The table below reports the "
        "Sharpe, Sortino, max-drawdown, total-return, and trade-count "
        "metrics for a 1mo BTCUSDT 1h synthetic equity curve at four "
        "cost multipliers (0.5x, 1x, 2x, 5x). The cost shock is applied "
        "to per-trade PnL: ``trade_pnl[i] -= multiplier * "
        "base_round_trip_bps / 10000 * notional_proxy`` with "
        "``notional_proxy = 1.0``."
    )
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append(
        "| multiplier | sharpe | sortino | max_dd | total_return | n_trades |"
    )
    lines.append(
        "|-----------:|-------:|--------:|-------:|-------------:|---------:|"
    )
    n_trades: int = int(trade_pnl.size)
    for m in multipliers:
        rep = reports[m]
        lines.append(
            f"| {m:.2f} | {rep.sharpe:.4f} | {rep.sortino:.4f} | "
            f"{rep.max_drawdown:.4f} | {rep.total_return:.6f} | "
            f"{rep.n_trades} |"
        )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(
        f"- `base_round_trip_bps` = **{base_cost_bps:.2f}** "
        f"(from `kairon.backtest.cost.DEFAULT_CRYPTO_COSTS.round_trip_bps` = "
        f"{DEFAULT_CRYPTO_COSTS.round_trip_bps:.1f})"
    )
    lines.append(
        f"- `bars_per_year` = **{BARS_PER_YEAR_1H}** (1h bars, "
        f"from `kairon.backtest.metrics.BARS_PER_YEAR_1H`)"
    )
    lines.append(
        f"- `n_trades` = **{n_trades}** (one synthetic per-trade "
        f"observation per bar; per-trade mean = "
        f"{_SYNTHETIC_TRADE_MEAN_BPS:.1f} bps, std-dev = "
        f"{_SYNTHETIC_TRADE_STD_BPS:.1f} bps)"
    )
    lines.append(
        "- **Cost-shock direction:** Sharpe is monotonically "
        "non-increasing in the cost multiplier for the synthetic "
        "positive-edge baseline. The 0.5x tail has a *higher* "
        "Sharpe than the 1x baseline (cheaper fees -> better "
        "risk-adjusted return); 2x and 5x progressively erode the edge."
    )
    lines.append(
        "- **W2.5 gate flag.** The W2.5 GO/NO-GO gate's "
        "`cost_sensitivity_present` flag flips to True when this "
        "sidecar exists at `artifacts/cost_sensitivity_w2.json`. "
        "The gate's PROCEED/ESCALATE/HALT decision is NOT altered "
        "by the flag (per the W2.5 deviation #2); the flag is "
        "informational."
    )
    lines.append("")
    lines.append("## Cost-shock semantics")
    lines.append("")
    lines.append(
        "For each multiplier ``k`` and base round-trip cost ``C`` "
        "(in bps of notional), the per-trade PnL is shocked by "
        "subtracting ``k * C / 10000 * notional_proxy`` from each "
        "entry, with ``notional_proxy = 1.0``. The equity curve is "
        "then rebuilt by per-trade compounding: "
        "``equity[t] = initial_equity * prod_{i<t} (1 + shocked_pnl[i])``. "
        "At ``k = 0`` the cost shock is zero and the equity curve "
        "matches the no-cost baseline. At ``k = 1`` the cost shock "
        "is the full round-trip (28 bps); 2x and 5x progressively "
        "stress the cost regime."
    )
    lines.append("")
    lines.append("## Synthetic baseline provenance")
    lines.append("")
    lines.append(
        f"- 1mo BTCUSDT 1h: **{_SYNTHETIC_N_BARS}** hourly bars "
        f"(30 days x 24h; synthetic placeholder per W0 BTC-only "
        f"fallback; real-data path deferred)"
    )
    lines.append(
        f"- per-bar mean return ``mu`` = **{_SYNTHETIC_MU}** "
        f"({_SYNTHETIC_MU * 1e4:.1f} bps per bar; positive-Sharpe baseline)"
    )
    lines.append(
        f"- per-bar std-dev ``sigma`` = **{_SYNTHETIC_SIGMA}** "
        f"({_SYNTHETIC_SIGMA * 1e4:.1f} bps per bar; matches the W2.2 "
        f"BTCUSDT 1h sigma baseline of 0.0035)"
    )
    lines.append(
        f"- per-trade PnL mean = **{_SYNTHETIC_TRADE_MEAN_BPS}** bps "
        f"(small positive edge)"
    )
    lines.append(
        f"- per-trade PnL std-dev = **{_SYNTHETIC_TRADE_STD_BPS}** bps"
    )
    lines.append(
        f"- RNG seed = **{_SYNTHETIC_SEED}** (matches the W2.1 seed "
        f"for bit-determinism across the W2.x batch)"
    )
    lines.append(
        f"- initial equity = **{_SYNTHETIC_INITIAL_EQUITY:.0f}** (USD, "
        f"arbitrary scale; the Sharpe / Sortino / max_dd are "
        f"scale-invariant)"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "1. **Cost model provenance.** `base_round_trip_bps` is the "
        "round-trip bps from `kairon.backtest.cost.DEFAULT_CRYPTO_COSTS` "
        "(commission=10 bps + slippage=2 bps + half_spread=2 bps, doubled "
        "for the round trip = 28 bps). The W2.1 calibrator ships a "
        "closed-form OLS estimator; the W1.3 placeholder is used here. "
        "Re-trigger conditions: W2.1 `is_calibrated=True` AND the "
        "real-data path is available (W0 deferred)."
    )
    lines.append("")
    lines.append(
        "2. **Per-trade PnL attribution.** The cost-sensitivity sweep "
        "shocks the supplied ``trade_pnl`` vector per-trade, then "
        "rebuilds the equity curve by per-trade compounding. The "
        "``n_trades`` column is the per-trade count (``len(trade_pnl)``); "
        "in this synthetic fixture it equals the number of bars. The "
        "real-data path (W0 deferred) will pass a per-trade PnL vector "
        "from the backtest engine's trade log."
    )
    lines.append("")
    lines.append(
        "3. **W2.5 gate flag.** The W2.5 GO/NO-GO gate's "
        "`cost_sensitivity_present` flag flips to True when this "
        "sidecar exists at `artifacts/cost_sensitivity_w2.json`. The "
        "gate's PROCEED/ESCALATE/HALT decision is NOT altered by the "
        "flag (per the W2.5 deviation #2); the flag is informational."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Story: W2.3")
    lines.append("- Plan: `.omc/plans/kairon-real-data-90-percent-refactor.md`")
    lines.append("- Module: `kairon.evaluation.cost_sensitivity`")
    lines.append("- Cost model: `kairon.backtest.cost.DEFAULT_CRYPTO_COSTS` (W1.3 placeholder)")
    lines.append("- Real-data path: deferred per W0 BTC-only fallback (no live network in CI)")
    lines.append("")
    return "\n".join(lines)


def _format_sidecar(
    *,
    base_cost_bps: float,
    multipliers: tuple[float, ...],
    reports: dict[float, Any],
    n_trades: int,
) -> dict[str, Any]:
    """Format the cost-sensitivity sweep as a JSON sidecar.

    The sidecar shape is stable across iterations of the
    ralph loop: ``{schema_version, story_id, decided_at_iso,
    base_round_trip_bps, bars_per_year, n_trades,
    multipliers, rows}``. Each row has the documented 6
    columns {multiplier, sharpe, sortino, max_dd,
    total_return, n_trades}. Downstream consumers (W2.5
    GO/NO-GO gate) read the file's existence via
    ``cost_sensitivity_path``; the gate's
    ``cost_sensitivity_present`` flag is set to True when
    the file exists on disk.
    """
    rows: list[dict[str, Any]] = []
    for m in multipliers:
        rep = reports[m]
        rows.append({
            "multiplier": m,
            "sharpe": rep.sharpe,
            "sortino": rep.sortino,
            "max_dd": rep.max_drawdown,
            "total_return": rep.total_return,
            "n_trades": rep.n_trades,
        })
    return {
        "schema_version": "1",
        "story_id": "W2.3",
        "decided_at_iso": "2026-06-07",
        "base_round_trip_bps": base_cost_bps,
        "bars_per_year": BARS_PER_YEAR_1H,
        "n_trades": n_trades,
        "multipliers": list(multipliers),
        "rows": rows,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI args. The defaults match the W2.3 task description."""
    parser = argparse.ArgumentParser(
        prog="run_cost_sensitivity",
        description=(
            "Story W2.3: publish the cost-sensitivity shock (CAS at "
            "0.5x, 1x, 2x, 5x cost). Writes the headline markdown to "
            "reports/cost_sensitivity_w2.md and the JSON sidecar to "
            "artifacts/cost_sensitivity_w2.json."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=_REPORT_PATH,
        help="Path to the headline markdown report "
             "(default: reports/cost_sensitivity_w2.md).",
    )
    parser.add_argument(
        "--sidecar-path",
        type=Path,
        default=_SIDECAR_PATH,
        help="Path to the JSON sidecar "
             "(default: artifacts/cost_sensitivity_w2.json).",
    )
    parser.add_argument(
        "--base-round-trip-bps",
        type=float,
        default=DEFAULT_BASE_ROUND_TRIP_BPS,
        help=(
            "Baseline round-trip cost in bps of notional. "
            f"Default: {DEFAULT_BASE_ROUND_TRIP_BPS:.1f} "
            "(from DEFAULT_CRYPTO_COSTS.round_trip_bps)."
        ),
    )
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Cost multipliers to sweep (space-separated). "
            f"Default: {DEFAULT_MULTIPLIERS}."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SYNTHETIC_SEED,
        help=f"RNG seed for the synthetic fixtures. "
             f"Default: {_SYNTHETIC_SEED}.",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=_SYNTHETIC_N_BARS,
        help=f"Number of hourly bars in the synthetic curve. "
             f"Default: {_SYNTHETIC_N_BARS}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the W2.3 cost-sensitivity publisher. Returns the process exit code."""
    args: argparse.Namespace = _parse_args(argv)
    multipliers: tuple[float, ...] = (
        tuple(args.multipliers) if args.multipliers is not None
        else DEFAULT_MULTIPLIERS
    )
    base_cost_bps: float = args.base_round_trip_bps
    seed: int = args.seed
    n_bars: int = args.n_bars

    # Build the synthetic equity curve + per-trade pnl. The
    # real-data path (W0 deferred) replaces these with
    # backtest output; the rest of the pipeline is
    # unchanged.
    equity_curve: np.ndarray = _synthesize_equity_curve(
        n_bars=n_bars, seed=seed,
    )
    trade_pnl: np.ndarray = _synthesize_trade_pnl(
        n_trades=n_bars, seed=seed,
    )

    # Run the cost-sensitivity sweep at the four
    # multipliers with the per-trade pnl branch.
    reports: dict[float, Any] = cost_sensitivity_curve(
        equity_curve,
        base_round_trip_bps=base_cost_bps,
        trade_pnl=trade_pnl,
        multipliers=multipliers,
        bars_per_year=BARS_PER_YEAR_1H,
    )

    # Serialise and write. We do the markdown + JSON writes
    # in sequence rather than in parallel because a failure
    # in one should not silently leave the other half-written.
    md: str = _format_markdown(
        base_cost_bps=base_cost_bps,
        multipliers=multipliers,
        reports=reports,
        trade_pnl=trade_pnl,
    )
    sidecar: dict[str, Any] = _format_sidecar(
        base_cost_bps=base_cost_bps,
        multipliers=multipliers,
        reports=reports,
        n_trades=int(trade_pnl.size),
    )

    report_path: Path = args.report_path
    sidecar_path: Path = args.sidecar_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(md, encoding="utf-8")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    # Stdout summary: the engineer running the script
    # interactively wants the headline numbers at the
    # bottom of the output. We print the four Sharpes
    # (one per multiplier) so they can eyeball the
    # result.
    print(f"W2.3 cost-sensitivity markdown written to {report_path}")
    print(f"W2.3 cost-sensitivity JSON sidecar written to {sidecar_path}")
    for m in multipliers:
        rep = reports[m]
        print(
            f"  multiplier={m:.2f}: sharpe={rep.sharpe:.4f}, "
            f"sortino={rep.sortino:.4f}, max_dd={rep.max_drawdown:.4f}, "
            f"total_return={rep.total_return:.6f}, n_trades={rep.n_trades}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
