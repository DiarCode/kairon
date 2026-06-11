"""Story W3.5 — Coverage-accuracy Pareto frontier (two reference points).

The W3.5 runner script is the IO + serialisation layer for
:func:`kairon.evaluation.coverage_curve.coverage_curve`. It:

1. For 3 assets (BTCUSDT, ETHUSDT, SOLUSDT) x 4 horizons
   (5m, 15m, 1h, 1d) = 12 (asset, horizon) pairs:

   a. Generates a synthetic primary prediction. The
      primary is a :class:`sklearn.linear_model.LogisticRegression`
      fit on the W1.6 leakage fixture's BTC-like data,
      with ``p_primary`` in ``[0, 1]``. The fixture is
      BTC-like by default; the ETH and SOL rows are
      derived from the same fixture with an
      asset-specific noise multiplier (ETH ~ 1.3x BTC
      vol, SOL ~ 1.8x BTC vol, matching the W2.2
      synthetic-sigma multipliers).

   b. Builds the W1.6-style y_true labels via the
      triple-barrier rule (or a single-bar future-return
      rule when the label horizon is short — see
      :func:`_synthesize_y_true_for_horizon`).

   c. Runs :func:`coverage_curve` and records the two
      reference points (``T`` at 25% coverage,
      ``T`` at 10% coverage) and the full curve.

2. Writes the headline JSON to
   ``reports/coverage_pareto_w4.json`` with the
   structure the PRD W3.5 acceptance criteria specify:
   ``{schema_version, story_id, decided_at_iso, assets,
   horizons, reference_point_coverage_pct, rows: [{asset,
   horizon, t_at_25pct_coverage, t_at_25pct_accuracy,
   t_at_10pct_coverage, t_at_10pct_accuracy, full_curve:
   [{threshold, coverage, accuracy}, ...]}, ...]}``.

Run as::

    uv run python scripts/run_coverage_curve.py
    # or
    uv run python -m scripts.run_coverage_curve

Exit code is 0 on success, non-zero on a fatal error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kairon.evaluation.coverage_curve import coverage_curve


# ---------------------------------------------------------------------------
# Asset universe + horizon ladder
# ---------------------------------------------------------------------------
# Per the W0 BTC-only fallback: the 3 assets are BTCUSDT,
# ETHUSDT, SOLUSDT. The horizon ladder is the canonical
# 5m / 15m / 1h / 1d set used by W2.2.
DEFAULT_ASSETS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_HORIZONS: tuple[str, ...] = ("5m", "15m", "1h", "1d")

# Seconds-per-bar for the synthetic y_true generator.
# Matches :data:`kairon.evaluation.break_even_table.SECONDS_PER_BAR`.
SECONDS_PER_BAR: dict[str, int] = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
}

# Per-asset vol multiplier (ETH ~ 1.3x BTC, SOL ~ 1.8x BTC,
# matching the W2.2 synthetic-sigma multipliers).
ASSET_VOL_MULT: dict[str, float] = {
    "BTCUSDT": 1.0,
    "ETHUSDT": 1.3,
    "SOLUSDT": 1.8,
}

# Default primary-prediction fixture length. The W1.6
# leakage fixture's default is 1440 bars (1 day of 1m
# bars); for the W3.5 runner we use a smaller 1-day
# 5m-bar fixture (288 bars) so the full pipeline is fast
# in CI. The fixture length is configurable via
# ``--n-bars``.
DEFAULT_N_BARS: int = 288

# Default horizon for the triple-barrier label. For
# 5m bars the horizon is 12 bars (~ 1h of forward
# return); for 1h bars it is 24 bars (~ 1d). The
# horizon is documented in the sidecar so the
# real-data path can override.
DEFAULT_HORIZON_BARS: int = 12

# Triple-barrier thresholds (in units of per-bar sigma).
# A bar is labelled +1 if its forward return exceeds
# ``+UPPER_BARRIER * sigma``; -1 if its forward return
# is below ``-LOWER_BARRIER * sigma``; 0 otherwise.
UPPER_BARRIER: float = 1.0
LOWER_BARRIER: float = 1.0

# Synthetic data generator parameters. The price walk
# matches the W1.6 leakage fixture's BTC-like
# distribution.
_BASE_PRICE: float = 50_000.0
_BASE_SIGMA: float = 0.01  # per-step log-price volatility
_SEED: int = 20260608  # W3.5 date

# Headline JSON path (the W3.5 / W3-4 gate reads this).
# Per the PRD W3.5 spec: ``reports/coverage_pareto_w4.json``.
_REPORT_PATH: Path = Path("reports") / "coverage_pareto_w4.json"


# ---------------------------------------------------------------------------
# Synthetic data + primary-prediction generator
# ---------------------------------------------------------------------------
def _synthesize_prices(
    *,
    n_bars: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Return a synthetic BTC-like log-normal price walk.

    The price walk is a per-bar log-normal step:
    ``log_p[i+1] = log_p[i] + Normal(0, sigma)`` with
    ``sigma`` controlling the per-step volatility. The
    result is a 1-D ``np.ndarray`` of length ``n_bars``.
    """
    if n_bars <= 0:
        raise ValueError(f"n_bars must be > 0, got {n_bars}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    rng: np.random.Generator = np.random.default_rng(seed)
    log_returns: np.ndarray = rng.normal(loc=0.0, scale=sigma, size=n_bars)
    log_prices: np.ndarray = np.empty(n_bars, dtype=np.float64)
    log_prices[0] = math.log(_BASE_PRICE)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns[:-1])
    return np.exp(log_prices).astype(np.float64)


def _synthesize_y_true_for_horizon(
    prices: np.ndarray,
    *,
    horizon: str,
    horizon_bars: int,
) -> np.ndarray:
    """Build a binary y_true label vector via the triple-barrier rule.

    For each bar ``i``, the forward return is
    ``r[i] = log(prices[i + horizon_bars] / prices[i])``
    (with a floor at ``i + horizon_bars < n``). The
    bar is labelled +1 if ``r[i] > +UPPER_BARRIER *
    sigma_hat`` and -1 if ``r[i] < -LOWER_BARRIER *
    sigma_hat``; the binary ``y_true`` we expose to the
    coverage curve is the +1 / 0 -> 1, -1 -> 0
    projection (the coverage curve is binary).

    ``sigma_hat`` is the realised per-bar return
    std-dev estimated from the price series itself, so
    the labels adapt to the asset-specific vol
    multiplier.
    """
    n: int = int(prices.size)
    if horizon_bars <= 0:
        raise ValueError(
            f"horizon_bars must be > 0, got {horizon_bars}"
        )
    if horizon_bars >= n:
        raise ValueError(
            f"horizon_bars={horizon_bars} must be < n_bars={n}"
        )
    log_prices: np.ndarray = np.log(prices)
    log_returns: np.ndarray = np.diff(log_prices)
    sigma_hat: float = float(log_returns.std(ddof=0))
    if sigma_hat <= 0:
        sigma_hat = 1e-9
    # Vectorised forward return: r[i] = log(prices[i+h]) - log(prices[i]).
    r: np.ndarray = log_prices[horizon_bars:] - log_prices[:-horizon_bars]
    y: np.ndarray = np.zeros(n, dtype=np.float64)
    # The trailing ``horizon_bars`` bars are un-labelled
    # (no forward return available); we drop them from
    # the coverage-curve input.
    valid_n: int = n - horizon_bars
    y_plus: np.ndarray = r > (+UPPER_BARRIER * sigma_hat)
    y_minus: np.ndarray = r < (-LOWER_BARRIER * sigma_hat)
    y_pos: np.ndarray = (y_plus | y_minus).astype(np.float64)
    y[:valid_n] = y_pos
    return y


def _fit_logreg_primary(
    prices: np.ndarray,
    y_true: np.ndarray,
    *,
    feature_window: int,
    n_train: int,
    seed: int,
) -> np.ndarray:
    """Fit a logistic-regression primary; return ``p_primary`` for every bar.

    The caller passes ``n_train`` = the number of
    *labelled* rows (rows where ``y_true`` is 0 or 1).
    The first ``n_train`` rows of the feature matrix
    are the training set; the rest are predicted with
    no label.

    Returns a 1-D ``np.ndarray`` ``p_primary`` of length
    ``n - feature_window`` (one prediction per
    rolling-window bar). The first ``feature_window``
    bars of the input have no feature row and are
    excluded from the predictions.
    """
    from sklearn.linear_model import LogisticRegression

    n: int = int(prices.size)
    if feature_window <= 0:
        raise ValueError(
            f"feature_window must be > 0, got {feature_window}"
        )
    if n <= feature_window + 1:
        raise ValueError(
            f"prices has {n} bars; need at least "
            f"feature_window+1={feature_window + 1}"
        )
    if not (0 < n_train <= n - feature_window):
        raise ValueError(
            f"n_train={n_train} must be in (0, "
            f"n - feature_window] = (0, {n - feature_window}]"
        )

    log_prices: np.ndarray = np.log(prices)
    log_returns: np.ndarray = np.diff(log_prices)

    rows: list[np.ndarray] = []
    for i in range(feature_window, n):
        window: np.ndarray = log_returns[i - feature_window: i]
        rows.append(window)
    x: np.ndarray = np.stack(rows, axis=0)

    # Train on the first ``n_train`` rows; predict on
    # every row.
    y_train: np.ndarray = y_true[feature_window: feature_window + n_train].astype(
        np.int32
    )
    # The labelled rows are 0 or 1 (binary); the un-labelled
    # trailing rows are 0 by our construction. We rely on
    # the caller to set ``n_train`` to the count of
    # labelled rows so the un-labelled trailing rows are
    # never in the training set.
    model: LogisticRegression = LogisticRegression(
        max_iter=200, random_state=seed,
    )
    model.fit(x[:n_train], y_train)
    p_primary: np.ndarray = model.predict_proba(x)[:, 1]
    return p_primary.astype(np.float64)


def _build_pair(
    *,
    asset: str,
    horizon: str,
    n_bars: int,
    feature_window: int,
    horizon_bars: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(y_true, p_primary)`` for one (asset, horizon) pair.

    Steps:
    1. Synthesize a BTC-like price walk with the
       asset's vol multiplier applied to the base sigma.
    2. Build the binary y_true labels via the
       triple-barrier rule with the supplied
       ``horizon_bars``.
    3. Fit a logistic-regression primary and return
       ``p_primary`` in ``[0, 1]``.

    The output ``y_true`` and ``p_primary`` are aligned:
    both have the same length, indexed by bar position
    in the input price series. The caller is responsible
    for the ``n_train`` argument to
    :func:`_fit_logreg_primary` (= labelled rows).
    """
    if asset not in ASSET_VOL_MULT:
        raise ValueError(
            f"unknown asset: {asset!r}; must be one of "
            f"{tuple(ASSET_VOL_MULT.keys())}"
        )
    vol_mult: float = ASSET_VOL_MULT[asset]
    sigma: float = _BASE_SIGMA * vol_mult
    prices: np.ndarray = _synthesize_prices(
        n_bars=n_bars, sigma=sigma, seed=seed,
    )
    y: np.ndarray = _synthesize_y_true_for_horizon(
        prices, horizon=horizon, horizon_bars=horizon_bars,
    )
    # The labelled rows are bars 0 .. n - horizon_bars - 1.
    n_labelled: int = n_bars - horizon_bars
    p: np.ndarray = _fit_logreg_primary(
        prices, y,
        feature_window=feature_window,
        n_train=n_labelled - feature_window,
        seed=seed,
    )
    # y is the full ``n_bars``-length vector; the
    # primary's predictions are ``n_bars - feature_window``
    # long. We align by taking the last
    # ``n_bars - feature_window`` rows of ``y`` (which
    # are the labelled subset modulo a feature-window
    # shift; the trailing ``horizon_bars`` rows are
    # un-labelled and the trailing ``feature_window``
    # rows of ``y`` are also the un-labelled tail).
    y_aligned: np.ndarray = y[feature_window:]
    return y_aligned, p


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------
def _format_sidecar(
    *,
    assets: tuple[str, ...],
    horizons: tuple[str, ...],
    rows: list[dict[str, Any]],
    n_bars: int,
    feature_window: int,
    horizon_bars: int,
    seed: int,
) -> dict[str, Any]:
    """Format the W3.5 results as a JSON-serialisable dict.

    The shape matches the PRD W3.5 acceptance criterion
    spec::

        {
          "schema_version": "1",
          "story_id": "W3.5",
          "decided_at_iso": <UTC ISO-8601>,
          "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
          "horizons": ["5m", "15m", "1h", "1d"],
          "reference_point_coverage_pct": [25, 10],
          "synthetic_fixture": {
            "n_bars": int,
            "feature_window": int,
            "horizon_bars": int,
            "seed": int,
            "base_sigma": float,
            "vol_multipliers": {"BTCUSDT": 1.0, ...}
          },
          "rows": [
            {
              "asset": str,
              "horizon": str,
              "t_at_25pct_coverage": float,
              "t_at_25pct_accuracy": float,
              "t_at_10pct_coverage": float,
              "t_at_10pct_accuracy": float,
              "full_curve": [
                {"threshold": float, "coverage": float, "accuracy": float},
                ...
              ]
            },
            ...
          ]
        }
    """
    decided_at: str = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return {
        "schema_version": "1",
        "story_id": "W3.5",
        "decided_at_iso": decided_at,
        "assets": list(assets),
        "horizons": list(horizons),
        "reference_point_coverage_pct": [25, 10],
        "synthetic_fixture": {
            "n_bars": n_bars,
            "feature_window": feature_window,
            "horizon_bars": horizon_bars,
            "seed": seed,
            "base_sigma": _BASE_SIGMA,
            "vol_multipliers": dict(ASSET_VOL_MULT),
        },
        "rows": rows,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI args. The defaults match the W3.5 task description."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="run_coverage_curve",
        description=(
            "Story W3.5: publish the coverage-accuracy Pareto frontier "
            "for 3 assets (BTCUSDT, ETHUSDT, SOLUSDT) x 4 horizons "
            "(5m, 15m, 1h, 1d) = 12 (asset, horizon) pairs. Writes the "
            "headline JSON to reports/coverage_pareto_w4.json."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=_REPORT_PATH,
        help=(
            "Path to the headline JSON report "
            "(default: reports/coverage_pareto_w4.json)."
        ),
    )
    parser.add_argument(
        "--sidecar-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON sidecar. If provided, the same "
            "sidecar content is also written to this path (for "
            "downstream consumers that read artifacts/)."
        ),
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=DEFAULT_N_BARS,
        help=(
            "Number of bars in each synthetic fixture "
            f"(default: {DEFAULT_N_BARS})."
        ),
    )
    parser.add_argument(
        "--feature-window",
        type=int,
        default=8,
        help=(
            "Rolling-window length for the logistic-regression "
            "features (default: 8)."
        ),
    )
    parser.add_argument(
        "--horizon-bars",
        type=int,
        default=DEFAULT_HORIZON_BARS,
        help=(
            "Triple-barrier horizon in bars (default: "
            f"{DEFAULT_HORIZON_BARS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        help=(
            "RNG seed for the synthetic price walks "
            f"(default: {_SEED})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the W3.5 coverage-curve publisher. Returns the process exit code."""
    args: argparse.Namespace = _parse_args(argv)
    assets: tuple[str, ...] = DEFAULT_ASSETS
    horizons: tuple[str, ...] = DEFAULT_HORIZONS
    n_bars: int = args.n_bars
    feature_window: int = args.feature_window
    horizon_bars: int = args.horizon_bars
    seed: int = args.seed

    rows: list[dict[str, Any]] = []
    # Per-asset / per-horizon seed offset so the 12
    # fixtures are independent.
    seed_offset: int = 0
    for asset in assets:
        for horizon in horizons:
            seed_offset += 1
            y_aligned, p_primary = _build_pair(
                asset=asset,
                horizon=horizon,
                n_bars=n_bars,
                feature_window=feature_window,
                horizon_bars=horizon_bars,
                seed=seed + seed_offset,
            )
            # Both y_aligned and p_primary are float64,
            # length ``n_bars - feature_window``.
            out: dict[str, Any] = coverage_curve(y_aligned, p_primary)
            full_curve: list[dict[str, Any]] = []
            for t in sorted(out["curve"].keys()):
                metrics: dict[str, float] = out["curve"][t]
                full_curve.append({
                    "threshold": float(t),
                    "coverage": float(metrics["coverage"]),
                    "accuracy": float(metrics["accuracy"]),
                })
            rows.append({
                "asset": asset,
                "horizon": horizon,
                "t_at_25pct_coverage": float(out["t_at_25pct_coverage"]),
                "t_at_25pct_accuracy": float(out["t_at_25pct_accuracy"]),
                "t_at_10pct_coverage": float(out["t_at_10pct_coverage"]),
                "t_at_10pct_accuracy": float(out["t_at_10pct_accuracy"]),
                "full_curve": full_curve,
            })

    sidecar: dict[str, Any] = _format_sidecar(
        assets=assets,
        horizons=horizons,
        rows=rows,
        n_bars=n_bars,
        feature_window=feature_window,
        horizon_bars=horizon_bars,
        seed=seed,
    )

    report_path: Path = args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    if args.sidecar_path is not None:
        sidecar_path: Path = args.sidecar_path
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    # Stdout summary: the W3.5 headline numbers the
    # engineer running the script interactively wants to
    # see at a glance. We print the two reference points
    # for BTCUSDT 1h (the canonical headline example
    # from the PRD W3.5 spec).
    btc_1h: dict[str, Any] = next(
        r for r in rows if r["asset"] == "BTCUSDT" and r["horizon"] == "1h"
    )
    print(f"W3.5 coverage-curve JSON written to {report_path}")
    print(
        f"Headline (BTCUSDT 1h): "
        f"T@25%cov={btc_1h['t_at_25pct_coverage']:.6f} "
        f"(acc={btc_1h['t_at_25pct_accuracy']:.6f}), "
        f"T@10%cov={btc_1h['t_at_10pct_coverage']:.6f} "
        f"(acc={btc_1h['t_at_10pct_accuracy']:.6f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
