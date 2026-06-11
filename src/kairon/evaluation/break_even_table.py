"""Build the per-(asset, horizon) break-even accuracy table.

Story W2.2 publishes a ``(asset, horizon)`` cross product of
break-even accuracies ``p* = 0.5 + C / (2R)`` for the 3 W0
assets (BTCUSDT, ETHUSDT, SOLUSDT) and the 4 horizons
(5m, 15m, 1h, 1d). The result is a :class:`pyarrow.Table`
that the W2.2 runner script (``scripts/run_break_even.py``)
materialises into the ``reports/break_even_w2.md`` headline
artifact (the markdown format the W2.5 GO/NO-GO gate reads)
and the ``artifacts/break_even_w2.json`` sidecar for
downstream consumers.

The table is intentionally a small in-memory builder: it
takes a pre-computed ``realized_sigma`` mapping (so the
caller can swap in real or synthetic per-bar volatility),
applies the conservative annualized-to-horizon rescaling
``expected_move = sigma * sqrt(252 * 24 * 60 / seconds_per_bar)``,
and emits the break-even accuracy via
:func:`kairon.evaluation.break_even.break_even_accuracy`.

The "viable" column is the load-bearing downstream contract:
``True`` when ``break_even_pct <= 0.60`` (60% accuracy, the
plan's W2.5 viability threshold). The W2.5 gate does its own
PROCEED/ESCALATE/HALT decision on the MAX of
``break_even_pct`` across all rows; ``viable`` here is the
per-row "is this single (asset, horizon) trade theoretically
viable" flag.

The module is pure: no IO, no async, no global state. The
caller (the runner script) owns the markdown / JSON
serialisation and the file writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from kairon.evaluation.break_even import break_even_accuracy

if TYPE_CHECKING:
    from kairon.backtest.cost import CostModel


# Viability threshold for the "viable" column. Matches the
# plan's W2.5 reference: a per-trade break-even accuracy of
# 60% is the maximum tolerable for the model to be worth
# shipping. Above 60% the trade's edge is too thin to clear
# fees with statistical confidence; below 60% the trade has
# enough cushion to be expectation-positive.
DEFAULT_VIABLE_THRESHOLD: float = 0.60

# Default asset universe (per W0 BTC-only fallback: the 3
# assets are BTCUSDT, ETHUSDT, SOLUSDT, not the original
# {BTC,ETH,SPY} mix the plan suggested before W0.1/W0.3 were
# de-scoped).
DEFAULT_ASSETS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

# Default horizon set. The seconds-per-bar mapping is the
# canonical market-data convention; 1d = 86400s = 24*3600.
DEFAULT_HORIZONS: tuple[str, ...] = ("5m", "15m", "1h", "1d")
SECONDS_PER_BAR: dict[str, int] = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
}

# Annualization factor for the conservative "expected move"
# conversion. Crypto trades 24/7/365, so 1 year = 365 days =
# 365 * 24 * 60 minutes. The conversion is the standard
# ``sigma * sqrt(annualization_factor / seconds_per_bar)``
# rescale from a per-bar return std-dev to an "expected
# absolute move over the horizon" (a one-sigma move in bps).
# Using a one-sigma upper bound is conservative: a real
# E[|r|] is the half-normal mean ``sigma * sqrt(2/pi)``,
# roughly ``0.8 * sigma``, so the table errs on the side
# of MORE expected move (i.e. MORE optimistic) — that is
# the plan's stated "conservative" direction because
# under-stating R inflates p* and makes unviable trades
# look viable. We deliberately over-state R to make the
# break-even *harder* to clear.
CRYPTO_BARS_PER_YEAR: int = 365 * 24 * 60  # 525_600 minutes


def expected_move_bps(
    *,
    realized_sigma: float,
    horizon: str,
) -> float:
    """Convert a per-bar return std-dev into an expected move in bps.

    Implements the conservative annualized-to-horizon rescale

        expected_move_bps = realized_sigma
                          * sqrt(CRYPTO_BARS_PER_YEAR / SECONDS_PER_BAR[horizon])
                          * 10_000

    The factor of ``10_000`` converts the dimensionless
    return std-dev to basis points. The result is the
    one-sigma move in bps over the horizon; using the
    one-sigma upper bound is the plan's stated
    "conservative" direction because under-stating R would
    inflate p* and make unviable trades look viable.

    Raises
    ------
    ValueError
        If ``horizon`` is not in :data:`SECONDS_PER_BAR`, or
        if ``realized_sigma`` is negative or non-finite.
    """
    import math

    if horizon not in SECONDS_PER_BAR:
        raise ValueError(
            f"horizon must be one of {tuple(SECONDS_PER_BAR.keys())}, "
            f"got {horizon!r}"
        )
    if not math.isfinite(realized_sigma):
        raise ValueError(
            f"realized_sigma must be finite, got {realized_sigma!r}"
        )
    if realized_sigma < 0:
        raise ValueError(
            f"realized_sigma must be >= 0, got {realized_sigma!r}"
        )
    seconds_per_bar: int = SECONDS_PER_BAR[horizon]
    annualization_factor: float = float(CRYPTO_BARS_PER_YEAR) / float(seconds_per_bar)
    return realized_sigma * math.sqrt(annualization_factor) * 10_000.0


def build_break_even_table(
    *,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    horizons: tuple[str, ...] = DEFAULT_HORIZONS,
    cost_models: dict[str, "CostModel"] | None = None,
    realized_sigma: dict[tuple[str, str], float],
    viable_threshold: float = DEFAULT_VIABLE_THRESHOLD,
) -> pa.Table:
    """Build the (asset, horizon) x metrics break-even table.

    Parameters
    ----------
    assets
        The asset universe (default: BTCUSDT, ETHUSDT, SOLUSDT).
        Each asset must have a corresponding ``CostModel`` in
        ``cost_models`` and a ``realized_sigma`` entry for every
        horizon; missing entries raise ``ValueError``.
    horizons
        The horizon set (default: 5m, 15m, 1h, 1d). Each
        horizon must be a key in :data:`SECONDS_PER_BAR` and
        must have a corresponding ``realized_sigma`` entry
        for every asset.
    cost_models
        Per-asset :class:`CostModel` map. When ``None`` the
        caller MUST still pass a non-empty dict — we do not
        fall back to a global default here because the
        per-asset cost asymmetry (e.g. ETH being cheaper
        to trade than BTC) is a load-bearing input to the
        break-even calculation. The dict is typed as
        ``dict[str, CostModel]`` for the strict-typing gate.
    realized_sigma
        ``(asset, horizon) -> per-bar return std-dev`` map.
        The keys are ``(asset, horizon)`` tuples (NOT a
        nested dict) so the caller's data shape is
        explicit and the table builder is trivial to
        exercise in tests.
    viable_threshold
        The break-even accuracy threshold below which a row
        is marked ``viable=True``. Defaults to 0.60 (the
        plan's W2.5 reference). Must be in (0.5, 1.0].

    Returns
    -------
    pyarrow.Table
        A 12-row table (3 assets x 4 horizons) with columns
        ``{asset: str, horizon: str, expected_move_bps: float,
        round_trip_cost_bps: float, break_even_pct: float,
        viable: bool}``. The schema is stable so the runner
        script can serialise the table to markdown and JSON
        without column-name surprises.

    Raises
    ------
    ValueError
        If an asset is not in ``cost_models``, if a
        ``(asset, horizon)`` key is missing from
        ``realized_sigma``, if a horizon is not in
        :data:`SECONDS_PER_BAR`, if ``viable_threshold`` is
        outside (0.5, 1.0], or if ``cost_models`` is ``None``.
    """
    import math

    if cost_models is None:
        # We deliberately do NOT fall back to a global default
        # here — the per-asset cost asymmetry is a load-bearing
        # input and silently collapsing to DEFAULT_CRYPTO_COSTS
        # would mask configuration bugs (e.g. forgetting to wire
        # the ETH/SOL cost models). The runner script is the
        # right place to do the default fallback.
        raise ValueError(
            "cost_models must be a non-empty dict mapping asset to "
            "CostModel; pass an explicit cost_models arg to suppress "
            "this error. (The runner script falls back to "
            "DEFAULT_CRYPTO_COSTS for missing assets before calling "
            "this builder.)"
        )
    if not cost_models:
        raise ValueError(
            "cost_models must be a non-empty dict mapping asset to "
            "CostModel; got an empty dict"
        )
    if not (0.5 < viable_threshold <= 1.0):
        raise ValueError(
            f"viable_threshold must be in (0.5, 1.0], got {viable_threshold!r}"
        )
    for horizon in horizons:
        if horizon not in SECONDS_PER_BAR:
            raise ValueError(
                f"horizon must be one of {tuple(SECONDS_PER_BAR.keys())}, "
                f"got {horizon!r}"
            )
    unknown_assets: set[str] = set(assets) - set(cost_models.keys())
    if unknown_assets:
        raise ValueError(
            f"assets not in cost_models: {sorted(unknown_assets)}. "
            f"cost_models has keys: {sorted(cost_models.keys())}"
        )
    missing_sigma: list[tuple[str, str]] = [
        (a, h) for a in assets for h in horizons if (a, h) not in realized_sigma
    ]
    if missing_sigma:
        raise ValueError(
            f"realized_sigma is missing entries for: {missing_sigma}. "
            f"Provide a per-bar return std-dev for every (asset, horizon) "
            f"pair so the annualized-to-horizon conversion is well-defined."
        )

    asset_list: list[str] = []
    horizon_list: list[str] = []
    expected_move_list: list[float] = []
    round_trip_cost_list: list[float] = []
    break_even_pct_list: list[float] = []
    viable_list: list[bool] = []

    for asset in assets:
        cost_model: "CostModel" = cost_models[asset]
        round_trip_cost: float = cost_model.round_trip_bps
        # Guard against a non-finite round-trip cost (e.g. a
        # CostModel with NaN half_spread). The table is a
        # downstream-of-the-gate artefact and a NaN here would
        # silently poison the W2.5 PROCEED/ESCALATE/HALT
        # decision.
        if not math.isfinite(round_trip_cost):
            raise ValueError(
                f"cost_models[{asset!r}].round_trip_bps is non-finite: "
                f"{round_trip_cost!r}"
            )
        if round_trip_cost < 0:
            raise ValueError(
                f"cost_models[{asset!r}].round_trip_bps is negative: "
                f"{round_trip_cost!r}"
            )
        for horizon in horizons:
            sigma: float = realized_sigma[(asset, horizon)]
            em_bps: float = expected_move_bps(
                realized_sigma=sigma, horizon=horizon,
            )
            p_star: float = break_even_accuracy(
                expected_move_bps=em_bps,
                round_trip_cost_bps=round_trip_cost,
            )
            asset_list.append(asset)
            horizon_list.append(horizon)
            expected_move_list.append(em_bps)
            round_trip_cost_list.append(round_trip_cost)
            break_even_pct_list.append(p_star)
            viable_list.append(p_star <= viable_threshold)

    table: pa.Table = pa.table(
        {
            "asset": pa.array(asset_list, type=pa.string()),
            "horizon": pa.array(horizon_list, type=pa.string()),
            "expected_move_bps": pa.array(expected_move_list, type=pa.float64()),
            "round_trip_cost_bps": pa.array(round_trip_cost_list, type=pa.float64()),
            "break_even_pct": pa.array(break_even_pct_list, type=pa.float64()),
            "viable": pa.array(viable_list, type=pa.bool_()),
        },
        schema=pa.schema(
            [
                pa.field("asset", pa.string(), nullable=False),
                pa.field("horizon", pa.string(), nullable=False),
                pa.field("expected_move_bps", pa.float64(), nullable=False),
                pa.field("round_trip_cost_bps", pa.float64(), nullable=False),
                pa.field("break_even_pct", pa.float64(), nullable=False),
                pa.field("viable", pa.bool_(), nullable=False),
            ]
        ),
    )
    return table


__all__ = [
    "CRYPTO_BARS_PER_YEAR",
    "DEFAULT_ASSETS",
    "DEFAULT_HORIZONS",
    "DEFAULT_VIABLE_THRESHOLD",
    "SECONDS_PER_BAR",
    "build_break_even_table",
    "expected_move_bps",
]
