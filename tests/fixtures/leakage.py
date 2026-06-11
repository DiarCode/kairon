"""Bar-level leakage fixtures for downstream test authors.

This module provides *bar-level* (single-table) leakage helpers. It is
the OHLCV-table counterpart to the splits-layer leakage tests in
:mod:`tests.splits.test_leakage_invariants`. The two layers cover
different things and intentionally do not duplicate each other:

* The splits-layer invariants check that *fold definitions* (walk-forward
  splits, purge windows, embargo windows) do not let a test fold's label
  window overlap a train fold's features. They operate on index ranges.

* The bar-level helpers in this module check that a *single* OHLCV
  :class:`pyarrow.Table` is well-formed enough to realise labels with a
  given label horizon, and that its timestamps are chronologically sane
  (no future timestamps, no backwards time travel). They operate on a
  pyarrow table directly and are layout-neutral (they do not care
  whether the table was written via the W1.5 hive-style partitioned
  writer, the canonical ``write_ohlcv`` writer, or assembled in
  memory by a test).

Downstream tests should import from this module rather than
re-implementing the helpers::

    from tests.fixtures.leakage import (
        real_history_fixture,
        assert_no_leakage,
        assert_timestamp_monotonic,
    )

All public functions are fully type-annotated and conform to
``OHLCV_SCHEMA`` from :mod:`kairon.data.io`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA

# Small slack used by ``assert_timestamp_monotonic`` when checking the
# upper bound on the ``ts`` column. A backtest re-replay of a bar that
# was captured moments ago should not be flagged as a future
# timestamp. Five minutes is generous enough to cover slow CI
# runners and clock skew on commodity hardware, but small enough that
# an obviously-wrong fixture (e.g. one with year 2099 timestamps) is
# still flagged.
_FUTURE_SLACK: timedelta = timedelta(minutes=5)

# The default bar cadence for ``real_history_fixture``. Exposed as a
# module-level constant so downstream tests can refer to it without
# hard-coding ``60`` (one minute).
DEFAULT_EVERY_SECONDS: int = 60

# The default fixture length: 1440 bars at 60s cadence = 1 day of
# 1-minute BTC bars.
DEFAULT_N_BARS: int = 1440


def real_history_fixture(
    *,
    n_bars: int = DEFAULT_N_BARS,
    start: datetime | None = None,
    every_seconds: int = DEFAULT_EVERY_SECONDS,
    base_price: float = 50_000.0,
    sigma: float = 0.01,
    seed: int = 42,
) -> pa.Table:
    """Return a synthetic but realistic BTC-like 1-minute OHLCV table.

    The fixture is hermetic (no network, no real data) and deterministic
    for a given ``seed``: two calls with the same arguments return tables
    with identical ``open``/``high``/``low``/``close``/``volume`` values
    and the same ``ts`` sequence.

    The price walk is a per-bar log-normal step: ``log_p[i+1] =
    log_p[i] + Normal(0, sigma)`` with ``sigma`` controlling the
    per-step volatility. Open is the previous close (with the first
    row's open equal to ``base_price``); high/low are open/close
    bracketed by a small within-bar range; volume is a positive
    pseudo-random quantity.

    Parameters
    ----------
    n_bars:
        Number of bars to generate. Default ``1440`` = 1 day of 1m
        bars.
    start:
        UTC datetime for the first bar's ``ts``. If ``None`` (the
        default), the fixture starts 1 day before "now" so the entire
        fixture lies in the past (no future timestamps).
    every_seconds:
        Bar cadence in seconds. Default ``60`` (1-minute bars).
    base_price:
        Price level for the very first bar's open. Default
        ``50_000.0`` (BTC-like).
    sigma:
        Per-step log-price volatility. Default ``0.01``.
    seed:
        Seed for the internal :class:`numpy.random.Generator`. Two
        calls with the same ``seed`` and same other parameters
        produce identical tables.

    Returns
    -------
    pyarrow.Table
        A table that exactly matches :data:`kairon.data.io.OHLCV_SCHEMA`.
    """
    if n_bars <= 0:
        raise ValueError(f"n_bars must be > 0, got {n_bars}")
    if every_seconds <= 0:
        raise ValueError(f"every_seconds must be > 0, got {every_seconds}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    rng = np.random.default_rng(seed)
    # log_price[i+1] = log_price[i] + N(0, sigma)
    log_returns = rng.normal(loc=0.0, scale=sigma, size=n_bars)
    # log_price[0] = log(base_price); log_price[i+1] uses the
    # cumulative sum of returns.
    log_prices = np.empty(n_bars, dtype=np.float64)
    log_prices[0] = np.log(base_price)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns[:-1])
    closes = np.exp(log_prices)
    # Open[i] = Close[i-1]; Open[0] = base_price.
    opens = np.empty(n_bars, dtype=np.float64)
    opens[0] = base_price
    opens[1:] = closes[:-1]
    # Small within-bar range: high = max(open, close) + |N(0, 0.0005)|;
    # low = min(open, close) - |N(0, 0.0005)|. The half-width is
    # tiny relative to sigma so the high/low/close relationship
    # holds (high >= max(open, close), low <= min(open, close)).
    half_width = np.abs(rng.normal(loc=0.0, scale=0.0005, size=n_bars))
    highs = np.maximum(opens, closes) + half_width
    lows = np.minimum(opens, closes) - half_width
    # Positive volume: 1.0 + |N(0, 0.5)|, clipped to a positive
    # minimum so volume is always strictly > 0 (a degenerate zero
    # volume bar would be unrealistic for BTC 1m).
    volumes = 1.0 + np.abs(rng.normal(loc=0.0, scale=0.5, size=n_bars))

    if start is None:
        # Place the fixture firmly in the past so downstream
        # ``assert_timestamp_monotonic`` (no future timestamps) is
        # not flaky on slow CI: end 1 day ago, start 2 days ago.
        now_utc = datetime.now(UTC).replace(microsecond=0)
        end = now_utc - timedelta(days=1)
        start_dt: datetime = end - timedelta(seconds=every_seconds * (n_bars - 1))
    else:
        if start.tzinfo is None:
            raise ValueError("start must be timezone-aware (UTC)")
        start_dt = start.astimezone(UTC)

    step = timedelta(seconds=every_seconds)
    ts_list = [start_dt + step * i for i in range(n_bars)]

    return pa.table(
        {
            "ts": ts_list,
            "open": opens.tolist(),
            "high": highs.tolist(),
            "low": lows.tolist(),
            "close": closes.tolist(),
            "volume": volumes.tolist(),
        },
        schema=OHLCV_SCHEMA,
    )


def assert_no_leakage(
    table: pa.Table,
    *,
    label_horizon_seconds: int,
) -> None:
    """Assert that a label with the given horizon can be fully realised.

    The check is structural rather than semantic: it verifies that the
    table's total time span (``max(ts) - min(ts)``) is at least
    ``label_horizon_seconds``, *and* that the table's schema is the
    canonical :data:`OHLCV_SCHEMA`. This is the bar-level "no future
    leakage" guard: a clean fixture whose span covers the label
    window passes, and a short fixture (whose span is smaller than
    the label horizon) fails with a clear message.

    The check is::

        (max(ts) - min(ts)).total_seconds() >= label_horizon_seconds

    Concretely: a 10-bar 1-minute fixture has a 9-minute span, so it
    can realise a 60s (1-minute) label but not a 1-day (86400s) label.
    A 1440-bar 1-minute fixture (the default 1-day fixture) has a
    ~1439-minute span and can realise labels up to ~1 day wide.

    Parameters
    ----------
    table:
        The OHLCV pyarrow table to validate. Must conform to
        :data:`OHLCV_SCHEMA`.
    label_horizon_seconds:
        Width of the label window in seconds. A common value is
        ``60`` (1-minute label) or ``86400`` (1-day label).

    Raises
    ------
    AssertionError
        If the table's schema is wrong, the table is empty, the
        label horizon is not strictly positive, or the table's
        total time span is smaller than the label horizon.
    """
    if label_horizon_seconds <= 0:
        raise AssertionError(
            f"label_horizon_seconds must be > 0, got {label_horizon_seconds}"
        )
    if table.schema != OHLCV_SCHEMA:
        raise AssertionError(
            f"schema mismatch: expected {OHLCV_SCHEMA}, got {table.schema}"
        )
    n_bars = len(table)
    if n_bars == 0:
        raise AssertionError(
            "table is empty; cannot realise any label horizon"
        )
    # Compute the table's total time span from the first and last
    # bars. This is layout-neutral (works for 1m, 5m, daily, ...)
    # and is the *correct* check: the label window must fit inside
    # the table's actual span, regardless of bar cadence.
    ts_values: list[datetime] = table.column("ts").to_pylist()
    first_ts = ts_values[0]
    last_ts = ts_values[-1]
    span_seconds = (last_ts - first_ts).total_seconds()
    if span_seconds < label_horizon_seconds:
        raise AssertionError(
            f"future-leakage detected: table has {n_bars} bars with "
            f"a total span of {span_seconds:.0f}s, but "
            f"label_horizon_seconds={label_horizon_seconds} requires "
            f"a span of at least {label_horizon_seconds}s. The label "
            f"window extends past the table's last bar."
        )


def assert_timestamp_monotonic(
    table: pa.Table,
    *,
    future_slack: timedelta = _FUTURE_SLACK,
) -> None:
    """Assert that the ``ts`` column is well-formed and sane.

    Two checks are performed:

    1. The ``ts`` column is *strictly non-decreasing* (each bar's
       timestamp is at or after the previous bar's timestamp). This
       is the basic ordering invariant: a backtest cannot run over a
       table whose bars arrive out of order in time.
    2. The table contains *no future timestamps*: ``max(ts) <=
       datetime.now(UTC) + future_slack``. The small ``future_slack``
       accounts for backtest reproduction of data captured moments
       ago. A clearly-broken fixture (e.g. one with year 2099
       timestamps, or one with all timestamps in the future) is
       flagged.

    Parameters
    ----------
    table:
        The OHLCV pyarrow table to validate. Must conform to
        :data:`OHLCV_SCHEMA`.
    future_slack:
        How far into the future, relative to "now", a bar's
        timestamp is allowed to be. Default 5 minutes. Pass a
        tighter ``timedelta(0)`` for strict no-future checks.

    Raises
    ------
    AssertionError
        If the schema is wrong, the table is empty, any ``ts`` value
        is strictly before the previous ``ts``, or any ``ts`` value
        is more than ``future_slack`` past "now".
    """
    if table.schema != OHLCV_SCHEMA:
        raise AssertionError(
            f"schema mismatch: expected {OHLCV_SCHEMA}, got {table.schema}"
        )
    n_bars = len(table)
    if n_bars == 0:
        raise AssertionError("table is empty; cannot validate timestamps")
    ts_values: list[datetime] = table.column("ts").to_pylist()
    # 1. Monotonicity: each ts[i] >= ts[i-1]. Pyright can see that
    # the schema binds ``ts`` to pa.timestamp('us', tz='UTC') so
    # ``to_pylist()`` returns ``list[datetime]`` here — no runtime
    # ``isinstance`` guard is needed (and would be flagged as
    # redundant by pyright).
    for i in range(1, n_bars):
        prev = ts_values[i - 1]
        cur = ts_values[i]
        if cur < prev:
            raise AssertionError(
                f"ts column is not non-decreasing: ts[{i}]={cur!r} < "
                f"ts[{i - 1}]={prev!r}"
            )
    # 2. No future timestamps: max(ts) <= now(UTC) + future_slack.
    # Compute max via Python (n_bars is small for fixtures; n_bars is
    # bounded by the test's call to real_history_fixture, which is
    # typically <= 1440). The first row's tzinfo is the source of
    # truth: if the fixture was built with a tz-aware start, every
    # row is tz-aware.
    first = ts_values[0]
    if first.tzinfo is None:
        raise AssertionError(
            "ts column is timezone-naive; OHLCV_SCHEMA requires "
            "timezone-aware (UTC) timestamps"
        )
    last = ts_values[-1]
    now_utc = datetime.now(UTC)
    last_utc = last.astimezone(UTC) if last.tzinfo is not None else last
    upper_bound = now_utc + future_slack
    if last_utc > upper_bound:
        raise AssertionError(
            f"future timestamp detected: max(ts)={last_utc.isoformat()} "
            f"is after now(UTC) + future_slack={upper_bound.isoformat()}"
        )
