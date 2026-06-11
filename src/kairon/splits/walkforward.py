"""Walk-forward splits.

The default backtest harness: given a list of bar timestamps, build
``(train, val, test)`` triples in *strictly chronological* order. Each
test fold is the next time window; the training window is the
preceding ``train_size`` bars; the validation window is the preceding
``val_size`` bars (or zero).

Two modes:

- **Rolling** (``anchored=False``): the training window slides forward
  so its end is always the bar just before the validation window.
- **Anchored** (``anchored=True``): the training window always starts
  at the first bar; its end slides forward.

Default sizes (overridable per call):

- 5-min crypto: train=7d (2016 bars), val=1d (288), test=1d (288)
- 1-hour:      train=90d (~2160), val=14d (~336), test=14d (~336)
- 1-day:       train=4y (~1000), val=6m (~126), test=6m (~126)

These are *not* the universal right answer; they're a sane default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """A typed walk-forward split spec."""

    train_size: int
    val_size: int
    test_size: int
    anchored: bool = False
    purge_bars: int = 0
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        if self.train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {self.train_size}")
        if self.test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {self.test_size}")
        if self.val_size < 0:
            raise ValueError(f"val_size must be >= 0, got {self.val_size}")
        if self.purge_bars < 0:
            raise ValueError(f"purge_bars must be >= 0, got {self.purge_bars}")
        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {self.embargo_bars}")


# Default split specs per horizon
DEFAULT_SPLIT_5M: Final[SplitSpec] = SplitSpec(train_size=2016, val_size=288, test_size=288)
DEFAULT_SPLIT_1H: Final[SplitSpec] = SplitSpec(train_size=2160, val_size=336, test_size=336)
DEFAULT_SPLIT_1D: Final[SplitSpec] = SplitSpec(train_size=1000, val_size=126, test_size=126)

DEFAULT_SPLIT_BY_HORIZON: Final[dict[str, SplitSpec]] = {
    "5m": DEFAULT_SPLIT_5M,
    "15m": DEFAULT_SPLIT_5M,
    "1h": DEFAULT_SPLIT_1H,
    "4h": DEFAULT_SPLIT_1H,
    "1d": DEFAULT_SPLIT_1D,
    "1w": DEFAULT_SPLIT_1D,
}


@dataclass(frozen=True, slots=True)
class Fold:
    """A single (train, val, test) fold.

    Indices are bar indices, not timestamps.  Use ``Fold.split(timestamps)``
    to materialize the timestamps.
    """

    fold_id: int
    train_start: int
    train_end: int  # exclusive
    val_start: int
    val_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    def __post_init__(self) -> None:
        if not (self.train_start <= self.train_end <= self.val_start <= self.val_end):
            raise ValueError(
                f"fold {self.fold_id}: train [{self.train_start},{self.train_end}) must precede "
                f"val [{self.val_start},{self.val_end})"
            )
        if not (self.val_end <= self.test_start <= self.test_end):
            raise ValueError(
                f"fold {self.fold_id}: val [{self.val_start},{self.val_end}) must precede "
                f"test [{self.test_start},{self.test_end})"
            )

    def n_train(self) -> int:
        return self.train_end - self.train_start

    def n_val(self) -> int:
        return self.val_end - self.val_start

    def n_test(self) -> int:
        return self.test_end - self.test_start

    def split(self, ts: Sequence[datetime]) -> tuple[Sequence[datetime], Sequence[datetime], Sequence[datetime]]:
        """Return (train_ts, val_ts, test_ts) timestamp slices."""
        return ts[self.train_start : self.train_end], ts[self.val_start : self.val_end], ts[self.test_start : self.test_end]


def walkforward(
    n_bars: int,
    *,
    spec: SplitSpec | None = None,
) -> list[Fold]:
    """Build a list of walk-forward folds over ``n_bars``.

    Each fold uses indices ``[train_start, train_end)`` for training,
    ``[val_start, val_end)`` for validation, and ``[test_start, test_end)``
    for testing. Each subsequent fold slides forward by ``test_size``
    bars (rolling) or by ``test_size`` (anchored — the train start
    stays at 0, the train end slides).

    The *first* fold has train_size bars preceding the validation
    window; in the rolling case the very first fold has
    ``train_start = 0, train_end = train_size`` (no warm-up) and
    subsequent folds slide the train window forward.
    """
    if spec is None:
        spec = DEFAULT_SPLIT_1D
    if n_bars < spec.train_size + spec.val_size + spec.test_size:
        raise ValueError(
            f"need at least {spec.train_size + spec.val_size + spec.test_size} bars; got {n_bars}"
        )
    folds: list[Fold] = []
    fold_id = 0
    cursor = spec.train_size  # train ends just before val starts
    if spec.anchored:
        train_start = 0
    else:
        train_start = 0
    while True:
        train_end = cursor
        val_start = train_end
        val_end = val_start + spec.val_size
        test_start = val_end
        test_end = test_start + spec.test_size
        if test_end > n_bars:
            break
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        # advance
        if spec.anchored:
            train_start = 0  # unchanged; only the end slides
        else:
            train_start = max(0, train_start + spec.test_size)
        cursor += spec.test_size
    return folds


def total_fold_coverage(spec: SplitSpec) -> timedelta:
    """How much test time a single fold covers, in ``timedelta`` terms.

    Useful for documenting the backtest.  We treat each bar as
    ``test_size`` units of the spec's primary horizon.
    """
    from kairon.data.diagnostics import timeframe_to_timedelta

    seconds = spec.test_size * timeframe_to_timedelta("5m")  # default unit is 5m
    return timedelta(seconds=seconds)
