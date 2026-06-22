"""Drift kill-switch — halt trading when live performance drifts off-edge.

Phase 3 of the scalping-edge-enhancement plan. The setup-selection matrix
(``MEAN_REVERSION_ONLY``) was data-discovered on the 8-week testnet store, so
overfitting is the dominant risk once it runs on *fresh* live bars. This monitor
is the out-of-sample guardrail: it keeps a rolling window of realized trade
outcomes and halts the loop when live win-rate / expectancy falls below the
threshold that the research edge implies.

This is a *performance* kill-switch, deliberately distinct from
:mod:`kairon.live.drift` (which does feature-distribution PSI/KS drift on
indicator values). That module answers "has the input distribution shifted?";
this one answers "has the *edge* stopped paying on live bars?".

It is simple and conservative:

* A **global** rolling window over the last ``window`` closed trades. If the
  live win-rate drops below ``min_win_rate`` or the per-trade expectancy (in
  bankroll-fraction terms) drops below ``min_expectancy``, the loop should halt.
* A **per-setup** rolling window (when ``per_setup=True``): each setup_id is
  tracked independently, so a single setup bleeding (e.g. mean-reversion
  misfiring in a trend the regime gate missed) trips the switch even when the
  blended global window still looks acceptable.

The monitor only *reports*; the orchestrator owns the actual halt side-effects
(``store.halt``, ledger row, ``_running = False``). It is opt-in — ``None``
(default) preserves the legacy behaviour with no drift kill-switch.

Thresholds are in plain units: win-rate as a fraction in [0, 1], expectancy as
realized-PnL per trade in **bankroll-fraction** terms (the caller passes
``realized_pnl / bankroll_at_close`` so the threshold is bankroll-invariant
across compounding).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

__all__ = ["DriftKillSwitch", "DriftKillSwitchConfig", "DriftVerdict"]


@dataclass(frozen=True, slots=True)
class DriftKillSwitchConfig:
    """Thresholds for the drift kill-switch.

    Defaults encode the research edge: the MEAN_REVERSION_ONLY matrix lifted the
    best symbol/timeframe to 62-68% win rate, so a live floor of 40% with a
    minimum sample of 10 trades is a generous "edge has clearly evaporated"
    trigger (well below the in-sample edge, far above random). The expectancy
    floor is -0.5% bankroll per trade — allow a run of bad luck, halt only on a
    structural break.
    """

    window: int = 20
    min_trades: int = 10
    min_win_rate: float = 0.40
    min_expectancy: float = -0.005  # bankroll-fraction per trade
    per_setup: bool = True
    per_setup_window: int = 12
    per_setup_min_trades: int = 6
    per_setup_min_win_rate: float = 0.30
    per_setup_min_expectancy: float = -0.005  # bankroll-fraction per trade


@dataclass(frozen=True, slots=True)
class DriftVerdict:
    """Result of a drift check — whether to halt and why."""

    halt: bool
    reason: str | None = None
    win_rate: float = 0.0
    expectancy: float = 0.0
    setup_id: str | None = None


@dataclass(slots=True)
class DriftKillSwitch:
    """Rolling-window live-performance monitor (opt-in kill-switch).

    Call :meth:`record` after every closed trade, then :meth:`check`. When
    :meth:`check` returns a verdict with ``halt=True``, the orchestrator halts
    the loop. The orchestrator drives the loop from one async task, so no
    internal locking is needed.
    """

    config: DriftKillSwitchConfig = field(default_factory=DriftKillSwitchConfig)
    _pnls: deque[float] = field(default_factory=deque)
    _by_setup: dict[str, deque[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The deque maxlen must honour ``config.window``; the default_factory
        # can't read the instance config, so rebind here.
        self._pnls = deque(maxlen=self.config.window)

    def record(self, realized_pnl_fraction: float, setup_id: str | None = None) -> None:
        """Record one closed trade's realized PnL as a bankroll fraction.

        ``realized_pnl_fraction`` should be ``realized_pnl / bankroll_at_close``
        so the window is comparable across a compounding bankroll. ``setup_id``
        is optional; when present and ``per_setup`` is enabled, the trade also
        feeds that setup's independent window.

        Non-finite values (NaN/inf from a bad fill or a near-depleted bankroll)
        are coerced to a full-loss ``-1.0``. A bare ``NaN`` would make every
        ``win_rate < floor`` and ``expectancy < floor`` comparison ``False``,
        silently neutralising the guard for the whole window — coercing instead
        keeps the monitor decisive and biases toward halting on the bad fill.
        """
        if not math.isfinite(realized_pnl_fraction):
            realized_pnl_fraction = -1.0
        self._pnls.append(realized_pnl_fraction)
        if self.config.per_setup and setup_id:
            buf = self._by_setup.get(setup_id)
            if buf is None:
                buf = deque(maxlen=self.config.per_setup_window)
                self._by_setup[setup_id] = buf
            buf.append(realized_pnl_fraction)

    @staticmethod
    def _window_stats(pnls: deque[float]) -> tuple[int, float, float]:
        n = len(pnls)
        if n == 0:
            return 0, 0.0, 0.0
        wins = sum(1 for p in pnls if p > 0.0)
        win_rate = wins / n
        expectancy = sum(pnls) / n
        return n, win_rate, expectancy

    def check(self) -> DriftVerdict:
        """Evaluate the rolling windows; return a halt verdict."""
        n, win_rate, expectancy = self._window_stats(self._pnls)
        if n >= self.config.min_trades:
            if win_rate < self.config.min_win_rate:
                return DriftVerdict(
                    halt=True,
                    reason=(
                        f"drift: global win-rate {win_rate:.1%} < "
                        f"{self.config.min_win_rate:.1%} over {n} trades"
                    ),
                    win_rate=win_rate,
                    expectancy=expectancy,
                )
            if expectancy < self.config.min_expectancy:
                return DriftVerdict(
                    halt=True,
                    reason=(
                        f"drift: global expectancy {expectancy:+.4f} < "
                        f"{self.config.min_expectancy:+.4f} over {n} trades"
                    ),
                    win_rate=win_rate,
                    expectancy=expectancy,
                )

        if self.config.per_setup:
            for setup_id, buf in self._by_setup.items():
                sn, swin, sexp = self._window_stats(buf)
                if sn >= self.config.per_setup_min_trades:
                    if swin < self.config.per_setup_min_win_rate:
                        return DriftVerdict(
                            halt=True,
                            reason=(
                                f"drift: setup {setup_id} win-rate {swin:.1%} < "
                                f"{self.config.per_setup_min_win_rate:.1%} over {sn} trades"
                            ),
                            win_rate=swin,
                            expectancy=sexp,
                            setup_id=setup_id,
                        )
                    if sexp < self.config.per_setup_min_expectancy:
                        return DriftVerdict(
                            halt=True,
                            reason=(
                                f"drift: setup {setup_id} expectancy {sexp:+.4f} < "
                                f"{self.config.per_setup_min_expectancy:+.4f} over {sn} trades"
                            ),
                            win_rate=swin,
                            expectancy=sexp,
                            setup_id=setup_id,
                        )
        return DriftVerdict(halt=False, win_rate=win_rate, expectancy=expectancy)
