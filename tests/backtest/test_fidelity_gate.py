"""Phase 1.3 fidelity gate — proves the backtest reproduces known live trade math.

Two layers:

1. **Sim ↔ pure-fn parity (deterministic).** A scripted strategy drives the engine
   over a controlled bar sequence; the recorded trade's net PnL must equal
   :func:`fidelity_expected_net_pnl` computed from the same inputs, and an SL-hit
   loss must equal :func:`implied_loss_fraction`. One implementation of the
   risk/cost math, shared between live and sim.

2. **Real-bar reproduction of the documented SOL min-lot-overshoot case.** The
   ralplan context recorded a live SOL short risk-sized to 0.074 but bumped to
   the 0.1 min lot, losing ~3.4% of the $10 bankroll on the SL hit (vs the 2.5%
   target) — the exact case :func:`kairon.live.pure_fns.post_rounding_guard` was
   built to catch. The gate reproduces it on real SOL 5m bars:

   * with the risk-cap guard **off** (the pre-fix live behaviour), the overshoot
     trades and an SL hit loses MORE than the 2.5% target — the documented bug;
   * with the guard **on** (Phase 0.2), those >2.75% overshoots are skipped
     (``risk_cap_breach_overshoot``), and with overshoot disabled they are
     skipped as ``below_min_lot``.

Skipped when the research parquet store is absent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from kairon.backtest.position import Side
from kairon.backtest.scalping_cost import (
    BYBIT_TESTNET_COSTS,
    fidelity_expected_net_pnl,
    implied_loss_fraction,
    round_trip_cost_bps,
)
from kairon.backtest.scalping_engine import ScalpBacktestConfig, run_scalp_backtest
from kairon.data.io import OHLCV_SCHEMA
from kairon.live.predictor import LivePrediction
from kairon.live.pure_fns import implied_risk, risk_size_qty


# ---------------------------------------------------------------------------
# Scripted strategy + helpers (shared with the engine tests' contract)
# ---------------------------------------------------------------------------
class _ScriptedStrategy:
    def __init__(self, scripts: list[tuple], *, warmup_bars: int = 1) -> None:
        self._scripts = scripts
        self._i = 0
        self._warmup = warmup_bars
        self._last_snapshot: dict = {}
        self._last_justs: tuple[str, ...] = ()

    @property
    def warmup_bars(self) -> int:
        return self._warmup

    @property
    def last_indicator_snapshot(self) -> dict:
        return self._last_snapshot

    @property
    def last_justifications(self) -> tuple[str, ...]:
        return self._last_justs

    def predict(self, bars: pa.Table, symbol: str) -> LivePrediction:
        if self._i < len(self._scripts):
            d, sl, tp, justs = self._scripts[self._i]
        else:
            d, sl, tp, justs = 0.0, None, None, ()
        self._i += 1
        close = float(bars.column("close")[-1].as_py()) if bars.num_rows else 100.0
        self._last_snapshot = {"sl_price": sl, "tp_price": tp, "close": close}
        self._last_justs = justs
        return LivePrediction(
            symbol=symbol, direction=d, magnitude=0.01, volatility=0.01,
            confidence=0.5, horizon="scalp",
            ts=datetime.now(UTC).isoformat(), justifications=justs,
        )


def _bars(closes: list[float], highs: list[float], lows: list[float]) -> pa.Table:
    n = len(closes)
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    return pa.table(
        {"ts": ts, "open": closes, "high": highs, "low": lows,
         "close": closes, "volume": [1.0] * n},
        schema=OHLCV_SCHEMA,
    )


def _cfg(**overrides) -> ScalpBacktestConfig:
    base: dict = {
        "bankroll_start": 10.0, "leverage": 10.0, "allocation": 1.0,
        "risk_per_trade": 0.025, "rr_ratio": 1.3, "max_sl_pct": 0.04,
        "buffer_bars": 1, "attach_stops": True, "flip_to_flat": True,
        "max_drawdown": None, "stop_at": None, "cooldown_bars": 0,
        "cost": BYBIT_TESTNET_COSTS,
    }
    base.update(overrides)
    return ScalpBacktestConfig(**base)


# ---------------------------------------------------------------------------
# 1. Cost preset
# ---------------------------------------------------------------------------
class TestBybitTestnetCosts:
    def test_round_trip_is_15_bps(self) -> None:
        # 2 * (5.5 commission + 1 slippage + 1 half-spread) = 15 bps.
        assert round_trip_cost_bps(BYBIT_TESTNET_COSTS) == pytest.approx(15.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Sim ↔ pure-fn parity (deterministic)
# ---------------------------------------------------------------------------
class TestFidelityParity:
    def test_long_tp_net_pnl_matches_pure_fn(self) -> None:
        # Entry long at 100 (sl=98, tp=104 -> sl_distance=2). TP hit at 104.
        bars = _bars([100.0, 100.0], [100.0, 104.0], [100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 98.0, 104.0, ())])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        t = res.trades[0]
        # qty = risk_size_qty(10, 0.025, 2, notional_cap=10*10*1/100=1.0) = 0.125
        expected_qty = risk_size_qty(
            bankroll=10.0, risk_per_trade=0.025, sl_distance=2.0, notional_cap_qty=1.0,
        )
        assert t.qty == pytest.approx(expected_qty, rel=1e-9)
        expected_net = fidelity_expected_net_pnl(
            side=Side.LONG, qty=t.qty, entry_price=100.0, exit_price=104.0,
            cost=BYBIT_TESTNET_COSTS,
        )
        assert t.net_pnl == pytest.approx(expected_net, rel=1e-9)
        assert t.net_pnl > 0.0

    def test_short_sl_net_loss_matches_pure_fn(self) -> None:
        # Entry short at 100 (sl=102, tp=96 -> sl_distance=2). SL hit at 102.
        bars = _bars([100.0, 100.0], [100.0, 102.0], [100.0, 99.0])
        strat = _ScriptedStrategy([(-1.0, 102.0, 96.0, ())])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        t = res.trades[0]
        assert t.hit_sl is True
        # Exact parity: the engine's net PnL equals the pure-fn restatement, which
        # charges exit fees on the exit notional (slightly > entry at the SL).
        expected_net = fidelity_expected_net_pnl(
            side=Side.SHORT, qty=t.qty, entry_price=100.0, exit_price=102.0,
            cost=BYBIT_TESTNET_COSTS,
        )
        assert t.net_pnl == pytest.approx(expected_net, rel=1e-9)
        actual_loss_frac = -t.net_pnl / t.entry_bankroll
        # Risk-sized qty clears the target: gross loss fraction ≈ 2.5% (risk_per_trade),
        # plus ~15bps round-trip fees on the notional.
        assert actual_loss_frac == pytest.approx(0.025, abs=0.002)


# ---------------------------------------------------------------------------
# 3. Real-bar reproduction of the documented SOL min-lot-overshoot case
# ---------------------------------------------------------------------------
def _real_sol_5m() -> pa.Table:
    from kairon.data.history_store import read_history
    root = Path(__file__).resolve().parent.parent.parent / "data"
    bars = read_history(root, "SOL-USDT-PERP", "5m")
    if bars.num_rows < 500:
        pytest.skip("research history store not populated (run scripts/fetch_history.py)")
    return bars


def _is_overshoot_trade(t, cfg: ScalpBacktestConfig) -> bool:
    """True if the trade's qty was bumped up to the min lot from a smaller risk qty."""
    if cfg.min_qty <= 0:
        return False
    notional_cap = (t.entry_bankroll * cfg.leverage * cfg.allocation) / t.entry_price
    risk_qty = risk_size_qty(
        bankroll=t.entry_bankroll, risk_per_trade=cfg.risk_per_trade,
        sl_distance=t.sl_distance, notional_cap_qty=notional_cap,
    )
    return t.qty >= cfg.min_qty and risk_qty < cfg.min_qty


class TestSolMinLotOvershootFidelity:
    def test_guard_off_reproduces_documented_overshoot_loss(self) -> None:
        # Pre-fix live behaviour: overshoot trades; an SL hit loses > 2.5% target.
        bars = _real_sol_5m()
        from kairon.live.strategy import ScalpingStrategy
        cfg = _cfg(
            min_qty=0.1, qty_step=0.1, allow_min_lot_overshoot=True,
            enforce_risk_cap=False,  # guard OFF = the pre-fix live path
            buffer_bars=200,
        )
        res = run_scalp_backtest(
            bars=bars, strategy=ScalpingStrategy(), symbol="SOL-USDT-PERP", config=cfg,
        )
        overshoot_sl = [
            t for t in res.trades if t.hit_sl and _is_overshoot_trade(t, cfg)
        ]
        assert overshoot_sl, "expected at least one overshoot SL trade on real SOL 5m"
        t = overshoot_sl[0]
        loss_frac = -t.net_pnl / t.entry_bankroll
        # The overshoot loss must exceed the 2.5% target — the documented bug.
        assert loss_frac > 0.025
        # And it must match the pure-fn implied loss (fidelity to the risk math).
        expected = implied_loss_fraction(
            qty=t.qty, sl_distance=t.sl_distance, bankroll=t.entry_bankroll,
            cost=BYBIT_TESTNET_COSTS, entry_price=t.entry_price,
        )
        assert loss_frac == pytest.approx(expected, rel=2e-3)

    def test_guard_on_skips_overshoot_above_tol(self) -> None:
        # Phase 0.2 fix: with the risk-cap guard on, >2.75% overshoots are skipped.
        bars = _real_sol_5m()
        from kairon.live.strategy import ScalpingStrategy
        cfg = _cfg(
            min_qty=0.1, qty_step=0.1, allow_min_lot_overshoot=True,
            enforce_risk_cap=True, risk_cap_tol=0.10, buffer_bars=200,
        )
        res = run_scalp_backtest(
            bars=bars, strategy=ScalpingStrategy(), symbol="SOL-USDT-PERP", config=cfg,
        )
        # No trade should be an overshoot that breaches the cap; the guard skips
        # those as risk_cap_breach_overshoot before entry.
        overshoot_trades = [t for t in res.trades if _is_overshoot_trade(t, cfg)]
        # Any overshoot that did trade must be within the guard's tolerance. The
        # guard checks the GROSS implied_risk (qty*sl_distance/bankroll) against
        # risk_per_trade*(1+tol); fees are a separate cost layer, so assert on
        # gross implied_risk, not the fee-inclusive loss fraction.
        for t in overshoot_trades:
            gross = implied_risk(t.qty, t.sl_distance, t.entry_bankroll)
            assert gross <= 0.025 * 1.10 + 1e-9
        # The skip reason exists in the engine's vocabulary (guard is wired) —
        # either it skipped oversize overshoots, or none fired this window.
        assert any(s.reason == "risk_cap_breach_overshoot" for s in res.skips) or not overshoot_trades

    def test_overshoot_disabled_skips_below_min_lot(self) -> None:
        # The conservative alternative: don't overshoot at all -> sub-min-lot skips.
        bars = _real_sol_5m()
        from kairon.live.strategy import ScalpingStrategy
        cfg = _cfg(
            min_qty=0.1, qty_step=0.1, allow_min_lot_overshoot=False,
            enforce_risk_cap=True, buffer_bars=200,
        )
        res = run_scalp_backtest(
            bars=bars, strategy=ScalpingStrategy(), symbol="SOL-USDT-PERP", config=cfg,
        )
        below = [s for s in res.skips if s.reason == "below_min_lot"]
        assert below, "expected below_min_lot skips when overshoot is disabled"
        # None of the recorded trades is an overshoot (qty bumped from sub-min).
        assert not [t for t in res.trades if _is_overshoot_trade(t, cfg)]
