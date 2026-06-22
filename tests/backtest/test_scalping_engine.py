"""Tests for the vectorized scalping backtest engine (no network).

A scripted strategy replaces ``ScalpingStrategy`` so the tests assert exact
entry/exit/halt semantics without depending on indicator math. The scripted
strategy mimics the real contract: ``predict(bars, symbol) -> LivePrediction``
plus ``last_indicator_snapshot()`` exposing ``sl_price``/``tp_price`` and
``last_justifications()``. It advances one script entry per ``predict`` call
(the engine's call pattern is deterministic: one call per flat bar that is
eligible to enter, and one call per open bar in the flip-to-flat branch).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from kairon.backtest.cost import CostModel
from kairon.backtest.position import Side
from kairon.backtest.scalping_engine import (
    ScalpBacktestConfig,
    ScalpBacktestResult,
    _setup_id_from_justifications,
    run_scalp_backtest,
)
from kairon.data.io import OHLCV_SCHEMA
from kairon.live.predictor import LivePrediction


def _bars(
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: datetime | None = None,
) -> pa.Table:
    n = len(closes)
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    highs = highs or closes
    lows = lows or closes
    return pa.table(
        {
            "ts": ts,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1.0] * n,
        },
        schema=OHLCV_SCHEMA,
    )


class _ScriptedStrategy:
    """Returns one scripted (direction, sl, tp, justifications) per predict call."""

    def __init__(self, scripts: list[tuple], *, warmup_bars: int = 1) -> None:
        # scripts: list of (direction, sl_price, tp_price, justifications)
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


def _cfg(**overrides) -> ScalpBacktestConfig:
    base: dict = {
        "bankroll_start": 10.0, "leverage": 10.0, "allocation": 1.0, "risk_per_trade": 0.025,
        "rr_ratio": 1.3, "max_sl_pct": 0.04, "buffer_bars": 1, "attach_stops": True,
        "flip_to_flat": True, "max_drawdown": None, "stop_at": None, "cooldown_bars": 0,
        "cost": CostModel(commission_bps=0.0, slippage_bps=0.0, half_spread_bps=0.0, min_trade_bps=0.0),
    }
    base.update(overrides)
    return ScalpBacktestConfig(**base)


class TestScalpingEngineExits:
    def test_long_tp_hit_exits_at_tp_level(self) -> None:
        # bar0: entry long at 100 (sl=99, tp=102). bar1: high reaches 102 -> TP.
        bars = _bars([100.0, 100.0], highs=[100.0, 102.0], lows=[100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ("Momentum trend-following long",))])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        assert res.n_trades == 1
        t = res.trades[0]
        assert t.side is Side.LONG
        assert t.hit_tp is True
        assert t.hit_sl is False
        assert t.exit_price == 102.0  # stop LEVEL, not the crossing high
        assert t.entry_price == 100.0
        assert t.net_pnl > 0.0
        assert t.exit_bankroll > t.entry_bankroll
        assert t.setup_id == "momentum_long"

    def test_short_sl_hit_exits_at_sl_level(self) -> None:
        # bar0: entry short at 100 (sl=101, tp=98). bar1: high reaches 101 -> SL.
        bars = _bars([100.0, 100.0], highs=[100.0, 101.0], lows=[100.0, 99.0])
        strat = _ScriptedStrategy([(-1.0, 101.0, 98.0, ("Overbought mean-reversion short",))])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        assert res.n_trades == 1
        t = res.trades[0]
        assert t.side is Side.SHORT
        assert t.hit_sl is True
        assert t.hit_tp is False
        assert t.exit_price == 101.0
        assert t.net_pnl < 0.0
        assert t.setup_id == "mr_short"

    def test_both_crossed_assumes_sl_first(self) -> None:
        # long sl=99 tp=102; bar1 spans [98, 103] -> both crossed, SL wins.
        bars = _bars([100.0, 100.0], highs=[100.0, 103.0], lows=[100.0, 98.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ())])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        t = res.trades[0]
        assert t.hit_sl is True
        assert t.hit_tp is False
        assert t.exit_price == 99.0


class TestScalpingEngineSkips:
    def test_min_lot_skip_records_skip(self) -> None:
        # raw_qty = 0.025*10/1 = 0.25 << min_qty=10 -> below_min_lot skip.
        bars = _bars([100.0, 100.0], highs=[100.0, 102.0], lows=[100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ())])
        cfg = _cfg(min_qty=10.0, allow_min_lot_overshoot=False)
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=cfg)
        assert res.n_trades == 0
        assert res.n_skips == 1
        assert res.skips[0].reason == "below_min_lot"
        assert res.skips[0].side is Side.LONG

    def test_risk_cap_overshoot_skip(self) -> None:
        # min_qty=1 bumps 0.25 -> 1.0; implied_risk = 1*2/10 = 0.20 > 0.025*1.1.
        bars = _bars([100.0, 100.0], highs=[100.0, 102.0], lows=[100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 102.0, 94.0, ())])  # sl_distance = 2
        cfg = _cfg(min_qty=1.0, allow_min_lot_overshoot=True, risk_per_trade=0.025, risk_cap_tol=0.10)
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=cfg)
        assert res.n_trades == 0
        assert res.n_skips == 1
        assert res.skips[0].reason == "risk_cap_breach_overshoot"


class TestScalpingEngineFlipAndHalt:
    def test_flip_to_flat_closes_on_signal_flip(self) -> None:
        # bar0: long entry. bar1: signal flips to short -> close at bar1 close.
        bars = _bars([100.0, 101.0, 101.0])
        strat = _ScriptedStrategy([
            (1.0, 99.0, 102.0, ()),   # bar0: enter long
            (-1.0, 102.0, 99.0, ()),  # bar1: flip -> close long at 101
        ])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        assert res.n_trades == 1
        t = res.trades[0]
        assert t.flip_close is True
        assert t.hit_sl is False
        assert t.hit_tp is False
        assert t.exit_price == 101.0

    def test_drawdown_halt_stops_new_entries(self) -> None:
        # One SL loss of ~2.5% then a 2% drawdown halt -> no second entry.
        bars = _bars([100.0, 100.0, 100.0, 100.0], highs=[100.0, 101.0, 100.0, 102.0],
                     lows=[100.0, 99.0, 100.0, 99.0])
        strat = _ScriptedStrategy([
            (-1.0, 101.0, 98.0, ()),  # bar0: short entry, SL at 101
            # bar1: SL hit (loss ~2.5%)
            (-1.0, 101.0, 98.0, ()),  # bar2: would re-enter, but halted
        ])
        cfg = _cfg(max_drawdown=0.02)
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=cfg)
        assert res.n_trades == 1
        assert res.trades[0].hit_sl is True
        assert res.halted is True
        assert res.halt_reason == "max_drawdown"

    def test_stop_at_target_halt(self) -> None:
        # A TP win pushes bankroll past a tiny stop_at -> target_reached.
        bars = _bars([100.0, 100.0, 100.0], highs=[100.0, 102.0, 100.0], lows=[100.0, 100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ())])
        cfg = _cfg(stop_at=10.05)  # +0.5% target; a TP win clears it easily
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=cfg)
        assert res.halted is True
        assert res.halt_reason == "target_reached"
        assert res.final_bankroll > 10.05


class TestScalpingEngineCompounding:
    def test_two_wins_compound_qty_and_bankroll(self) -> None:
        # Constant close=100; two sequential TP wins at bars 1 and 3.
        bars = _bars(
            [100.0, 100.0, 100.0, 100.0],
            highs=[100.0, 102.0, 100.0, 102.0],
            lows=[100.0, 100.0, 100.0, 100.0],
        )
        strat = _ScriptedStrategy([
            (1.0, 99.0, 102.0, ()),  # bar0: enter long
            # bar1: TP hit (win 1)
            (1.0, 99.0, 102.0, ()),  # bar2: re-enter long at grown bankroll
            # bar3: TP hit (win 2)
        ])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        assert res.n_trades == 2
        assert res.trades[0].hit_tp is True
        assert res.trades[1].hit_tp is True
        # Second trade sized off the compounded bankroll -> larger qty + entry bankroll.
        assert res.trades[1].entry_bankroll > res.trades[0].entry_bankroll
        assert res.trades[1].qty > res.trades[0].qty
        assert res.final_bankroll > 10.0


class TestScalpingEngineMisc:
    def test_flat_signal_no_trades(self) -> None:
        bars = _bars([100.0, 100.0, 100.0])
        strat = _ScriptedStrategy([(0.0, None, None, ())])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        assert res.n_trades == 0
        assert res.final_bankroll == 10.0
        assert len(res.equity_curve) == 3
        assert all(eq == 10.0 for eq in res.equity_curve)

    def test_equity_curve_and_timestamps_length_equals_bars(self) -> None:
        bars = _bars([100.0] * 5)
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ())])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        assert len(res.equity_curve) == 5
        assert len(res.timestamps) == 5

    def test_open_position_closed_at_end(self) -> None:
        # Entry at the last bar; no stop hit afterwards -> end-of-data close.
        bars = _bars([100.0, 100.0], highs=[100.0, 100.0], lows=[100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ())])  # enters at bar0, never hits
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        assert res.n_trades == 1
        t = res.trades[0]
        assert t.flip_close is True
        assert t.exit_ts == res.timestamps[-1]

    def test_to_table_serializes_trades(self) -> None:
        bars = _bars([100.0, 100.0], highs=[100.0, 102.0], lows=[100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ("Momentum trend-following long",))])
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=_cfg())
        table = res.to_table()
        assert table.num_rows == 1
        assert "setup_id" in table.column_names
        assert table.column("side")[0].as_py() == "long"
        assert table.column("setup_id")[0].as_py() == "momentum_long"

    def test_qty_step_rounds_lot(self) -> None:
        bars = _bars([100.0, 100.0], highs=[100.0, 102.0], lows=[100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, ())])
        cfg = _cfg(qty_step=0.1)
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL", config=cfg)
        qty = res.trades[0].qty
        # raw_qty = 0.025*10/1 = 0.25 -> round(0.25/0.1) = round(2.5) = 2 (banker's) -> 0.2
        assert qty == pytest.approx(0.2, rel=1e-9)


class TestSetupIdHelper:
    @pytest.mark.parametrize(
        ("justs", "expected"),
        [
            (("Momentum trend-following short",), "momentum_short"),
            (("Momentum trend-following long",), "momentum_long"),
            (("Breakdown below lower Bollinger + volume surge",), "breakdown"),
            (("Breakout above upper Bollinger + volume surge",), "breakout"),
            (("Overbought mean-reversion short",), "mr_short"),
            (("Oversold mean-reversion long",), "mr_long"),
            (("something unrelated",), "unknown"),
            ((), "unknown"),
        ],
    )
    def test_setup_id_mapping(self, justs: tuple[str, ...], expected: str) -> None:
        assert _setup_id_from_justifications(justs) == expected


class TestRealHistorySmoke:
    """If the 8-week research store is present, run the real strategy over it.

    Skipped when the parquet is absent (e.g. CI without the testnet fetch).
    """

    def test_runs_over_real_sol_history(self) -> None:
        from kairon.data.history_store import read_history
        from kairon.live.strategy import ScalpingStrategy

        root = Path(__file__).resolve().parent.parent.parent / "data"
        bars = read_history(root, "SOL-USDT-PERP", "5m")
        if bars.num_rows < 500:
            pytest.skip("research history store not populated (run scripts/fetch_history.py)")
        # Subsample to keep the smoke test fast: every 20th bar of the last 4000.
        bars = bars.slice(max(0, bars.num_rows - 4000), 4000)
        strat = ScalpingStrategy()
        cfg = ScalpBacktestConfig(
            bankroll_start=10.0, leverage=10.0, allocation=1.0, risk_per_trade=0.025,
            buffer_bars=200, min_qty=0.1, qty_step=0.1, max_drawdown=0.30,
        )
        res = run_scalp_backtest(bars=bars, strategy=strat, symbol="SOL-USDT-PERP", config=cfg)
        assert isinstance(res, ScalpBacktestResult)
        assert len(res.equity_curve) == bars.num_rows
        assert res.final_bankroll > 0  # never went negative / blew up
