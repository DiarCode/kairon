"""Tests for the setup-selection analysis harness (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa

from kairon.backtest.scalping_engine import ScalpBacktestConfig
from kairon.backtest.setup_analysis import SetupEdge, analyze_setups, edge_table
from kairon.data.io import OHLCV_SCHEMA
from kairon.live.predictor import LivePrediction


class _ScriptedStrategy:
    """Returns one scripted (direction, sl, tp, setup_id) per predict call."""

    def __init__(self, scripts: list[tuple]) -> None:
        self._scripts = scripts
        self._i = 0
        self._snap: dict = {}

    @property
    def warmup_bars(self) -> int:
        return 1

    @property
    def last_indicator_snapshot(self) -> dict:
        return self._snap

    @property
    def last_justifications(self) -> tuple[str, ...]:
        return ()

    def predict(self, bars: pa.Table, symbol: str) -> LivePrediction:
        if self._i < len(self._scripts):
            d, sl, tp, setup_id = self._scripts[self._i]
        else:
            d, sl, tp, setup_id = 0.0, None, None, None
        self._i += 1
        close = float(bars.column("close")[-1].as_py()) if bars.num_rows else 100.0
        self._snap = {"sl_price": sl, "tp_price": tp, "close": close, "setup_id": setup_id}
        return LivePrediction(
            symbol=symbol, direction=d, magnitude=0.01, volatility=0.01,
            confidence=0.5, horizon="scalp", ts=datetime.now(UTC).isoformat(),
            justifications=(),
        )


def _bars(closes: list[float], highs: list[float], lows: list[float]) -> pa.Table:
    n = len(closes)
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    return pa.table(
        {"ts": ts, "open": closes, "high": highs, "low": lows,
         "close": closes, "volume": [1.0] * n},
        schema=OHLCV_SCHEMA,
    )


def _cfg() -> ScalpBacktestConfig:
    return ScalpBacktestConfig(
        bankroll_start=10.0, leverage=10.0, allocation=1.0, risk_per_trade=0.025,
        buffer_bars=1, min_qty=0.0, qty_step=0.0, max_drawdown=None, attach_stops=True,
        flip_to_flat=False,
    )


class TestAnalyzeSetups:
    def test_buckets_trades_by_setup_id(self) -> None:
        # Two TP winners (mr_long, mr_short) on a 4-bar sequence.
        bars = _bars(
            [100.0, 100.0, 100.0, 100.0],
            highs=[100.0, 102.0, 100.0, 102.0],
            lows=[100.0, 100.0, 100.0, 100.0],
        )
        strat = _ScriptedStrategy([
            (1.0, 99.0, 102.0, "mr_long"),    # bar0 entry, bar1 TP
            (-1.0, 101.0, 98.0, "mr_short"),  # bar2 entry, bar3 TP
        ])
        edges = analyze_setups(bars=bars, strategy=strat, symbol="SOL", timeframe="5m", config=_cfg())
        by_setup = {e.setup_id: e for e in edges}
        assert "mr_long" in by_setup
        assert "mr_short" in by_setup
        long_e = by_setup["mr_long"]
        assert long_e.n == 1
        assert long_e.hit_tp == 1
        assert long_e.win_rate == 1.0
        assert long_e.sum_pnl > 0.0
        assert isinstance(long_e, SetupEdge)

    def test_edge_table_renders_rows(self) -> None:
        bars = _bars([100.0, 100.0], [100.0, 102.0], [100.0, 100.0])
        strat = _ScriptedStrategy([(1.0, 99.0, 102.0, "mr_long")])
        edges = analyze_setups(bars=bars, strategy=strat, symbol="SOL", timeframe="5m", config=_cfg())
        table = edge_table(edges)
        assert "mr_long" in table
        assert "symbol" in table  # header
        assert "SOL" in table

    def test_empty_when_no_trades(self) -> None:
        bars = _bars([100.0, 100.0], [100.0, 100.0], [100.0, 100.0])
        strat = _ScriptedStrategy([(0.0, None, None, None)])
        edges = analyze_setups(bars=bars, strategy=strat, symbol="SOL", timeframe="5m", config=_cfg())
        assert edges == []
