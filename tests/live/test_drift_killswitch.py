"""Tests for the Phase 3 drift kill-switch + setup_id journaling + wiring.

Covers:
- :class:`DriftKillSwitch` rolling-window semantics (no halt under threshold,
  halt on low win-rate, halt on negative expectancy, per-setup halt, min-sample
  guard, window eviction).
- :meth:`LiveStore.decision_setup_id` reading the setup_id back from the
  journaled indicator snapshot.
- Orchestrator wiring: feeding the killswitch via ``_apply_bankroll_close``
  halts the loop and writes a ``drift_killswitch`` event.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pytest

from kairon.live.broker.base import Order, OrderStatus
from kairon.live.config import BankrollConfig, LiveConfig
from kairon.live.drift_killswitch import (
    DriftKillSwitch,
    DriftKillSwitchConfig,
)
from kairon.live.guardian import Guardian
from kairon.live.journal import IndicatorSnapshot, RiskSnapshot, TradeDecision
from kairon.live.orchestrator import TradingLoop
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore


# ---------------------------------------------------------------------------
# DriftKillSwitch unit semantics
# ---------------------------------------------------------------------------
class TestDriftKillSwitch:
    def test_no_halt_below_min_trades(self) -> None:
        ks = DriftKillSwitch(DriftKillSwitchConfig(min_trades=10, min_win_rate=0.4))
        for _ in range(9):
            ks.record(-0.01)  # all losses, but under min sample
        assert ks.check().halt is False

    def test_halt_on_low_win_rate(self) -> None:
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=10, min_trades=10, min_win_rate=0.5, per_setup=False))
        # 3 wins, 7 losses -> 30% < 50%
        for i in range(10):
            ks.record(0.02 if i < 3 else -0.01)
        v = ks.check()
        assert v.halt is True
        assert v.reason is not None
        assert "win-rate" in v.reason

    def test_halt_on_negative_expectancy(self) -> None:
        # Win rate passes (60%) but per-trade expectancy is deeply negative
        # (small wins, large losses) -> expectancy floor trips.
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=10, min_trades=10, min_win_rate=0.4,
            min_expectancy=-0.002, per_setup=False))
        for i in range(10):
            ks.record(0.001 if i < 6 else -0.01)
        v = ks.check()
        assert v.halt is True
        assert v.reason is not None
        assert "expectancy" in v.reason

    def test_no_halt_when_edge_intact(self) -> None:
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=20, min_trades=10, min_win_rate=0.4, per_setup=False))
        for i in range(12):
            ks.record(0.01 if i < 8 else -0.005)  # 67% win, positive expectancy
        assert ks.check().halt is False

    def test_per_setup_halt_trips_independently(self) -> None:
        # Global window healthy, but one setup bleeds -> per-setup halt.
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=20, min_trades=10, min_win_rate=0.4,
            per_setup=True, per_setup_window=6, per_setup_min_trades=6,
            per_setup_min_win_rate=0.5))
        # Healthy mr_long trades keep the global window fine.
        for _ in range(10):
            ks.record(0.01, "mr_long")
        # momentum_short bleeds: 6 losses.
        for _ in range(6):
            ks.record(-0.01, "momentum_short")
        v = ks.check()
        assert v.halt is True
        assert v.setup_id == "momentum_short"
        assert v.reason is not None
        assert "momentum_short" in v.reason

    def test_window_evicts_old_trades(self) -> None:
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=5, min_trades=5, min_win_rate=0.4, per_setup=False))
        # 5 early losses fill the window, then 5 wins evict them -> healthy.
        for _ in range(5):
            ks.record(-0.01)
        for _ in range(5):
            ks.record(0.01)
        v = ks.check()
        assert v.halt is False
        assert v.win_rate == 1.0  # window is now all wins

    def test_per_setup_halt_on_negative_expectancy(self) -> None:
        # Global window healthy (under min_trades), but one setup has a
        # borderline win-rate (passes the per-setup win floor) with deeply
        # negative per-trade expectancy -> per-setup expectancy floor trips.
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=20, min_trades=20, min_win_rate=0.4,
            per_setup=True, per_setup_window=10, per_setup_min_trades=6,
            per_setup_min_win_rate=0.4, per_setup_min_expectancy=-0.002))
        # mr_short: 3 small wins, 3 large losses -> 50% win (>= 0.4 passes) but
        # expectancy (0.003 - 0.03) / 6 = -0.0045 < -0.002.
        for i in range(6):
            ks.record(0.001 if i < 3 else -0.01, "mr_short")
        v = ks.check()
        assert v.halt is True
        assert v.setup_id == "mr_short"
        assert v.reason is not None
        assert "expectancy" in v.reason

    def test_per_setup_window_evicts_old_trades(self) -> None:
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=20, min_trades=20, min_win_rate=0.4,
            per_setup=True, per_setup_window=5, per_setup_min_trades=5,
            per_setup_min_win_rate=0.8))
        # 5 losses -> 0% win < 0.8 -> per-setup halt.
        for _ in range(5):
            ks.record(-0.01, "mr_long")
        assert ks.check().halt is True
        # 5 wins evict the losses -> 100% win >= 0.8 -> no halt.
        for _ in range(5):
            ks.record(0.01, "mr_long")
        assert ks.check().halt is False

    def test_no_halt_when_edge_intact_per_setup_populated(self) -> None:
        # Exercises the fall-through: global OK, per-setup loop runs with a
        # populated healthy bucket, and check() returns halt=False.
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=20, min_trades=10, min_win_rate=0.4,
            per_setup=True, per_setup_window=12, per_setup_min_trades=6,
            per_setup_min_win_rate=0.3, per_setup_min_expectancy=-0.005))
        for _ in range(12):
            ks.record(0.01, "mr_long")  # healthy global + healthy mr_long
        v = ks.check()
        assert v.halt is False

    def test_record_none_setup_id_skips_per_setup(self) -> None:
        ks = DriftKillSwitch(DriftKillSwitchConfig(per_setup=True))
        ks.record(0.01, None)
        ks.record(0.01, "")
        assert ks._by_setup == {}

    def test_nonfinite_outcome_coerced_to_loss_not_neutralized(self) -> None:
        # A NaN/inf outcome (bad testnet fill) is coerced to a full-loss so it
        # drives the window toward halting instead of poisoning every
        # subsequent comparison (NaN makes win-rate/expectancy checks False).
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=10, min_trades=10, min_win_rate=0.4, per_setup=False,
            min_expectancy=-0.005))
        # 5 real wins + 5 non-finite (coerced to -1.0). Win-rate 50% passes the
        # 0.4 floor, but expectancy (0.05 - 5.0) / 10 = -0.495 trips the floor.
        for i in range(10):
            if i < 5:
                ks.record(0.01)
            else:
                ks.record(float("nan") if i % 2 == 0 else float("inf"))
        v = ks.check()
        assert v.halt is True
        assert v.reason is not None
        assert "expectancy" in v.reason


# ---------------------------------------------------------------------------
# Store setup_id round-trip
# ---------------------------------------------------------------------------
class TestStoreSetupId:
    def test_decision_setup_id_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            order_id = "order-1"
            decision = TradeDecision(
                order_id=order_id, symbol="SOL-USDT-PERP",
                timestamp="2026-06-19T10:00:00+00:00", strategy_name="scalping",
                direction=-1.0, confidence=0.6, magnitude=0.01, volatility=0.01,
                horizon="scalp",
                indicators=IndicatorSnapshot(setup_id="mr_short", regime="RANGE"),
                risk=RiskSnapshot(),
                justifications=("mr_short",),
            )
            store.write_decision(decision)
            assert store.decision_setup_id(order_id) == "mr_short"
            store.close()

    def test_decision_setup_id_none_when_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            order_id = "order-2"
            decision = TradeDecision(
                order_id=order_id, symbol="SOL-USDT-PERP",
                timestamp="2026-06-19T10:00:00+00:00", strategy_name="scalping",
                direction=-1.0, confidence=0.6, magnitude=0.01, volatility=0.01,
                horizon="scalp", indicators=IndicatorSnapshot(),
                risk=RiskSnapshot(), justifications=(),
            )
            store.write_decision(decision)
            assert store.decision_setup_id(order_id) is None
            # Unknown order id -> None, no error.
            assert store.decision_setup_id("does-not-exist") is None
            store.close()


# ---------------------------------------------------------------------------
# Per-setup report (scripts/run_scalping_session._scalping_extras)
# ---------------------------------------------------------------------------
class TestPerSetupReport:
    def test_per_setup_buckets_closed_decisions(self) -> None:
        from scripts.run_scalping_session import _scalping_extras

        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.db"
            store = LiveStore(p)
            for oid, sid, pnl, outcome in [
                ("o1", "mr_short", -0.25, "hit_sl"),
                ("o2", "mr_long", 0.32, "hit_tp"),
            ]:
                store.write_decision(TradeDecision(
                    order_id=oid, symbol="SOL-USDT-PERP",
                    timestamp="2026-06-19T10:00:00+00:00", strategy_name="scalping",
                    direction=-1.0, confidence=0.6, magnitude=0.01, volatility=0.01,
                    horizon="scalp",
                    indicators=IndicatorSnapshot(setup_id=sid, regime="RANGE"),
                    risk=RiskSnapshot(), justifications=(sid,),
                ))
                store.update_decision_outcome(oid, outcome, pnl,
                                              "2026-06-19T10:05:00+00:00")
            store.close()

            # Must not raise "Cannot operate on a closed database" — the
            # per-setup query runs before the connection is closed.
            extras = _scalping_extras(p)
            assert extras["n_sl"] == 1
            assert extras["n_tp"] == 1
            by_sid = {r["setup_id"]: r for r in extras["per_setup"]}
            assert set(by_sid) == {"mr_long", "mr_short"}
            assert by_sid["mr_long"]["tp"] == 1
            assert by_sid["mr_long"]["win_rate"] == 1.0
            assert by_sid["mr_short"]["sl"] == 1
            assert by_sid["mr_short"]["win_rate"] == 0.0

    def test_per_setup_empty_when_no_closed_decisions(self) -> None:
        from scripts.run_scalping_session import _scalping_extras

        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.db"
            store = LiveStore(p)
            store.close()
            extras = _scalping_extras(p)
            assert extras["per_setup"] == []
            assert extras["n_trades"] == 0


# ---------------------------------------------------------------------------
# Closed-bar feed alignment (scripts/run_scalping_session._drop_inprogress_bars)
# ---------------------------------------------------------------------------
def _bar_table(ts_list: list[datetime]) -> pa.Table:
    return pa.table({
        "ts": pa.array(ts_list, type=pa.timestamp("ms", tz="UTC")),
        "close": pa.array([1.0] * len(ts_list), type=pa.float64()),
    })


class TestDropInprogressBars:
    def test_drops_inprogress_tail_bar(self) -> None:
        from scripts.run_scalping_session import _drop_inprogress_bars

        # now = 15:34; on 5m the 15:25 bar (closes 15:30) is closed and the
        # 15:30 bar (closes 15:35) is in-progress -> drop the 15:30 row.
        now = datetime(2026, 6, 19, 15, 34, 0, tzinfo=UTC)
        ts = [
            datetime(2026, 6, 19, 15, 25, 0, tzinfo=UTC),
            datetime(2026, 6, 19, 15, 30, 0, tzinfo=UTC),
        ]
        out = _drop_inprogress_bars(_bar_table(ts), now, 5)
        assert out.num_rows == 1
        assert out.column("ts")[0].as_py() == ts[0]

    def test_keeps_table_when_last_bar_already_closed(self) -> None:
        from scripts.run_scalping_session import _drop_inprogress_bars

        # now = 15:30:01; the 15:25 bar closes at 15:30 <= now -> closed, kept.
        now = datetime(2026, 6, 19, 15, 30, 1, tzinfo=UTC)
        ts = [datetime(2026, 6, 19, 15, 25, 0, tzinfo=UTC)]
        out = _drop_inprogress_bars(_bar_table(ts), now, 5)
        assert out.num_rows == 1

    def test_empty_table_unchanged(self) -> None:
        from scripts.run_scalping_session import _drop_inprogress_bars

        now = datetime(2026, 6, 19, 15, 34, 0, tzinfo=UTC)
        out = _drop_inprogress_bars(_bar_table([]), now, 5)
        assert out.num_rows == 0


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------
class _Broker:
    def __init__(self) -> None:
        self.orders: list[Order] = []

    async def place_order(self, order: Order) -> Order:
        self.orders.append(order)
        return order.model_copy(
            update={"status": OrderStatus.FILLED, "broker_id": f"mock-{order.id[:8]}"}
        )

    async def get_balances(self):
        return []

    async def get_positions(self, symbol: str | None = None):
        return []

    async def get_last_price(self, symbol: str):
        return None

    def min_qty_for(self, symbol: str) -> float:
        return 1e-9


class _Strategy:
    warmup_bars = 1

    def __init__(self) -> None:
        self.last_indicator_snapshot: dict = {}

    def predict(self, table: object, symbol: str):
        from kairon.live.predictor import LivePrediction
        return LivePrediction(
            symbol=symbol, direction=0.0, magnitude=0.1, volatility=0.01,
            confidence=0.6, horizon="scalp", ts="2026-06-19T10:00:00+00:00",
        )


def _config() -> LiveConfig:
    return LiveConfig(
        symbols=("SOL-USDT-PERP",), timeframe="1m", cadence_seconds=60,
        max_daily_loss_pct=0.03, max_open_positions=3, warmup_bars=1,
        reconcile_interval_seconds=300, reconcile_grace_seconds=120, dry_run=True,
    )


class TestDriftKillswitchWiring:
    @pytest.mark.asyncio
    async def test_drift_halt_stops_loop_and_writes_event(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        ks = DriftKillSwitch(DriftKillSwitchConfig(
            window=5, min_trades=3, min_win_rate=0.5, per_setup=False))
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_Strategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
                drift_killswitch=ks,
            )
            loop._running = True
            # Three losing closes -> win-rate 0% < 50% over 3 trades -> halt.
            loop._apply_bankroll_close("SOL-USDT-PERP", -0.25)
            assert loop._running is True  # only 1 trade, under min_trades=3
            loop._apply_bankroll_close("SOL-USDT-PERP", -0.25)
            loop._apply_bankroll_close("SOL-USDT-PERP", -0.25)
            assert loop._running is False
            assert store.is_halted()
            notes = [r["note"] for r in store.get_ledger() if r["kind"] == "halt"]
            assert any("drift" in n for n in notes)
            events = store.get_all_events()
            assert any(e["kind"] == "drift_killswitch" for e in events)
            store.close()

    @pytest.mark.asyncio
    async def test_no_drift_halt_when_killswitch_off(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_Strategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
                drift_killswitch=None,
            )
            loop._running = True
            for _ in range(5):
                loop._apply_bankroll_close("SOL-USDT-PERP", -0.25)
            # Drawdown halt may fire (bankroll 10 -> 8.75 -> ...); but no
            # *drift* event is written when the killswitch is off.
            events = store.get_all_events()
            assert not any(e["kind"] == "drift_killswitch" for e in events)
            store.close()
