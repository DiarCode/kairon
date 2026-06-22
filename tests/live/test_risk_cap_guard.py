"""Tests for the Phase 0.2 risk-cap post-rounding guard + startup preflight.

The risk cap (risk_per_trade) is inviolable. The broker floors quantity DOWN
to the lot step (so plain risk sizing only shrinks risk), but min-lot
oversoot bumps qty UP to the min lot and confidence-scaled sizing inflates the
intended qty — both can breach the cap. These tests cover the pure functions
in ``kairon.live.pure_fns`` and the orchestrator integration that skips a
breaching trade with a ``risk_cap_breach_skip`` ledger row.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kairon.live.broker.base import Order, OrderStatus
from kairon.live.config import BankrollConfig, LiveConfig
from kairon.live.guardian import Guardian
from kairon.live.orchestrator import TradingLoop
from kairon.live.predictor import LivePrediction
from kairon.live.pure_fns import (
    clamp_effective_risk,
    classify_symbol_risk_cap,
    implied_risk,
    min_bankroll_to_clear_min_lot,
    post_rounding_guard,
    risk_size_qty,
)
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestPureRiskFns:
    def test_risk_size_qty_caps_at_notional(self) -> None:
        # risk_qty = 0.025*10/0.1 = 2.5; notional_cap = 10*10*1/0.5 = 200 -> 2.5
        q = risk_size_qty(bankroll=10.0, risk_per_trade=0.025, sl_distance=0.1,
                          notional_cap_qty=200.0)
        assert q == pytest.approx(2.5, rel=1e-9)
        # Tight stop -> risk_qty huge -> notional binds.
        q2 = risk_size_qty(bankroll=10.0, risk_per_trade=0.025, sl_distance=0.0001,
                           notional_cap_qty=200.0)
        assert q2 == pytest.approx(200.0, rel=1e-9)

    def test_risk_size_qty_falls_back_to_notional_without_sl(self) -> None:
        q = risk_size_qty(bankroll=10.0, risk_per_trade=0.025, sl_distance=None,
                          notional_cap_qty=200.0)
        assert q == 200.0

    def test_implied_risk(self) -> None:
        # 0.1 qty, 3.0 sl_distance, 10 bankroll -> 0.1*3/10 = 0.03 (3%)
        assert implied_risk(0.1, 3.0, 10.0) == pytest.approx(0.03, rel=1e-9)
        assert implied_risk(0.1, None, 10.0) == 0.0
        assert implied_risk(0.1, 3.0, 0.0) == 0.0

    def test_clamp_effective_risk_never_exceeds_cap(self) -> None:
        # 1.5x on 0.15 = 0.225 -> clamped to 0.20 (the Critic-required runtime
        # guarantee; config validates le=0.2 at LOAD only).
        assert clamp_effective_risk(risk_per_trade=0.15, multiplier=1.5, cap=0.2) == 0.2
        # Below cap passes through.
        assert clamp_effective_risk(risk_per_trade=0.025, multiplier=1.5, cap=0.2) == pytest.approx(0.0375, rel=1e-9)

    def test_post_rounding_guard_below_min_lot_skips(self) -> None:
        _qty, reason = post_rounding_guard(
            raw_qty=0.05, min_qty=0.1, sl_distance=3.0, bankroll=10.0,
            risk_per_trade=0.025, tol=0.10, enforce_risk_cap=True,
            allow_min_lot_overshoot=False,
        )
        assert reason == "below_min_lot"

    def test_post_rounding_guard_overshoot_within_tol_trades(self) -> None:
        # risk_qty=0.0926 < 0.1 -> bump to 0.1; implied=0.1*2.7/10=0.027 <= 0.0275
        qty, reason = post_rounding_guard(
            raw_qty=0.0926, min_qty=0.1, sl_distance=2.7, bankroll=10.0,
            risk_per_trade=0.025, tol=0.10, enforce_risk_cap=True,
            allow_min_lot_overshoot=True,
        )
        assert reason is None
        assert qty == pytest.approx(0.1, rel=1e-9)

    def test_post_rounding_guard_overshoot_breach_skips(self) -> None:
        # bump to 0.1; implied=0.1*3/10=0.03 > 0.025*1.1=0.0275 -> breach
        _qty, reason = post_rounding_guard(
            raw_qty=0.0833, min_qty=0.1, sl_distance=3.0, bankroll=10.0,
            risk_per_trade=0.025, tol=0.10, enforce_risk_cap=True,
            allow_min_lot_overshoot=True,
        )
        assert reason == "risk_cap_breach_overshoot"

    def test_post_rounding_guard_no_breach_when_clears(self) -> None:
        qty, reason = post_rounding_guard(
            raw_qty=2.5, min_qty=0.1, sl_distance=0.1, bankroll=10.0,
            risk_per_trade=0.025, tol=0.10, enforce_risk_cap=True,
            allow_min_lot_overshoot=False,
        )
        assert reason is None
        assert qty == pytest.approx(2.5, rel=1e-9)

    def test_post_rounding_guard_disabled_when_enforce_false(self) -> None:
        # enforce_risk_cap=False -> even a blatant overshoot is allowed.
        qty, reason = post_rounding_guard(
            raw_qty=0.0833, min_qty=0.1, sl_distance=3.0, bankroll=10.0,
            risk_per_trade=0.025, tol=0.10, enforce_risk_cap=False,
            allow_min_lot_overshoot=True,
        )
        assert reason is None
        assert qty == pytest.approx(0.1, rel=1e-9)


class TestPreflightClassification:
    def test_tradeable_when_risk_qty_clears_min_lot(self) -> None:
        # SOL @ 50, sl 2% -> sl_distance=1.0; risk_qty=0.025*10/1=0.25 >= 0.1
        r = classify_symbol_risk_cap(
            symbol="SOL-USDT-PERP", bankroll=10.0, risk_per_trade=0.025,
            leverage=10.0, allocation=1.0, min_qty=0.1, price=50.0,
            sl_distance_pct=0.02,
        )
        assert r["clears_min_lot"] is True
        assert r["verdict"] == "tradeable"
        assert r["risk_qty"] == pytest.approx(0.25, rel=1e-9)

    def test_skip_below_min_lot_when_overshoot_within_tol(self) -> None:
        # sl_distance_pct=0.054 -> sl_distance=2.7; risk_qty=0.25/2.7=0.0926<0.1
        # overshoot implied=0.1*2.7/10=0.027 <= 0.0275 -> skip_below_min_lot
        r = classify_symbol_risk_cap(
            symbol="SOL-USDT-PERP", bankroll=10.0, risk_per_trade=0.025,
            leverage=10.0, allocation=1.0, min_qty=0.1, price=50.0,
            sl_distance_pct=0.054,
        )
        assert r["clears_min_lot"] is False
        assert r["verdict"] == "skip_below_min_lot"

    def test_skip_risk_cap_breach_when_overshoot_too_large(self) -> None:
        # sl_distance_pct=0.06 -> sl_distance=3.0; risk_qty=0.0833<0.1;
        # overshoot implied=0.1*3/10=0.03 > 0.0275 -> skip_risk_cap_breach
        r = classify_symbol_risk_cap(
            symbol="SOL-USDT-PERP", bankroll=10.0, risk_per_trade=0.025,
            leverage=10.0, allocation=1.0, min_qty=0.1, price=50.0,
            sl_distance_pct=0.06,
        )
        assert r["verdict"] == "skip_risk_cap_breach"
        assert r["min_bankroll_to_clear"] == pytest.approx(12.0, rel=1e-9)

    def test_min_bankroll_to_clear_min_lot(self) -> None:
        # min_qty=0.1, price=50, sl_pct=0.06, risk=0.025 -> 0.1*50*0.06/0.025 = 12
        b = min_bankroll_to_clear_min_lot(
            min_qty=0.1, price=50.0, sl_distance_pct=0.06, risk_per_trade=0.025,
        )
        assert b == pytest.approx(12.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


class _ScalpStrategy:
    warmup_bars = 1

    def __init__(self, direction: float = 1.0, sl_price: float | None = None,
                 tp_price: float | None = None, close: float | None = None) -> None:
        self._direction = direction
        self._sl_price = sl_price
        self._tp_price = tp_price
        self._close = close
        self.last_indicator_snapshot: dict = {}

    def predict(self, table: object, symbol: str) -> LivePrediction:
        snap: dict = {"sl_price": self._sl_price, "tp_price": self._tp_price}
        if self._close is not None:
            snap["close"] = self._close
        self.last_indicator_snapshot = snap
        return LivePrediction(
            symbol=symbol, direction=self._direction, magnitude=0.1,
            volatility=0.01, confidence=0.6, horizon="scalp",
            ts="2026-06-18T12:00:00+00:00",
        )


class _SolBroker:
    """Broker whose SOL min lot is 0.1 (the live overshoot case)."""

    def __init__(self) -> None:
        self.orders: list[Order] = []
        self._equity = 10_000.0

    async def place_order(self, order: Order) -> Order:
        self.orders.append(order)
        return order.model_copy(
            update={"status": OrderStatus.FILLED, "broker_id": f"mock-{order.id[:8]}"}
        )

    async def get_balances(self):
        from kairon.live.broker.base import Balance
        return [Balance(currency="USDT", available=self._equity, total=self._equity,
                        ts="2026-06-18T12:00:00+00:00")]

    async def get_positions(self, symbol=None):
        return []

    async def get_last_price(self, symbol):
        return None

    def min_qty_for(self, symbol):
        return 0.1  # SOL min lot


def _bar(symbol: str, close: float, minute: int = 0):
    from datetime import UTC, datetime

    import pyarrow as pa
    ts = datetime(2026, 6, 18, 12, minute, 0, tzinfo=UTC)
    return pa.table(
        {"ts": pa.array([ts], type=pa.timestamp("us", tz="UTC")),
         "open": pa.array([close], type=pa.float64()),
         "high": pa.array([close], type=pa.float64()),
         "low": pa.array([close], type=pa.float64()),
         "close": pa.array([close], type=pa.float64()),
         "volume": pa.array([100.0], type=pa.float64()),
         "symbol": pa.array([symbol], type=pa.string())},
    )


def _config(**overrides):
    defaults = {
        "symbols": ("SOL-USDT-PERP",), "timeframe": "1m", "cadence_seconds": 60,
        "max_daily_loss_pct": 0.03, "max_open_positions": 3, "warmup_bars": 1,
        "reconcile_interval_seconds": 300, "reconcile_grace_seconds": 120,
        "dry_run": True,
    }
    defaults.update(overrides)
    return LiveConfig(**defaults)


class TestOrchestratorRiskCapGuard:
    @pytest.mark.asyncio
    async def test_overshoot_breach_skips_with_ledger_row(self) -> None:
        # SOL @ 50, sl_price=47 -> sl_distance=3.0; risk_qty=0.0833 < min 0.1.
        # allow_min_lot_overshoot=True bumps to 0.1; implied=0.03 > 0.0275 ->
        # risk_cap_breach_overshoot -> skip, no order, distinct ledger row.
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30,
                             enforce_risk_cap=True, allow_min_lot_overshoot=True,
                             risk_cap_tol=0.10)
        broker = _SolBroker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=47.0, tp_price=50.0 + 3.9)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(), store=store,
                feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("SOL-USDT-PERP", 50.0, minute=0))
            await loop._process_tick(_bar("SOL-USDT-PERP", 50.0, minute=1))
            assert broker.orders == []  # skipped, no order sent
            kinds = [row["kind"] for row in store.get_ledger()]
            assert "risk_cap_breach_skip" in kinds
            assert loop.bankroll == 10.0  # unchanged
            store.close()

    @pytest.mark.asyncio
    async def test_overshoot_within_tol_trades_min_lot(self) -> None:
        # SOL @ 50, sl_price=47.3 -> sl_distance=2.7; risk_qty=0.0926 < 0.1.
        # bump to 0.1; implied=0.027 <= 0.0275 -> trade min lot 0.1.
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30,
                             enforce_risk_cap=True, allow_min_lot_overshoot=True,
                             risk_cap_tol=0.10)
        broker = _SolBroker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=47.3, tp_price=50.0 + 3.51)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(), store=store,
                feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("SOL-USDT-PERP", 50.0, minute=0))
            await loop._process_tick(_bar("SOL-USDT-PERP", 50.0, minute=1))
            assert len(broker.orders) == 1
            assert broker.orders[0].qty == pytest.approx(0.1, rel=1e-9)
            store.close()

    @pytest.mark.asyncio
    async def test_default_skips_sub_min_lot_without_overshoot(self) -> None:
        # allow_min_lot_overshoot defaults False -> sub-min-lot skips (legacy).
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _SolBroker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=47.0, tp_price=53.9)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(), store=store,
                feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("SOL-USDT-PERP", 50.0, minute=0))
            await loop._process_tick(_bar("SOL-USDT-PERP", 50.0, minute=1))
            assert broker.orders == []
            kinds = [row["kind"] for row in store.get_ledger()]
            assert "skip" in kinds
            assert "risk_cap_breach_skip" not in kinds
            store.close()
