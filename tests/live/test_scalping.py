"""Tests for the scalping engine: ScalpingStrategy signals + ATR SL/TP, and the
opt-in orchestrator behaviour (risk-based sizing, native TP/SL attach, bankroll
drawdown halt, per-symbol SL cooldown on losing closes).

All new orchestrator behaviour is opt-in (``attach_stops=False`` and
``risk_per_trade=0`` by default), so these tests construct the loop with the
scalping knobs explicitly enabled.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pytest

from kairon.live.broker.base import (
    Balance,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.config import BankrollConfig, LiveConfig
from kairon.live.guardian import Guardian
from kairon.live.orchestrator import TradingLoop
from kairon.live.predictor import LivePrediction
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore
from kairon.live.strategy import ScalpingStrategy

# ---------------------------------------------------------------------------
# Bar builders
# ---------------------------------------------------------------------------

_SCHEMA = pa.schema([
    ("ts", pa.timestamp("us", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
])


def _make_flat_then_spike(
    n_flat: int, direction: float, spike_pct: float, base: float = 100.0
) -> pa.Table:
    """Flat range then a single spike bar on a volume surge.

    ``direction=+1`` → final bar jumps up (upper-Bollinger breakout → long).
    ``direction=-1`` → final bar drops (lower-Bollinger breakdown → short).
    The flat portion makes Bollinger bands collapse to ~base, so the spike
    clearly breaches the band; the spike bar carries 2x volume (a surge).
    """
    closes: list[float] = [base] * n_flat
    final = base * (1 + direction * spike_pct)
    closes.append(final)
    highs = [c * 1.0002 for c in closes]
    lows = [c * 0.9998 for c in closes]
    opens = closes
    volumes = [100.0] * n_flat + [200.0]  # last bar = 2x the flat avg (surge)
    ts = [datetime(2026, 1, 1, 0, i, 0, tzinfo=UTC) for i in range(n_flat + 1)]
    return pa.table(
        {"ts": ts, "open": opens, "high": highs, "low": lows, "close": closes,
         "volume": volumes},
        schema=_SCHEMA,
    )


def _bar(symbol: str, close: float, minute: int = 0) -> pa.Table:
    ts = datetime(2026, 6, 18, 12, minute, 0, tzinfo=UTC)
    return pa.table(
        {
            "ts": pa.array([ts], type=pa.timestamp("us", tz="UTC")),
            "open": pa.array([close], type=pa.float64()),
            "high": pa.array([close], type=pa.float64()),
            "low": pa.array([close], type=pa.float64()),
            "close": pa.array([close], type=pa.float64()),
            "volume": pa.array([100.0], type=pa.float64()),
            "symbol": pa.array([symbol], type=pa.string()),
        }
    )


# ---------------------------------------------------------------------------
# ScalpingStrategy: signals + ATR SL/TP
# ---------------------------------------------------------------------------


class TestScalpingStrategySignals:
    def test_warmup_bars(self) -> None:
        s = ScalpingStrategy()
        assert s.warmup_bars == max(s.ema_slow + 1, s.macd_slow + s.macd_signal + 1,
                                   s.bollinger_period, 30)

    def test_neutral_before_warmup(self) -> None:
        s = ScalpingStrategy()
        bars = _make_flat_then_spike(10, -1.0, 0.05)  # 11 bars < 36 warmup
        pred = s.predict(bars, "BTC-USDT-PERP")
        assert pred.direction == 0.0
        assert pred.confidence == s.confidence_floor
        snap = s.last_indicator_snapshot
        assert snap.get("sl_price") is None
        assert snap.get("tp_price") is None

    def test_breakdown_short(self) -> None:
        s = ScalpingStrategy()
        bars = _make_flat_then_spike(40, direction=-1.0, spike_pct=0.05)
        pred = s.predict(bars, "BTC-USDT-PERP")
        snap = s.last_indicator_snapshot
        close = snap["close"]
        assert pred.direction == -1.0, (
            f"expected SHORT on breakdown, got {pred.direction}; "
            f"justifications={pred.justifications}"
        )
        # Short: SL above entry, TP below entry.
        assert snap["sl_price"] is not None
        assert snap["sl_price"] > close
        assert snap["tp_price"] is not None
        assert snap["tp_price"] < close
        sl_dist = abs(snap["sl_price"] - close)
        tp_dist = abs(close - snap["tp_price"])
        assert tp_dist == pytest.approx(s.rr_ratio * sl_dist, rel=1e-9)

    def test_momentum_trend_following_short(self) -> None:
        """A steady downtrend (no overbought, no volume surge, not at lower BB)
        must still produce a short via the momentum trend-following path."""
        s = ScalpingStrategy()
        # 40 flat bars then 20 bars of ACCELERATING selloff (each drop larger
        # than the last), so MACD histogram turns negative — a realistic crash
        # shape, unlike a constant-percentage decline whose histogram stays
        # positive because the absolute drop shrinks each bar.
        closes = [100.0] * 40 + [100.0 - 0.5 * (j ** 1.5) for j in range(1, 21)]
        n = len(closes)
        bars = pa.table(
            {"ts": [datetime(2026, 1, 1, 0, i, 0, tzinfo=UTC) for i in range(n)],
             "open": closes, "high": [c * 1.0005 for c in closes],
             "low": [c * 0.9995 for c in closes], "close": closes,
             "volume": [500.0] * n},
            schema=_SCHEMA,
        )
        pred = s.predict(bars, "BTC-USDT-PERP")
        snap = s.last_indicator_snapshot
        close = snap["close"]
        assert pred.direction == -1.0, (
            "expected momentum SHORT on steady downtrend, got "
            f"{pred.direction}; justifications={pred.justifications}"
        )
        assert any("Momentum trend-following short" in j for j in pred.justifications)
        # SL above entry, TP below, and the stop respects the max_sl_pct cap.
        assert snap["sl_price"] is not None
        assert snap["sl_price"] > close
        assert snap["tp_price"] is not None
        assert snap["tp_price"] < close
        sl_dist = abs(snap["sl_price"] - close)
        assert sl_dist <= close * s.max_sl_pct * (1 + 1e-9)

    def test_breakout_long(self) -> None:
        s = ScalpingStrategy()
        bars = _make_flat_then_spike(40, direction=+1.0, spike_pct=0.05)
        pred = s.predict(bars, "BTC-USDT-PERP")
        snap = s.last_indicator_snapshot
        close = snap["close"]
        assert pred.direction == +1.0, (
            f"expected LONG on breakout, got {pred.direction}; "
            f"justifications={pred.justifications}"
        )
        # Long: SL below entry, TP above entry.
        assert snap["sl_price"] is not None
        assert snap["sl_price"] < close
        assert snap["tp_price"] is not None
        assert snap["tp_price"] > close
        sl_dist = abs(close - snap["sl_price"])
        tp_dist = abs(snap["tp_price"] - close)
        assert tp_dist == pytest.approx(s.rr_ratio * sl_dist, rel=1e-9)

    def test_short_only_suppresses_long(self) -> None:
        s = ScalpingStrategy(short_only=True)
        bars = _make_flat_then_spike(40, direction=+1.0, spike_pct=0.05)
        pred = s.predict(bars, "BTC-USDT-PERP")
        assert pred.direction != 1.0, (
            f"short_only=True should suppress longs, got {pred.direction}; "
            f"justifications={pred.justifications}"
        )

    def test_no_short_into_strong_uptrend(self) -> None:
        """A clean strong uptrend must not produce a short (safety guard)."""
        s = ScalpingStrategy()
        base = 100.0
        step = base * 0.01  # +1% per bar
        n = 60
        closes = [base * (1 + step) ** i for i in range(n)]
        bars = pa.table(
            {"ts": [datetime(2026, 1, 1, 0, i, 0, tzinfo=UTC) for i in range(n)],
             "open": closes, "high": [c * 1.0005 for c in closes],
             "low": [c * 0.9995 for c in closes], "close": closes,
             "volume": [500.0] * n},
            schema=_SCHEMA,
        )
        pred = s.predict(bars, "BTC-USDT-PERP")
        assert pred.direction != -1.0, (
            "strategy shorted into a strong uptrend; "
            f"direction={pred.direction}; justifications={pred.justifications}"
        )

    def test_custom_rr_ratio_reflected_in_tp(self) -> None:
        s = ScalpingStrategy(rr_ratio=2.0)
        bars = _make_flat_then_spike(40, direction=-1.0, spike_pct=0.05)
        pred = s.predict(bars, "BTC-USDT-PERP")
        assert pred.direction == -1.0
        snap = s.last_indicator_snapshot
        close = snap["close"]
        sl_dist = abs(snap["sl_price"] - close)
        tp_dist = abs(close - snap["tp_price"])
        assert tp_dist == pytest.approx(2.0 * sl_dist, rel=1e-9)


# ---------------------------------------------------------------------------
# Test doubles for the orchestrator
# ---------------------------------------------------------------------------


class _Broker:
    """Minimal broker: records orders, reports a fixed equity, no fill stream."""

    def __init__(self, equity: float = 10_000.0, last_price: float | None = None) -> None:
        self._equity = equity
        self._last_price = last_price
        self.orders: list[Order] = []

    async def place_order(self, order: Order) -> Order:
        self.orders.append(order)
        return order.model_copy(
            update={"status": OrderStatus.FILLED, "broker_id": f"mock-{order.id[:8]}"}
        )

    async def get_balances(self) -> list[Balance]:
        return [Balance(currency="USDT", available=self._equity, total=self._equity,
                        ts="2026-06-18T12:00:00+00:00")]

    async def get_positions(self, symbol: str | None = None) -> list:
        return []

    async def get_last_price(self, symbol: str) -> float | None:
        return self._last_price

    def min_qty_for(self, symbol: str) -> float:
        return {"XRP-USDT-PERP": 0.1, "BTC-USDT-PERP": 0.001,
                "ETH-USDT-PERP": 0.01}.get(symbol, 1e-9)


class _ScalpStrategy:
    """Mock strategy exposing a controllable indicator snapshot for sizing/attach."""

    warmup_bars = 1

    def __init__(self, direction: float = 1.0, sl_price: float | None = None,
                 tp_price: float | None = None, close: float | None = None) -> None:
        self._direction = direction
        self._sl_price = sl_price
        self._tp_price = tp_price
        self._close = close
        self.last_indicator_snapshot: dict[str, Any] = {}

    def predict(self, table: object, symbol: str) -> LivePrediction:
        snap: dict[str, Any] = {"sl_price": self._sl_price,
                                "tp_price": self._tp_price}
        if self._close is not None:
            snap["close"] = self._close
        self.last_indicator_snapshot = snap
        return LivePrediction(
            symbol=symbol, direction=self._direction, magnitude=0.1,
            volatility=0.01, confidence=0.6, horizon="scalp",
            ts="2026-06-18T12:00:00+00:00",
        )


def _config(**overrides: object) -> LiveConfig:
    defaults: dict[str, object] = {
        "symbols": ("XRP-USDT-PERP",),
        "timeframe": "1m",
        "cadence_seconds": 60,
        "max_daily_loss_pct": 0.03,
        "max_open_positions": 3,
        "warmup_bars": 1,
        "reconcile_interval_seconds": 300,
        "reconcile_grace_seconds": 120,
        "dry_run": True,
    }
    defaults.update(overrides)
    return LiveConfig(**defaults)  # type: ignore[arg-type]


def _order(symbol: str, side: OrderSide, qty: float, price: float | None = None) -> Order:
    return Order(
        id=uuid.uuid4().hex, intent_id=uuid.uuid4().hex, trace_id=uuid.uuid4().hex,
        symbol=symbol, side=side, qty=qty, order_type=OrderType.MARKET, price=price,
        ts="2026-06-18T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Risk-based sizing + native TP/SL attach
# ---------------------------------------------------------------------------


class TestRiskBasedSizing:
    @pytest.mark.asyncio
    async def test_risk_qty_when_sl_distance_known(self) -> None:
        # bankroll=10, risk_per_trade=0.025 -> risk_amount=0.25 USDT.
        # XRP @ 0.5, sl_price=0.40 -> sl_distance=0.10 -> risk_qty=2.5.
        # notional_cap = 10*10*1/0.5 = 200 -> min(2.5? units...) risk_qty=0.25/0.10=2.5
        # Wait: risk_qty = 0.25/0.10 = 2.5 units; notional_cap=200 -> qty=2.5.
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=0.40, tp_price=0.513)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=0))
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=1))
            assert len(broker.orders) == 1
            qty = broker.orders[0].qty
            # risk_qty = 0.25/0.10 = 2.5, capped by notional 200 -> 2.5
            assert qty == pytest.approx(2.5, rel=1e-6)
            # Native TP/SL attached.
            assert broker.orders[0].sl == pytest.approx(0.40, rel=1e-9)
            assert broker.orders[0].tp == pytest.approx(0.513, rel=1e-9)
            store.close()

    @pytest.mark.asyncio
    async def test_notional_cap_binds_when_stop_too_tight(self) -> None:
        # sl_distance=0.01 -> risk_qty=0.25/0.01=25; notional_cap=200 -> qty=200? No:
        # min(25, 200)=25. Use an even tighter stop so risk_qty>notional_cap.
        # sl_price=0.4999 -> sl_distance=0.0001 -> risk_qty=2500 > notional_cap 200.
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=0.4999, tp_price=0.513)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=0))
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=1))
            assert len(broker.orders) == 1
            # notional_cap = 10*10*1/0.5 = 200 binds.
            assert broker.orders[0].qty == pytest.approx(200.0, rel=1e-6)
            store.close()

    @pytest.mark.asyncio
    async def test_skip_when_below_min_qty(self) -> None:
        # BTC @ 50000, sl_price=40000 -> sl_distance=10000 -> risk_qty=0.00025.
        # min_qty BTC=0.001 -> below min -> skip (no order, bankroll unchanged).
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=40000.0, tp_price=63000.0)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(symbols=("BTC-USDT-PERP",)), broker=broker,
                strategy=strategy, guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("BTC-USDT-PERP", 50000.0, minute=0))
            await loop._process_tick(_bar("BTC-USDT-PERP", 50000.0, minute=1))
            assert broker.orders == []
            kinds = [row["kind"] for row in store.get_ledger()]
            assert "skip" in kinds
            assert loop.bankroll == 10.0  # unchanged
            store.close()


class TestAttachStops:
    @pytest.mark.asyncio
    async def test_attach_stops_true_populates_sl_tp(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=0.40, tp_price=0.513)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=0))
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=1))
            assert len(broker.orders) == 1
            assert broker.orders[0].sl is not None
            assert broker.orders[0].tp is not None
            assert broker.orders[0].reduce_only is False  # fresh open, not a close
            store.close()

    @pytest.mark.asyncio
    async def test_attach_stops_false_leaves_sl_tp_none(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=0.40, tp_price=0.513)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=False,
            )
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=0))
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=1))
            assert len(broker.orders) == 1
            assert broker.orders[0].sl is None
            assert broker.orders[0].tp is None
            store.close()

    @pytest.mark.asyncio
    async def test_stops_reanchored_to_live_price(self) -> None:
        # Regression for the testnet TP/SL rejection. A short whose bar close is
        # 54.5 with strategy sl=56.5 (close+2.0) / tp=51.3 (close-3.2); the live
        # last price is 50.0 (price dropped ~8% between bar close and execution,
        # mirroring the volatile testnet book). Without re-anchoring, the
        # bar-close-anchored TP (51.3) sits ABOVE the live/fill price (50.0) and
        # Bybit rejects it ("TakeProfit for Sell should be lower than base").
        # Re-anchoring to the live price keeps TP below entry while preserving
        # the ATR sl/tp distances: tp = 50.0 - 3.2 = 46.8, sl = 50.0 + 2.0 = 52.0.
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker(last_price=50.0)
        strategy = _ScalpStrategy(direction=-1.0, sl_price=56.5, tp_price=51.3,
                                  close=54.5)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            await loop._process_tick(_bar("XRP-USDT-PERP", 54.5, minute=0))
            await loop._process_tick(_bar("XRP-USDT-PERP", 54.5, minute=1))
            assert len(broker.orders) == 1
            order = broker.orders[0]
            assert order.sl == pytest.approx(52.0, rel=1e-6)  # 50.0 + 2.0
            assert order.tp == pytest.approx(46.8, rel=1e-6)  # 50.0 - 3.2
            # Critical property: TP below the live/entry price (valid for a short).
            assert order.tp < 50.0
            store.close()


# ---------------------------------------------------------------------------
# Bankroll drawdown halt + per-symbol SL cooldown
# ---------------------------------------------------------------------------


class TestBankrollDrawdownHalt:
    @pytest.mark.asyncio
    async def test_30pct_drawdown_halts(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_ScalpStrategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            loop._running = True
            # Run bankroll up to 20 (peak), then drop to 13 -> 35% drawdown >= 30%.
            loop._apply_bankroll_close("XRP-USDT-PERP", 10.0)
            assert loop.bankroll_peak == 20.0
            loop._apply_bankroll_close("XRP-USDT-PERP", -7.0)
            assert loop.bankroll == 13.0
            assert loop._running is False
            assert store.is_halted()
            notes = [row["note"] for row in store.get_ledger() if row["kind"] == "halt"]
            assert any("drawdown" in n for n in notes)
            store.close()

    @pytest.mark.asyncio
    async def test_drawdown_under_threshold_does_not_halt(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_ScalpStrategy(),
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            loop._running = True
            loop._apply_bankroll_close("XRP-USDT-PERP", 10.0)  # peak 20
            loop._apply_bankroll_close("XRP-USDT-PERP", -5.0)  # -> 15, 25% dd < 30%
            assert loop.bankroll == 15.0
            assert loop._running is True
            assert not store.is_halted()
            store.close()


class TestSLCooldownOnLosingClose:
    @pytest.mark.asyncio
    async def test_record_sl_on_loss_when_attach_stops(self) -> None:
        guardian = Guardian(cooldown_seconds=300.0)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_ScalpStrategy(),
                guardian=guardian, reconciler=Reconciler(),
                store=store, feed=object(),
                bankroll=BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                        stop_at=100.0, milestones=(),
                                        risk_per_trade=0.025, rr_ratio=1.3,
                                        max_drawdown=0.30),
                attach_stops=True,
            )
            loop._running = True
            sym = "XRP-USDT-PERP"
            loop._positions[sym] = Position(symbol=sym, side=OrderSide.BUY, qty=1.0,
                                            avg_entry=100.0, ts="2026-06-18T12:00:00+00:00")
            loop._position_entry_orders[sym] = "entry-order-1"
            loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
            # Close the long at 90 -> realized = 1*(90-100) = -10 (loss).
            loop._update_position(sym, OrderSide.SELL, 1.0, 90.0, _order(sym, OrderSide.SELL, 1.0))
            assert guardian.is_cooling_down(sym) is True
            store.close()

    @pytest.mark.asyncio
    async def test_no_record_sl_on_win(self) -> None:
        guardian = Guardian(cooldown_seconds=300.0)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_ScalpStrategy(),
                guardian=guardian, reconciler=Reconciler(),
                store=store, feed=object(),
                bankroll=BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                        stop_at=100.0, milestones=(),
                                        risk_per_trade=0.025, rr_ratio=1.3,
                                        max_drawdown=0.30),
                attach_stops=True,
            )
            loop._running = True
            sym = "ETH-USDT-PERP"
            loop._positions[sym] = Position(symbol=sym, side=OrderSide.BUY, qty=1.0,
                                            avg_entry=100.0, ts="2026-06-18T12:00:00+00:00")
            loop._position_entry_orders[sym] = "entry-order-2"
            loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
            # Close at 110 -> realized = +10 (win) -> no cooldown.
            loop._update_position(sym, OrderSide.SELL, 1.0, 110.0, _order(sym, OrderSide.SELL, 1.0))
            assert guardian.is_cooling_down(sym) is False
            store.close()

    @pytest.mark.asyncio
    async def test_no_record_sl_when_attach_stops_false(self) -> None:
        guardian = Guardian(cooldown_seconds=300.0)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=_Broker(), strategy=_ScalpStrategy(),
                guardian=guardian, reconciler=Reconciler(),
                store=store, feed=object(),
                bankroll=BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                        stop_at=100.0, milestones=(),
                                        risk_per_trade=0.025, rr_ratio=1.3,
                                        max_drawdown=0.30),
                attach_stops=False,
            )
            loop._running = True
            sym = "SOL-USDT-PERP"
            loop._positions[sym] = Position(symbol=sym, side=OrderSide.BUY, qty=1.0,
                                            avg_entry=100.0, ts="2026-06-18T12:00:00+00:00")
            loop._position_entry_orders[sym] = "entry-order-3"
            loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
            # Losing close but attach_stops=False -> no cooldown engaged.
            loop._update_position(sym, OrderSide.SELL, 1.0, 90.0, _order(sym, OrderSide.SELL, 1.0))
            assert guardian.is_cooling_down(sym) is False
            store.close()


class TestSoftwareStopMonitor:
    """The software-side stop monitor caps losses independent of the exchange's
    attached TP/SL trigger reliability (on testnet, attached stops have been
    observed NOT to fire). Each tick it closes (reduce-only) any open position
    whose intended SL/TP has been crossed."""

    def _loop_with_long(self, *, store_path: Path, sl: float, tp: float,
                        last_price: float | None = None) -> TradingLoop:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker(last_price=last_price)
        loop = TradingLoop(
            config=_config(), broker=broker, strategy=_ScalpStrategy(),
            guardian=Guardian(cooldown_seconds=300.0), reconciler=Reconciler(),
            store=LiveStore(store_path), feed=object(),
            bankroll=cfg, attach_stops=True,
        )
        sym = "XRP-USDT-PERP"
        loop._positions[sym] = Position(
            symbol=sym, side=OrderSide.BUY, qty=1.0, avg_entry=100.0,
            ts="2026-06-18T12:00:00+00:00",
        )
        loop._position_entry_orders[sym] = "entry-stop-1"
        loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
        loop._position_stops[sym] = (sl, tp)
        return loop

    @pytest.mark.asyncio
    async def test_sl_cross_closes_position_reduce_only(self) -> None:
        # Long @ 100, SL=95, TP=110. Bar close drops to 90 (below SL) -> the
        # monitor must close the position reduce-only, realize the loss, and
        # engage the per-symbol SL cooldown. No new/second order this tick.
        with TemporaryDirectory() as tmp:
            loop = self._loop_with_long(
                store_path=Path(tmp) / "s.db", sl=95.0, tp=110.0,
            )
            await loop._process_tick(_bar("XRP-USDT-PERP", 90.0, minute=0))
            broker = loop._broker
            assert len(broker.orders) == 1
            close_order = broker.orders[0]
            assert close_order.side == OrderSide.SELL
            assert close_order.reduce_only is True
            assert close_order.sl is None
            assert close_order.tp is None
            assert "XRP-USDT-PERP" not in loop._positions  # position reconciled away
            assert loop._guardian.is_cooling_down("XRP-USDT-PERP")
            loop._store.close()

    @pytest.mark.asyncio
    async def test_tp_cross_closes_without_cooldown(self) -> None:
        # Long @ 100, SL=95, TP=110. Price rises to 115 (above TP) -> close on a
        # WIN -> no SL cooldown (cooldown is for losses only).
        with TemporaryDirectory() as tmp:
            loop = self._loop_with_long(
                store_path=Path(tmp) / "s.db", sl=95.0, tp=110.0, last_price=115.0,
            )
            # Bar close is irrelevant when the live last price is available;
            # use a neutral close so the bar itself doesn't imply a signal.
            await loop._process_tick(_bar("XRP-USDT-PERP", 100.0, minute=0))
            broker = loop._broker
            assert len(broker.orders) == 1
            assert broker.orders[0].side == OrderSide.SELL
            assert broker.orders[0].reduce_only is True
            assert "XRP-USDT-PERP" not in loop._positions
            assert not loop._guardian.is_cooling_down("XRP-USDT-PERP")
            loop._store.close()

    @pytest.mark.asyncio
    async def test_no_close_when_stops_not_crossed(self) -> None:
        # Long @ 100, SL=95, TP=110. Price stays at 100 -> no close, normal path
        # runs (and may itself trade); the monitor must not have fired a close.
        with TemporaryDirectory() as tmp:
            loop = self._loop_with_long(
                store_path=Path(tmp) / "s.db", sl=95.0, tp=110.0, last_price=100.0,
            )
            await loop._process_tick(_bar("XRP-USDT-PERP", 100.0, minute=0))
            # No reduce-only close order should have been placed this tick.
            assert not any(o.reduce_only for o in loop._broker.orders)
            assert "XRP-USDT-PERP" in loop._positions  # still open
            loop._store.close()

    @pytest.mark.asyncio
    async def test_rejected_close_reconciles_when_exchange_already_closed(self) -> None:
        # Regression for the testnet reconciliation race: the exchange's attached
        # SL fires and closes the position FIRST, the WS close fill does not
        # drain, and Bybit then rejects the redundant reduce-only software close
        # with ErrCode 110017 ("position is zero"). The orchestrator must
        # reconcile the local mirror to flat at the stop level (recording the
        # loss, SL cooldown, and bankroll compound) instead of leaving a stale
        # open position forever.
        class _RejectReduceBroker(_Broker):
            """Rejects reduce-only orders; reports the position as already gone."""

            async def place_order(self, order: Order) -> Order:
                # Record every order (including the rejected reduce-only close)
                # so tests can assert on it, then reject reduce-only ones.
                self.orders.append(order)
                if order.reduce_only:
                    return order.model_copy(update={"status": OrderStatus.REJECTED})
                return order.model_copy(
                    update={"status": OrderStatus.FILLED,
                            "broker_id": f"mock-{order.id[:8]}"}
                )

            async def get_positions(self, symbol: str | None = None) -> list:
                return []  # exchange already closed it

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                 stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                 rr_ratio=1.3, max_drawdown=0.30)
            broker = _RejectReduceBroker()
            loop = TradingLoop(
                config=_config(), broker=broker,
                strategy=_ScalpStrategy(), guardian=Guardian(cooldown_seconds=300.0),
                reconciler=Reconciler(), store=LiveStore(Path(tmp) / "s.db"),
                feed=object(), bankroll=cfg, attach_stops=True,
            )
            sym = "XRP-USDT-PERP"
            loop._positions[sym] = Position(
                symbol=sym, side=OrderSide.BUY, qty=1.0, avg_entry=100.0,
                ts="2026-06-18T12:00:00+00:00",
            )
            loop._position_entry_orders[sym] = "entry-recon-1"
            loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
            loop._position_stops[sym] = (98.0, 110.0)  # SL=98
            # Price 90 < SL 98 -> software SL close attempted, rejected, reconciled.
            await loop._process_tick(_bar(sym, 90.0, minute=0))
            # The reduce-only close was attempted (and rejected)...
            assert any(o.reduce_only for o in broker.orders)
            # ...and the local mirror is reconciled to flat at the SL level (98),
            # not left stale. Loss = 1*(98-100) = -2 -> bankroll 10 -> 8 (drawdown
            # 20% < 30%, so no halt — the reconcile is what we're testing here).
            assert sym not in loop._positions
            assert loop.bankroll == pytest.approx(8.0, rel=1e-9)
            assert loop._guardian.is_cooling_down(sym)  # losing close -> cooldown
            loop._store.close()


class TestReconcileDurationRepro:
    """Deterministic repro for the software-stop reconcile race + the
    duration_seconds fix. The exchange's attached SL fires and closes the
    position FIRST, the WS close fill does not drain, and Bybit rejects the
    redundant reduce-only software close (ErrCode 110017). The orchestrator
    must reconcile the local mirror to flat at the stop level AND record the
    TRUE holding duration (carried through the stops tuple), not a near-zero
    value from the synthetic close order's ``_utc_now_iso()`` timestamp."""

    @pytest.mark.asyncio
    async def test_reconcile_records_true_duration_and_event(self) -> None:
        class _RejectReduceBroker(_Broker):
            async def place_order(self, order: Order) -> Order:
                self.orders.append(order)
                if order.reduce_only:
                    return order.model_copy(update={"status": OrderStatus.REJECTED})
                return order.model_copy(
                    update={"status": OrderStatus.FILLED,
                            "broker_id": f"mock-{order.id[:8]}"}
                )

            async def get_positions(self, symbol: str | None = None) -> list:
                return []  # exchange already closed the position

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = Path(tmp) / "s.db"
            cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                 stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                 rr_ratio=1.3, max_drawdown=0.30)
            broker = _RejectReduceBroker()
            loop = TradingLoop(
                config=_config(), broker=broker,
                strategy=_ScalpStrategy(), guardian=Guardian(cooldown_seconds=300.0),
                reconciler=Reconciler(), store=LiveStore(db_path),
                feed=object(), bankroll=cfg, attach_stops=True,
            )
            sym = "XRP-USDT-PERP"
            # Position opened ~10 minutes ago. CRUCIALLY: do NOT set
            # _position_entry_ts (simulate it being absent) so the only source
            # of the true entry time is the 3-tuple carried in _position_stops.
            entry_ts = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
            loop._positions[sym] = Position(
                symbol=sym, side=OrderSide.BUY, qty=1.0, avg_entry=100.0,
                ts=entry_ts,
            )
            loop._position_entry_orders[sym] = "entry-recon-dur"
            loop._position_stops[sym] = (98.0, 110.0, entry_ts)  # SL=98, carry entry_ts
            await loop._process_tick(_bar(sym, 90.0, minute=0))

            # Reconciled to flat at the SL level (98): loss = 1*(98-100) = -2.
            assert sym not in loop._positions
            assert loop.bankroll == pytest.approx(8.0, rel=1e-9)
            assert loop._guardian.is_cooling_down(sym)

            # The reconcile event was written.
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            events = conn.execute(
                "SELECT kind FROM live_events WHERE kind = 'software_stop_reconcile'"
            ).fetchall()
            assert len(events) == 1
            # The closed-trade row records the TRUE holding duration (~600s),
            # not a near-zero value. Allow slack for test execution time.
            rows = conn.execute(
                "SELECT duration_seconds FROM live_closed_trades WHERE symbol = ?",
                (sym,),
            ).fetchall()
            conn.close()
            assert len(rows) == 1
            duration = rows[0]["duration_seconds"]
            assert duration is not None
            assert 570.0 <= float(duration) <= 900.0, (
                f"expected ~600s holding duration, got {duration} "
                "(near-zero would indicate the duration_seconds bug)"
            )
            loop._store.close()


class TestCancelOrphanStopsOnClose:
    """Phase 0.3 (c): every close path in attach_stops mode cancels the orphan
    attached TP/SL conditional orders for the symbol, so a stale stop cannot
    fire on a future move with no position to close (or interfere with the next
    position's freshly-attached stops)."""

    class _CancelBroker(_Broker):
        """``_Broker`` + a ``cancel_all`` that records the symbols cancelled."""

        def __init__(self, equity: float = 10_000.0, last_price: float | None = None) -> None:
            super().__init__(equity=equity, last_price=last_price)
            self.cancel_calls: list[str] = []

        async def cancel_all(self, symbol: str) -> list:
            self.cancel_calls.append(symbol)
            return []

    @pytest.mark.asyncio
    async def test_cancel_orphan_stops_noop_when_attach_stops_false(self) -> None:
        # The guard is active only in attach_stops mode; a no-op otherwise.
        with TemporaryDirectory() as tmp:
            broker = self._CancelBroker()
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=_ScalpStrategy(),
                guardian=Guardian(cooldown_seconds=300.0), reconciler=Reconciler(),
                store=LiveStore(Path(tmp) / "s.db"), feed=object(),
                bankroll=BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                        stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                        rr_ratio=1.3, max_drawdown=0.30),
                attach_stops=False,
            )
            await loop._cancel_orphan_stops("XRP-USDT-PERP")
            assert broker.cancel_calls == []
            loop._store.close()

    @pytest.mark.asyncio
    async def test_cancel_orphan_stops_called_when_attach_stops_true(self) -> None:
        with TemporaryDirectory() as tmp:
            broker = self._CancelBroker()
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=_ScalpStrategy(),
                guardian=Guardian(cooldown_seconds=300.0), reconciler=Reconciler(),
                store=LiveStore(Path(tmp) / "s.db"), feed=object(),
                bankroll=BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                        stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                        rr_ratio=1.3, max_drawdown=0.30),
                attach_stops=True,
            )
            await loop._cancel_orphan_stops("XRP-USDT-PERP")
            assert broker.cancel_calls == ["XRP-USDT-PERP"]
            loop._store.close()

    @pytest.mark.asyncio
    async def test_software_sl_close_cancels_orphans(self) -> None:
        # Long @ 100, SL=95. Bar drops to 90 -> software SL close fires, the
        # position is reconciled away, and the orphan attached TP/SL are cancelled.
        with TemporaryDirectory() as tmp:
            cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                 stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                 rr_ratio=1.3, max_drawdown=0.30)
            broker = self._CancelBroker()
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=_ScalpStrategy(),
                guardian=Guardian(cooldown_seconds=300.0), reconciler=Reconciler(),
                store=LiveStore(Path(tmp) / "s.db"), feed=object(),
                bankroll=cfg, attach_stops=True,
            )
            sym = "XRP-USDT-PERP"
            loop._positions[sym] = Position(
                symbol=sym, side=OrderSide.BUY, qty=1.0, avg_entry=100.0,
                ts="2026-06-18T12:00:00+00:00",
            )
            loop._position_entry_orders[sym] = "entry-cancel-1"
            loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
            loop._position_stops[sym] = (95.0, 110.0)
            await loop._process_tick(_bar(sym, 90.0, minute=0))
            assert sym not in loop._positions
            assert broker.cancel_calls == [sym]
            loop._store.close()

    @pytest.mark.asyncio
    async def test_reconcile_close_cancels_orphans(self) -> None:
        # The rejected-close reconcile path (exchange SL fired first) must also
        # cancel the orphan conditional orders after mirroring to flat.

        class _RejectCancelBroker(self._CancelBroker):
            async def place_order(self, order: Order) -> Order:
                self.orders.append(order)
                if order.reduce_only:
                    return order.model_copy(update={"status": OrderStatus.REJECTED})
                return order.model_copy(
                    update={"status": OrderStatus.FILLED,
                            "broker_id": f"mock-{order.id[:8]}"}
                )

            async def get_positions(self, symbol: str | None = None) -> list:
                return []

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                 stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                 rr_ratio=1.3, max_drawdown=0.30)
            broker = _RejectCancelBroker()
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=_ScalpStrategy(),
                guardian=Guardian(cooldown_seconds=300.0), reconciler=Reconciler(),
                store=LiveStore(Path(tmp) / "s.db"), feed=object(),
                bankroll=cfg, attach_stops=True,
            )
            sym = "XRP-USDT-PERP"
            loop._positions[sym] = Position(
                symbol=sym, side=OrderSide.BUY, qty=1.0, avg_entry=100.0,
                ts="2026-06-18T12:00:00+00:00",
            )
            loop._position_entry_orders[sym] = "entry-cancel-recon"
            loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
            loop._position_stops[sym] = (98.0, 110.0)
            await loop._process_tick(_bar(sym, 90.0, minute=0))
            assert sym not in loop._positions
            assert broker.cancel_calls == [sym]
            loop._store.close()


class TestFinalizeCoverage:
    """Phase 0.3 (b): ``finalize_open_positions`` journals a ``manual_close``
    for every position still open at shutdown — including one tracked only in the
    local mirror (``_positions``) with no entry-order record — so no open
    position is left without an outcome."""

    class _FinalizeBroker(_Broker):
        """Reports a configured open position per symbol with an unrealized PnL."""

        def __init__(self, positions: dict[str, Position]) -> None:
            super().__init__(equity=10_000.0)
            self._positions = positions
            self.cancel_calls: list[str] = []

        async def get_positions(self, symbol: str | None = None) -> list:
            if symbol is None:
                return list(self._positions.values())
            p = self._positions.get(symbol)
            return [p] if p is not None else []

        async def cancel_all(self, symbol: str) -> list:
            self.cancel_calls.append(symbol)
            return []

    @pytest.mark.asyncio
    async def test_finalizes_position_without_entry_order(self) -> None:
        # A position present in _positions but NOT in _position_entry_orders
        # (entry-order tracking drifted) must still be journaled as a
        # manual_close round trip with a bankroll debit + orphan cancel.
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "s.db"
            cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                 stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                 rr_ratio=1.3, max_drawdown=0.30)
            sym = "XRP-USDT-PERP"
            entry_ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
            broker = self._FinalizeBroker({
                sym: Position(symbol=sym, side=OrderSide.BUY, qty=1.0,
                              avg_entry=100.0, unrealized_pnl=-2.0, ts=entry_ts),
            })
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=_ScalpStrategy(),
                guardian=Guardian(cooldown_seconds=300.0), reconciler=Reconciler(),
                store=LiveStore(db_path), feed=object(),
                bankroll=cfg, attach_stops=True,
            )
            # Position in the local mirror, but NO entry order tracked.
            loop._positions[sym] = Position(
                symbol=sym, side=OrderSide.BUY, qty=1.0, avg_entry=100.0,
                unrealized_pnl=-2.0, ts=entry_ts,
            )
            loop._position_entry_ts[sym] = entry_ts
            await loop.finalize_open_positions()
            # Bankroll debited by the realized loss (-2): 10 -> 8.
            assert loop.bankroll == pytest.approx(8.0, rel=1e-9)
            assert sym in broker.cancel_calls  # orphan stops cancelled
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT realized_pnl FROM live_closed_trades WHERE symbol = ?",
                (sym,),
            ).fetchall()
            conn.close()
            assert len(rows) == 1
            assert float(rows[0]["realized_pnl"]) == pytest.approx(-2.0, rel=1e-9)
            loop._store.close()

    @pytest.mark.asyncio
    async def test_finalizes_both_entry_order_and_mirror_only_positions(self) -> None:
        # The union iteration covers a symbol with an entry order AND a symbol
        # tracked only in the mirror in the same finalize pass.
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "s.db"
            cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                                 stop_at=100.0, milestones=(), risk_per_trade=0.025,
                                 rr_ratio=1.3, max_drawdown=0.30)
            sym_a = "XRP-USDT-PERP"
            sym_b = "BTC-USDT-PERP"
            entry_ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
            broker = self._FinalizeBroker({
                sym_a: Position(symbol=sym_a, side=OrderSide.BUY, qty=1.0,
                                avg_entry=100.0, unrealized_pnl=3.0, ts=entry_ts),
                sym_b: Position(symbol=sym_b, side=OrderSide.BUY, qty=0.001,
                                avg_entry=150_000.0, unrealized_pnl=1.0, ts=entry_ts),
            })
            loop = TradingLoop(
                config=_config(symbols=(sym_a, sym_b)), broker=broker,
                strategy=_ScalpStrategy(), guardian=Guardian(cooldown_seconds=300.0),
                reconciler=Reconciler(), store=LiveStore(db_path), feed=object(),
                bankroll=cfg, attach_stops=True,
            )
            # sym_a has an entry order; sym_b is mirror-only.
            loop._positions[sym_a] = Position(
                symbol=sym_a, side=OrderSide.BUY, qty=1.0, avg_entry=100.0,
                unrealized_pnl=3.0, ts=entry_ts,
            )
            loop._position_entry_orders[sym_a] = "entry-finalize-a"
            loop._position_entry_ts[sym_a] = entry_ts
            loop._positions[sym_b] = Position(
                symbol=sym_b, side=OrderSide.BUY, qty=0.001, avg_entry=150_000.0,
                unrealized_pnl=1.0, ts=entry_ts,
            )
            await loop.finalize_open_positions()
            # Both realized PnLs compound into the bankroll: 10 + 3 + 1 = 14.
            assert loop.bankroll == pytest.approx(14.0, rel=1e-9)
            assert set(broker.cancel_calls) == {sym_a, sym_b}
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            syms = {r["symbol"] for r in conn.execute(
                "SELECT symbol FROM live_closed_trades")}
            conn.close()
            assert syms == {sym_a, sym_b}
            loop._store.close()


class TestFlipProtection:
    """When the signal flips against an open position, the loop must close to
    flat (reduce-only) rather than reversing in a single market order — the
    whipsaw behaviour that previously let a 2.5%-risk trade lose 12.8%."""
    @pytest.mark.asyncio
    async def test_flip_closes_to_flat_not_reverse(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        # Strategy now shouts SHORT while we hold a LONG.
        strategy = _ScalpStrategy(direction=-1.0, sl_price=0.40, tp_price=0.287)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(warmup_bars=1), broker=broker, strategy=strategy,
                guardian=Guardian(cooldown_seconds=300.0), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            sym = "XRP-USDT-PERP"
            # Hold a long of 2.5 units @ 0.5 (matches the risk-sized qty at this
            # bankroll/price, so without flip protection delta would REVERSE to
            # a 5-unit sell: 2.5 to close + 2.5 to open short).
            loop._positions[sym] = Position(
                symbol=sym, side=OrderSide.BUY, qty=2.5, avg_entry=0.5,
                ts="2026-06-18T12:00:00+00:00",
            )
            loop._position_entry_orders[sym] = "entry-flip-1"
            loop._position_entry_ts[sym] = "2026-06-18T12:00:00+00:00"
            await loop._process_tick(_bar(sym, 0.5, minute=0))  # warmup
            await loop._process_tick(_bar(sym, 0.5, minute=1))  # trades
            assert len(broker.orders) == 1
            order = broker.orders[0]
            assert order.side == OrderSide.SELL
            # Flattened to exactly the open qty — NOT a 2x reversal.
            assert order.qty == pytest.approx(2.5, rel=1e-6)
            assert order.reduce_only is True
            assert order.sl is None  # no stops on a close
            assert order.tp is None
            assert sym not in loop._positions  # flat, no new short opened
            store.close()


def _ohlcv_table(n: int, close: float = 100.0) -> pa.Table:
    """A flat n-bar OHLCV table for buffer pre-seeding (no symbol column)."""
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    ts = [base + timedelta(minutes=i) for i in range(n)]
    return pa.table(
        {"ts": ts, "open": [close] * n, "high": [close] * n, "low": [close] * n,
         "close": [close] * n, "volume": [100.0] * n},
        schema=_SCHEMA,
    )


class TestPrewarm:
    """History pre-seeding seeds the buffer without firing orders, so the first
    live bar trades immediately against current (not stale) prices."""

    @pytest.mark.asyncio
    async def test_prewarm_seeds_buffer_without_orders(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=0.40, tp_price=0.513)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            seeded = loop.prewarm("XRP-USDT-PERP", _ohlcv_table(40, 100.0))
            assert seeded == 40
            assert len(loop._bar_buffer["XRP-USDT-PERP"]) == 40
            # tick_count advanced past warmup (mock warmup_bars=1 -> 2).
            assert loop.tick_count == 2
            # No orders placed during prewarm (history never reaches the path).
            assert broker.orders == []
            store.close()

    @pytest.mark.asyncio
    async def test_first_live_bar_trades_after_prewarm(self) -> None:
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=0.40, tp_price=0.513)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            loop.prewarm("XRP-USDT-PERP", _ohlcv_table(40, 100.0))
            # Without prewarm, the first tick is a warmup skip. With the buffer
            # pre-seeded, the FIRST live bar trades immediately.
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=0))
            assert len(broker.orders) == 1
            assert broker.orders[0].sl is not None  # TP/SL attached
            store.close()

    @pytest.mark.asyncio
    async def test_large_buffer_reconstructs_and_trades(self) -> None:
        """Regression: a buffer longer than 60 bars must still reconstruct to a
        valid table (not None) and let the strategy fire. The prewarm seeds 90
        bars (the live runner's history depth); ``_buffer_to_table`` previously
        built ``datetime(2026,1,1,0,i)`` which raised ValueError for i>=60,
        was swallowed, and forced every tick to the neutral fallback — zero
        live trades despite valid signals."""
        cfg = BankrollConfig(start=10.0, leverage=10.0, allocation=1.0,
                             stop_at=100.0, milestones=(), risk_per_trade=0.025,
                             rr_ratio=1.3, max_drawdown=0.30)
        broker = _Broker()
        strategy = _ScalpStrategy(direction=1.0, sl_price=0.40, tp_price=0.513)
        with TemporaryDirectory() as tmp:
            store = LiveStore(Path(tmp) / "s.db")
            loop = TradingLoop(
                config=_config(), broker=broker, strategy=strategy,
                guardian=Guardian(), reconciler=Reconciler(),
                store=store, feed=object(), bankroll=cfg, attach_stops=True,
            )
            loop.prewarm("XRP-USDT-PERP", _ohlcv_table(90, 100.0))
            # The reconstructed table must exist and carry all 90 bars.
            table = loop._buffer_to_table("XRP-USDT-PERP")
            assert table is not None
            assert table.num_rows == 90
            # And the first live tick must actually trade (strategy called, not
            # the neutral fallback that a None table would trigger).
            await loop._process_tick(_bar("XRP-USDT-PERP", 0.5, minute=0))
            assert len(broker.orders) == 1
            store.close()
