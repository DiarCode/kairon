"""Tests for the paper trading engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kairon.backtest.cost import CostModel
from kairon.paper import (
    Fill,
    Order,
    OrderSide,
    OrderType,
    PaperTrader,
    PaperTraderConfig,
    PortfolioState,
    run_paper_scenario,
)


def _cfg(**kw) -> PaperTraderConfig:
    return PaperTraderConfig(cost=CostModel(commission_bps=0, slippage_bps=0, half_spread_bps=0), **kw)


def _order(symbol: str, side: OrderSide, size: float, **kw) -> Order:
    return Order(symbol=symbol, side=side, size=size, **kw)


def test_config_validates() -> None:
    with pytest.raises(ValueError):
        PaperTraderConfig(initial_cash=0)
    with pytest.raises(ValueError):
        PaperTraderConfig(initial_cash=-100)
    with pytest.raises(ValueError):
        PaperTraderConfig(max_position_size=0)


def test_on_price_rejects_non_positive() -> None:
    t = PaperTrader(_cfg())
    with pytest.raises(ValueError):
        t.on_price("BTC/USDT", 0)
    with pytest.raises(ValueError):
        t.on_price("BTC/USDT", -1)


def test_submit_market_buy_fills_at_price() -> None:
    t = PaperTrader(_cfg(initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    fill = t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    assert fill is not None
    assert fill.price == 50_000.0
    assert fill.size == 0.1
    snap = t.snapshot()
    assert snap.cash == pytest.approx(10_000 - 0.1 * 50_000)  # zero cost model
    assert snap.equity == pytest.approx(snap.cash)  # no PnL at mark
    assert snap.n_open == 1
    assert snap.positions[0].side.value == "long"


def test_long_close_realises_pnl() -> None:
    t = PaperTrader(_cfg(initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    t.on_price("BTC/USDT", 51_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.SELL, 0.1), fill_price=51_000.0)
    snap = t.snapshot()
    assert snap.realised_pnl == pytest.approx(100.0)
    assert snap.n_open == 0
    assert snap.cash == pytest.approx(10_000 + 100.0)


def test_long_loss_realises_negative_pnl() -> None:
    t = PaperTrader(_cfg(initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    t.on_price("BTC/USDT", 49_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.SELL, 0.1), fill_price=49_000.0)
    snap = t.snapshot()
    assert snap.realised_pnl == pytest.approx(-100.0)


def test_limit_order_unfilled_when_mark_above_limit() -> None:
    t = PaperTrader(_cfg())
    t.on_price("BTC/USDT", 50_000.0)
    fill = t.submit_order(
        _order("BTC/USDT", OrderSide.BUY, 0.1, order_type=OrderType.LIMIT, limit_price=49_500.0),
        fill_price=50_000.0,
    )
    assert fill is None
    assert t.n_fills == 0


def test_limit_order_fills_when_mark_crosses_limit() -> None:
    t = PaperTrader(_cfg())
    t.on_price("BTC/USDT", 49_000.0)
    fill = t.submit_order(
        _order("BTC/USDT", OrderSide.BUY, 0.1, order_type=OrderType.LIMIT, limit_price=49_500.0),
        fill_price=49_500.0,
    )
    assert fill is not None
    assert fill.price == 49_500.0


def test_short_selling_rejected_by_default() -> None:
    t = PaperTrader(_cfg(allow_short=False))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    t.on_price("BTC/USDT", 51_000.0)
    # Selling 0.2 closes the 0.1 long and *tries* to add a 0.1 short;
    # that should be rejected.
    with pytest.raises(ValueError, match="short"):
        t.submit_order(_order("BTC/USDT", OrderSide.SELL, 0.2), fill_price=51_000.0)
    # After the rejection the position is gone (it was closed) but no
    # new short position was opened. State should be: no positions, cash
    # reflects the close proceeds minus costs.
    snap = t.snapshot()
    assert snap.n_open == 0


def test_short_selling_allowed() -> None:
    t = PaperTrader(_cfg(allow_short=True, initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.SELL, 0.1), fill_price=50_000.0)
    snap = t.snapshot()
    assert snap.positions[0].side.value == "short"
    t.on_price("BTC/USDT", 49_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=49_000.0)
    snap = t.snapshot()
    assert snap.realised_pnl == pytest.approx(100.0)


def test_max_position_size_enforced() -> None:
    t = PaperTrader(_cfg(max_position_size=0.05))
    t.on_price("BTC/USDT", 50_000.0)
    with pytest.raises(ValueError, match="max_position_size"):
        t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)


def test_costs_applied() -> None:
    t = PaperTrader(PaperTraderConfig(cost=CostModel(commission_bps=10, slippage_bps=0, half_spread_bps=0)))
    t.on_price("BTC/USDT", 50_000.0)
    fill = t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    assert fill is not None
    assert fill.costs == pytest.approx(0.1 * 50_000 * 10 / 1e4)  # 5.0


def test_reconcile_detects_drift() -> None:
    t = PaperTrader(_cfg(initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    drift = t.reconcile(external_cash=10_000, external_positions={"BTC/USDT": 0.05})
    assert drift["drift_positions"]["BTC/USDT"]["internal"] == pytest.approx(0.1)
    assert drift["drift_positions"]["BTC/USDT"]["external"] == pytest.approx(0.05)


def test_reconcile_no_drift() -> None:
    t = PaperTrader(_cfg(initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    drift = t.reconcile(external_cash=5_000, external_positions={"BTC/USDT": 0.1})
    # external_cash differs but no position drift
    assert "BTC/USDT" not in drift["drift_positions"]


def test_events_logged() -> None:
    t = PaperTrader(_cfg())
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    evs = t.events()
    kinds = [e["kind"] for e in evs]
    assert "mark" in kinds
    assert "order" in kinds
    assert "fill" in kinds
    ids = [e["event_id"] for e in evs]
    assert ids == sorted(ids)


def test_run_paper_scenario() -> None:
    n = 50
    ts = np.array([datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)])
    prices = 50_000 + np.cumsum(np.random.default_rng(0).normal(0, 10, n))
    signals = np.zeros(n)
    signals[10:20] = 1
    signals[30:40] = -1
    trader, snap = run_paper_scenario(
        symbol="BTC/USDT",
        timestamps=ts,
        prices=prices,
        signals=signals,
        size_per_trade=0.01,
    )
    assert isinstance(trader, PaperTrader)
    assert isinstance(snap, PortfolioState)
    assert trader.n_fills > 0


def test_run_paper_scenario_validates_lengths() -> None:
    ts = np.array([datetime(2026, 1, 1)])
    prices = np.array([1.0, 2.0])
    signals = np.array([1.0])
    with pytest.raises(ValueError):
        run_paper_scenario(symbol="X", timestamps=ts, prices=prices, signals=signals)


def test_unrealised_pnl_marks_to_market() -> None:
    t = PaperTrader(_cfg(initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    # After entry: cash = 10_000 - 5_000 = 5_000
    t.on_price("BTC/USDT", 50_500.0)
    snap = t.snapshot()
    assert snap.cash == pytest.approx(5_000.0)
    assert snap.unrealised_pnl == pytest.approx(50.0)
    assert snap.equity == pytest.approx(5_050.0)


def test_partial_close_keeps_position() -> None:
    t = PaperTrader(_cfg(initial_cash=10_000))
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.2), fill_price=50_000.0)
    t.on_price("BTC/USDT", 51_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.SELL, 0.1), fill_price=51_000.0)
    snap = t.snapshot()
    assert snap.n_open == 1
    assert snap.positions[0].size == pytest.approx(0.1)
    assert snap.realised_pnl == pytest.approx(100.0)


def test_fills_returns_tuple() -> None:
    t = PaperTrader(_cfg())
    t.on_price("BTC/USDT", 50_000.0)
    t.submit_order(_order("BTC/USDT", OrderSide.BUY, 0.1), fill_price=50_000.0)
    fills = t.fills()
    assert isinstance(fills, tuple)
    assert all(isinstance(f, Fill) for f in fills)


def test_public_api_imports() -> None:
    from kairon.paper import (  # noqa: F401
        Fill,
        Order,
        OrderSide,
        PaperTrader,
        PaperTraderConfig,
        PortfolioState,
    )
