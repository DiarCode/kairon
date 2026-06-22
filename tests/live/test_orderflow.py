"""Tests for the order-flow / microstructure feature module (Phase 4b, no network)."""

from __future__ import annotations

import math

from kairon.live.orderflow import (
    OrderFlowSnapshot,
    compute_orderflow,
    orderflow_alignment,
)


class TestComputeOrderflow:
    def test_balanced_book(self) -> None:
        # Symmetric book: imbalance 0.5, depth_ratio 1.0, spread from 100..101.
        bids = [[100.0, 10.0], [99.0, 5.0]]
        asks = [[101.0, 10.0], [102.0, 5.0]]
        snap = compute_orderflow(bids, asks, depth=5)
        assert snap is not None
        assert snap.mid == 100.5
        assert math.isclose(snap.spread_pct, 1.0 / 100.5, rel_tol=1e-9)
        assert math.isclose(snap.imbalance, 0.5, abs_tol=1e-9)
        assert math.isclose(snap.depth_ratio, 1.0, abs_tol=1e-9)
        assert snap.bid_depth == 15.0
        assert snap.ask_depth == 15.0
        assert snap.best_bid == 100.0
        assert snap.best_ask == 101.0

    def test_bid_heavy_book(self) -> None:
        bids = [[100.0, 30.0], [99.0, 20.0]]
        asks = [[101.0, 5.0], [102.0, 5.0]]
        snap = compute_orderflow(bids, asks)
        assert snap is not None
        assert snap.imbalance > 0.5  # bid-heavy
        assert snap.depth_ratio > 1.0
        # 50 / 10 = 5.0
        assert math.isclose(snap.depth_ratio, 5.0, rel_tol=1e-9)

    def test_ask_heavy_book(self) -> None:
        bids = [[100.0, 5.0]]
        asks = [[101.0, 25.0], [102.0, 25.0]]
        snap = compute_orderflow(bids, asks)
        assert snap is not None
        assert snap.imbalance < 0.5
        assert snap.depth_ratio < 1.0

    def test_empty_ask_saturates_bid_heavy(self) -> None:
        # No ask size at all -> ratio capped, imbalance -> 1.0 (fully bid-heavy).
        bids = [[100.0, 10.0]]
        asks = [[101.0, 0.0]]
        snap = compute_orderflow(bids, asks)
        assert snap is not None
        assert math.isclose(snap.imbalance, 1.0, abs_tol=1e-9)
        assert snap.depth_ratio == 50.0  # capped

    def test_depth_limits_levels_summed(self) -> None:
        # 10 levels each but depth=2 -> only top 2 summed.
        bids = [[100.0, 10.0]] * 10
        asks = [[101.0, 4.0]] * 10
        snap = compute_orderflow(bids, asks, depth=2)
        assert snap is not None
        assert snap.bid_depth == 20.0
        assert snap.ask_depth == 8.0

    def test_returns_none_on_empty_book(self) -> None:
        assert compute_orderflow([], [[101.0, 5.0]]) is None
        assert compute_orderflow([[100.0, 5.0]], []) is None

    def test_returns_none_on_crossed_or_zero_book(self) -> None:
        # Crossed book (bid > ask) is invalid.
        assert compute_orderflow([[101.0, 5.0]], [[100.0, 5.0]]) is None
        # Zero prices are unusable.
        assert compute_orderflow([[0.0, 5.0]], [[101.0, 5.0]]) is None

    def test_tolerates_garbage_levels(self) -> None:
        # Non-numeric / short levels must not crash; they are coerced to 0.
        bids = [["not-a-price", 5.0], [100.0, 10.0]]
        asks = [[101.0], [102.0, 8.0]]
        snap = compute_orderflow(bids, asks, depth=5)
        # First bid price coerced to 0 -> best_bid falls to 100.0 (second level).
        assert snap is not None
        assert snap.best_bid == 100.0
        assert snap.best_ask == 101.0

    def test_depth_ratio_capped_on_extreme_one_sidedness(self) -> None:
        bids = [[100.0, 1000.0]]
        asks = [[101.0, 0.0001]]
        snap = compute_orderflow(bids, asks)
        assert snap is not None
        assert snap.depth_ratio == 50.0  # capped, not 1e7


class TestOrderflowAlignment:
    def _snap(self, imbalance: float) -> OrderFlowSnapshot:
        return OrderFlowSnapshot(
            mid=100.0, spread_pct=0.01, imbalance=imbalance,
            depth_ratio=1.0, bid_depth=1.0, ask_depth=1.0,
            best_bid=99.5, best_ask=100.5,
        )

    def test_neutral_direction_is_zero(self) -> None:
        assert orderflow_alignment(self._snap(0.9), 0.0) == 0.0

    def test_long_aligned_with_bid_heavy(self) -> None:
        # imbalance 1.0 -> raw +1.0 -> long gets +1.0 (supportive).
        assert math.isclose(orderflow_alignment(self._snap(1.0), 1.0), 1.0, abs_tol=1e-9)
        # imbalance 0.5 -> 0.0 (neutral book, no support either way).
        assert math.isclose(orderflow_alignment(self._snap(0.5), 1.0), 0.0, abs_tol=1e-9)

    def test_long_opposed_by_ask_heavy(self) -> None:
        # imbalance 0.0 -> raw -1.0 -> long gets -1.0 (opposed).
        assert math.isclose(orderflow_alignment(self._snap(0.0), 1.0), -1.0, abs_tol=1e-9)

    def test_short_flipped(self) -> None:
        # For a short, ask-heavy (imbalance 0.0) is supportive -> +1.0.
        assert math.isclose(orderflow_alignment(self._snap(0.0), -1.0), 1.0, abs_tol=1e-9)
        # bid-heavy (imbalance 1.0) opposes a short -> -1.0.
        assert math.isclose(orderflow_alignment(self._snap(1.0), -1.0), -1.0, abs_tol=1e-9)
