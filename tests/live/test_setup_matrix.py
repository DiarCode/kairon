"""Tests for the regime classifier + setup-selection matrix (Phase 2, no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.live.regime import (
    Regime,
    classify_regime,
    mean_reversion_allowed,
    trend_following_allowed,
)
from kairon.live.setup_matrix import LONG_ONLY, MEAN_REVERSION_ONLY, SetupMatrix
from kairon.live.strategy import ScalpingStrategy


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------
class TestClassifyRegime:
    def test_low_adx_narrow_bb_is_range(self) -> None:
        assert classify_regime(adx=15.0, bb_width_pct=0.02, ema_slope=0.0) is Regime.RANGE

    def test_high_adx_positive_slope_is_trend_up(self) -> None:
        assert classify_regime(adx=30.0, bb_width_pct=0.02, ema_slope=0.01) is Regime.TREND_UP

    def test_high_adx_negative_slope_is_trend_down(self) -> None:
        assert classify_regime(adx=30.0, bb_width_pct=0.02, ema_slope=-0.01) is Regime.TREND_DOWN

    def test_wide_bb_is_volatile_regardless_of_adx(self) -> None:
        assert classify_regime(adx=30.0, bb_width_pct=0.08, ema_slope=0.01) is Regime.VOLATILE

    def test_mid_adx_is_range(self) -> None:
        # Between range_adx(20) and trend_adx(25) -> treated as range.
        assert classify_regime(adx=22.0, bb_width_pct=0.02, ema_slope=0.01) is Regime.RANGE

    def test_mean_reversion_allowed_in_range_and_volatile(self) -> None:
        assert mean_reversion_allowed(Regime.RANGE) is True
        assert mean_reversion_allowed(Regime.VOLATILE) is True
        assert mean_reversion_allowed(Regime.TREND_UP) is False
        assert mean_reversion_allowed(Regime.TREND_DOWN) is False

    def test_trend_following_allowed_only_in_trends(self) -> None:
        assert trend_following_allowed(Regime.TREND_UP) is True
        assert trend_following_allowed(Regime.TREND_DOWN) is True
        assert trend_following_allowed(Regime.RANGE) is False
        assert trend_following_allowed(Regime.VOLATILE) is False


# ---------------------------------------------------------------------------
# Setup matrix
# ---------------------------------------------------------------------------
class TestSetupMatrix:
    def test_default_all_enabled_no_regime_gate(self) -> None:
        m = SetupMatrix()
        for sid in ("mr_short", "mr_long", "momentum_short", "momentum_long",
                    "breakdown", "breakout"):
            assert m.allowed(sid, Regime.TREND_UP) is True  # no gate -> always allowed

    def test_mean_reversion_only_kills_momentum_and_breakouts(self) -> None:
        m = MEAN_REVERSION_ONLY
        assert m.allowed("mr_short", Regime.RANGE) is True
        assert m.allowed("mr_long", Regime.RANGE) is True
        assert m.allowed("momentum_short", Regime.TREND_DOWN) is False
        assert m.allowed("momentum_long", Regime.TREND_UP) is False
        assert m.allowed("breakdown", Regime.RANGE) is False
        assert m.allowed("breakout", Regime.RANGE) is False

    def test_regime_gate_blocks_mr_in_trends(self) -> None:
        m = MEAN_REVERSION_ONLY
        assert m.allowed("mr_short", Regime.TREND_UP) is False
        assert m.allowed("mr_long", Regime.TREND_DOWN) is False
        # but allowed in a range
        assert m.allowed("mr_short", Regime.RANGE) is True

    def test_long_only_kills_mr_short_keeps_mr_long(self) -> None:
        # Phase 4 data-driven tightening: the universe backtest showed mr_short
        # is a universal loser on testnet, so LONG_ONLY kills it; mr_long (the
        # only edge) stays, and momentum/breakout stay killed.
        m = LONG_ONLY
        assert m.allowed("mr_short", Regime.RANGE) is False  # the losing side, killed
        assert m.allowed("mr_long", Regime.RANGE) is True    # the only edge, kept
        assert m.allowed("mr_long", Regime.VOLATILE) is True  # MR allowed in volatile
        assert m.allowed("mr_long", Regime.TREND_UP) is False  # regime gate still on
        assert m.allowed("momentum_long", Regime.TREND_UP) is False
        assert m.allowed("breakout", Regime.RANGE) is False

    def test_long_only_is_mr_subset_of_mean_reversion_only(self) -> None:
        # LONG_ONLY must allow everything MEAN_REVERSION_ONLY allows, minus the
        # mr_short side (i.e. it is a strict tightening, never a loosening).
        for sid in ("mr_short", "mr_long", "momentum_short", "momentum_long",
                    "breakdown", "breakout"):
            for regime in Regime:
                mr = MEAN_REVERSION_ONLY.allowed(sid, regime)
                lo = LONG_ONLY.allowed(sid, regime)
                if mr:
                    # If MR allows it, LONG_ONLY allows it iff it's not mr_short.
                    assert lo is (sid != "mr_short"), (sid, regime)
                else:
                    assert lo is False, (sid, regime)

    def test_unknown_setup_id_disallowed(self) -> None:
        m = SetupMatrix()
        assert m.allowed("nonsense", Regime.RANGE) is False


# ---------------------------------------------------------------------------
# Strategy gating wiring (smoke + contract)
# ---------------------------------------------------------------------------
def _oscillating_bars(n: int = 80, base: float = 100.0) -> pa.Table:
    """A mean-reverting price series (sine) warm enough for the strategy."""
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    closes = [base + 5.0 * np.sin(i / 5.0) for i in range(n)]
    return pa.table(
        {"ts": ts, "open": closes, "high": [c + 1 for c in closes],
         "low": [c - 1 for c in closes], "close": closes, "volume": [1.0] * n},
        schema=OHLCV_SCHEMA,
    )


class TestStrategyGatingWiring:
    def test_no_matrix_preserves_legacy_snapshot(self) -> None:
        # Without a matrix, the snapshot has setup_id=None and regime=None
        # (legacy behaviour) and predict does not crash.
        strat = ScalpingStrategy()
        bars = _oscillating_bars()
        strat.predict(bars, "SOL-USDT-PERP")
        snap = strat.last_indicator_snapshot
        assert "setup_id" in snap
        assert "regime" in snap
        assert snap["setup_id"] is None
        assert snap["regime"] is None

    def test_matrix_tags_regime_and_setup_id(self) -> None:
        # With a matrix, when a signal fires the snapshot records the regime
        # and (if not gated) the setup_id; predict does not crash.
        strat = ScalpingStrategy(setup_matrix=MEAN_REVERSION_ONLY)
        bars = _oscillating_bars()
        strat.predict(bars, "SOL-USDT-PERP")
        snap = strat.last_indicator_snapshot
        # regime is always classified when a matrix is set and a signal fired;
        # if no signal fired this bar, regime may be None — either is valid, the
        # contract is just that the keys exist and predict runs.
        assert "setup_id" in snap
        assert "regime" in snap

    def test_matrix_does_not_crash_on_flat_buffer(self) -> None:
        strat = ScalpingStrategy(setup_matrix=MEAN_REVERSION_ONLY)
        # A flat price series -> no signal; must not raise.
        ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(80)]
        flat = [100.0] * 80
        bars = pa.table(
            {"ts": ts, "open": flat, "high": flat, "low": flat, "close": flat,
             "volume": [1.0] * 80},
            schema=OHLCV_SCHEMA,
        )
        pred = strat.predict(bars, "SOL-USDT-PERP")
        assert pred.direction == 0.0
