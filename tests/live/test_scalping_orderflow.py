"""Strategy integration tests for the Phase 4b order-flow confluence nudge.

Isolates the confidence nudge by triggering a deterministic long breakout
signal with the setup matrix OFF (``setup_matrix=None``), so no regime /
exhaustion / calibration gating interferes — the only variable between runs is
the order-flow snapshot. Default ``use_orderflow=False`` must be byte-for-byte
identical to the legacy path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa

from kairon.data.io import OHLCV_SCHEMA
from kairon.live.orderflow import OrderFlowSnapshot
from kairon.live.strategy import ScalpingStrategy


def _breakout_bars(n: int = 60) -> pa.Table:
    """A flat series that spikes on the last bar -> a deterministic long breakout.

    55 bars flat at 100 (volume 1.0) seed a narrow Bollinger band; the last 5
    bars ramp to 110 with a volume surge, so ``close >= bb_upper`` and
    ``volume_vs_avg >= volume_surge_mult`` both fire -> ``breakout`` long signal.
    """
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    closes = [100.0] * (n - 5) + [102.0, 104.0, 106.0, 108.0, 110.0]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    volumes = [1.0] * (n - 1) + [5.0]  # last bar surges -> volume_surge True
    return pa.table(
        {"ts": ts, "open": closes, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        schema=OHLCV_SCHEMA,
    )


def _snapshot(imbalance: float) -> OrderFlowSnapshot:
    return OrderFlowSnapshot(
        mid=110.0, spread_pct=0.001, imbalance=imbalance,
        depth_ratio=1.0, bid_depth=1.0, ask_depth=1.0,
        best_bid=109.5, best_ask=110.5,
    )


class TestOrderflowConfidenceNudge:
    def _predict(self, strat: ScalpingStrategy) -> float:
        bars = _breakout_bars()
        pred = strat.predict(bars, "SOL-USDT-PERP")
        # Confirm the series actually fires a long (else the test is moot).
        assert pred.direction == 1.0, "fixture must fire a long breakout signal"
        return pred.confidence

    def test_use_orderflow_off_is_byte_identical_regardless_of_snapshot(self) -> None:
        # Default: use_orderflow=False. Setting last_orderflow must NOT change
        # confidence — the snapshot is ignored entirely.
        base = ScalpingStrategy()  # setup_matrix=None, use_orderflow=False
        c0 = self._predict(base)
        # Pollute the snapshot; the off-path must ignore it.
        base.last_orderflow = _snapshot(1.0)
        c1 = self._predict(base)
        assert c1 == c0

    def test_aligned_snapshot_raises_confidence(self) -> None:
        off = ScalpingStrategy()
        c_off = self._predict(off)
        on = ScalpingStrategy(use_orderflow=True)
        on.last_orderflow = _snapshot(1.0)  # bid-heavy -> align +1.0 for a long
        c_on = self._predict(on)
        assert c_on > c_off
        # The aligned justification is recorded.
        assert any("Order-flow supports" in j for j in on.last_justifications)

    def test_opposed_snapshot_lowers_confidence(self) -> None:
        off = ScalpingStrategy()
        c_off = self._predict(off)
        on = ScalpingStrategy(use_orderflow=True)
        on.last_orderflow = _snapshot(0.0)  # ask-heavy -> align -1.0 for a long
        c_on = self._predict(on)
        assert c_on < c_off
        assert any("Order-flow opposes" in j for j in on.last_justifications)

    def test_neutral_snapshot_is_noop(self) -> None:
        off = ScalpingStrategy()
        c_off = self._predict(off)
        on = ScalpingStrategy(use_orderflow=True)
        on.last_orderflow = _snapshot(0.5)  # balanced -> align 0.0 -> mult 1.0
        c_on = self._predict(on)
        assert c_on == c_off

    def test_none_snapshot_is_noop_even_when_use_orderflow_true(self) -> None:
        off = ScalpingStrategy()
        c_off = self._predict(off)
        on = ScalpingStrategy(use_orderflow=True)
        on.last_orderflow = None  # no book available (thin testnet book)
        c_on = self._predict(on)
        assert c_on == c_off

    def test_orderflow_fields_journaled_only_when_on(self) -> None:
        off = ScalpingStrategy()
        self._predict(off)
        snap_off = off.last_indicator_snapshot
        assert snap_off.get("of_imbalance") is None
        assert snap_off.get("of_spread_pct") is None

        on = ScalpingStrategy(use_orderflow=True)
        on.last_orderflow = _snapshot(0.8)
        self._predict(on)
        snap_on = on.last_indicator_snapshot
        assert snap_on["of_imbalance"] == 0.8
        assert snap_on["of_spread_pct"] == 0.001
        assert snap_on["of_depth_ratio"] == 1.0
