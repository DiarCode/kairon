"""Vectorized scalping backtest — an independent sim, pure-function based.

Per the scalping-edge-enhancement plan (Phase 1.2), this is **not** an
orchestrator replay. It iterates historical bars, calls the strategy's
``predict`` on a rolling buffer per bar, then the *shared* pure functions in
:mod:`kairon.live.pure_fns` (``risk_size_qty``, ``post_rounding_guard``,
``stop_exit_price``) to size, gate, and exit — the exact same functions the live
:class:`~kairon.live.orchestrator.TradingLoop` uses. Drift between sim and live
is impossible by construction: there is one implementation of each primitive.

It records per-trade outcomes (entry/exit, SL/TP hits, R:R, fees, net PnL) and a
mark-to-market equity curve, compounding the synthetic bankroll on each close.
No network, no async, no broker — deterministic and unit-testable.

Short-tilted: supports ``Side.LONG`` and ``Side.SHORT`` (the existing
:mod:`kairon.backtest.engine` is long/flat only by design). Reuses the existing
:class:`~kairon.backtest.cost.CostModel` / :class:`~kairon.backtest.position.Side`
so cost fidelity is shared with the Phase 1.3 fidelity gate.

Entry/exit semantics mirror the live orchestrator:

* **Entry** at the *closed* bar's close, sized by fixed-fractional risk
  (``risk_per_trade * bankroll / sl_distance``) capped by the leverage notional,
  then passed through ``post_rounding_guard`` (min-lot floor, risk-cap clamp).
* **Stop-level exit**: when a subsequent bar's high/low crosses the attached
  ``sl_price``/``tp_price``, the position exits at the *stop level* (via
  ``stop_exit_price``), not the crossing price — matching the exchange-side
  attached-stop fill. If both sides are crossed in one bar, the SL is assumed to
  have hit first (the conservative/worst-case ordering for a scalp where the SL
  is closer than the TP).
* **Flip-to-flat**: when ``attach_stops`` and the signal flips against an open
  position, close to flat at the current close rather than reversing.
* **Halts**: 30% bankroll peak-to-trough drawdown and the optional ``stop_at``
  target both halt trading cleanly (open position marked-to-close).

The strategy stores ``sl_price``/``tp_price`` in ``last_indicator_snapshot``; the
engine reads them from there (they are computed identically to
:func:`kairon.live.pure_fns.atr_sl_tp`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pyarrow as pa

from kairon.backtest.cost import DEFAULT_CRYPTO_COSTS, CostModel
from kairon.backtest.position import Side
from kairon.live.pure_fns import post_rounding_guard, risk_size_qty, stop_exit_price

__all__ = [
    "ScalpBacktestConfig",
    "ScalpBacktestResult",
    "ScalpSkip",
    "ScalpTrade",
    "run_scalp_backtest",
]


# ---------------------------------------------------------------------------
# Setup tagging — derived from the strategy's justifications so the engine can
# bucket trades by setup without the strategy exposing an explicit setup id.
# (Phase 2 adds an explicit setup id to the strategy snapshot; this keeps the
# engine working against the current strategy.)
# ---------------------------------------------------------------------------
_SETUP_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("momentum trend-following short", "momentum_short"),
    ("momentum trend-following long", "momentum_long"),
    ("breakdown below lower bollinger", "breakdown"),
    ("breakout above upper bollinger", "breakout"),
    ("overbought mean-reversion short", "mr_short"),
    ("oversold mean-reversion long", "mr_long"),
)


def _setup_id_from_justifications(justifications: tuple[str, ...]) -> str:
    blob = " ".join(j.lower() for j in justifications)
    for needle, tag in _SETUP_KEYWORDS:
        if needle in blob:
            return tag
    return "unknown"


def _resolve_setup_id(snap: dict, justifications: tuple[str, ...]) -> str:
    """Prefer the strategy snapshot's explicit ``setup_id``; fall back to the
    justification-derived tag (for strategies/doubles that don't tag setups)."""
    explicit = snap.get("setup_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    return _setup_id_from_justifications(justifications)


# ---------------------------------------------------------------------------
# Config + records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScalpBacktestConfig:
    """Configuration for one scalping backtest run.

    Defaults mirror the live scalping runner's medium-risk profile: $10 start,
    10x leverage, 2.5% risk/trade, 1.3 R:R, 4% max stop, 30% drawdown halt.
    """

    bankroll_start: float = 10.0
    leverage: float = 10.0
    allocation: float = 1.0
    risk_per_trade: float = 0.025
    rr_ratio: float = 1.3
    max_sl_pct: float = 0.04
    atr_sl_mult: float = 1.0
    # Broker lot constraints for the symbol (0 disables the min-lot guard).
    min_qty: float = 0.0
    qty_step: float = 0.0
    # Rolling buffer width passed to ``predict``. Wide enough for the indicators
    # to stabilize (MACD-26/9, ADX-14, BB-20, Stoch-14) yet small enough that the
    # per-bar predict is cheap. The live orchestrator passes a longer live
    # buffer, but a 200-bar warm window reproduces indicator values to within
    # rounding for these parameter sets.
    buffer_bars: int = 200
    # Risk-cap gate (shared semantics with the live orchestrator).
    allow_min_lot_overshoot: bool = False
    risk_cap_tol: float = 0.10
    enforce_risk_cap: bool = True
    # Stop/exit behaviour.
    attach_stops: bool = True
    flip_to_flat: bool = True
    # Halts.
    max_drawdown: float | None = 0.30
    stop_at: float | None = None
    # Bars of no-new-entry cooldown after a close (0 disables).
    cooldown_bars: int = 0
    # Cost model. DEFAULT_CRYPTO_COSTS is 0.10% taker round-trip (Bybit default).
    cost: CostModel = field(default_factory=lambda: DEFAULT_CRYPTO_COSTS)

    def __post_init__(self) -> None:
        if self.bankroll_start <= 0:
            msg = f"bankroll_start must be > 0, got {self.bankroll_start}"
            raise ValueError(msg)
        if self.leverage <= 0:
            msg = f"leverage must be > 0, got {self.leverage}"
            raise ValueError(msg)
        if not 0.0 < self.allocation <= 1.0:
            msg = f"allocation must be in (0, 1], got {self.allocation}"
            raise ValueError(msg)
        if self.risk_per_trade < 0:
            msg = f"risk_per_trade must be >= 0, got {self.risk_per_trade}"
            raise ValueError(msg)
        if self.rr_ratio <= 0:
            msg = f"rr_ratio must be > 0, got {self.rr_ratio}"
            raise ValueError(msg)
        if self.max_sl_pct <= 0:
            msg = f"max_sl_pct must be > 0, got {self.max_sl_pct}"
            raise ValueError(msg)
        if self.buffer_bars < 1:
            msg = f"buffer_bars must be >= 1, got {self.buffer_bars}"
            raise ValueError(msg)
        if self.max_drawdown is not None and not 0.0 < self.max_drawdown <= 1.0:
            msg = f"max_drawdown must be in (0, 1] or None, got {self.max_drawdown}"
            raise ValueError(msg)
        if self.stop_at is not None and self.stop_at <= 0:
            msg = f"stop_at must be > 0 or None, got {self.stop_at}"
            raise ValueError(msg)
        if self.cooldown_bars < 0:
            msg = f"cooldown_bars must be >= 0, got {self.cooldown_bars}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ScalpSkip:
    """A setup that fired but was skipped before entry (min-lot / risk cap)."""

    ts: datetime
    side: Side
    close: float
    sl_price: float
    tp_price: float
    sl_distance: float
    intended_qty: float
    reason: str  # "below_min_lot" | "risk_cap_breach_overshoot" | "risk_cap_breach"
    setup_id: str
    justifications: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class ScalpTrade:
    """A single round-trip scalping trade."""

    symbol: str
    side: Side
    setup_id: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    qty: float
    sl_price: float
    tp_price: float
    sl_distance: float
    hit_sl: bool
    hit_tp: bool
    flip_close: bool  # closed by signal-flip / end-of-data, not a stop trigger
    gross_pnl: float
    fees: float
    net_pnl: float
    entry_bankroll: float
    exit_bankroll: float
    duration_bars: int
    confidence: float
    justifications: tuple[str, ...]

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def realized_rr(self) -> float:
        """Realized reward:risk — net PnL in units of the risked amount.

        Guarded against numerical explosion: when ``sl_distance`` is tiny
        relative to price (a degenerate stop), the risked amount ``qty *
        sl_distance`` can be near zero and the ratio blows up to absurd values.
        Require the stop to be at least 1bp of the entry price for the ratio to
        be meaningful; otherwise return 0.0.
        """
        if self.sl_distance <= 0 or self.qty <= 0 or self.entry_price <= 0:
            return 0.0
        if self.sl_distance < self.entry_price * 1e-4:
            return 0.0
        risked = self.qty * self.sl_distance
        return self.net_pnl / risked if risked > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ScalpBacktestResult:
    """Output of a scalping backtest run."""

    symbol: str
    trades: tuple[ScalpTrade, ...]
    skips: tuple[ScalpSkip, ...]
    equity_curve: tuple[float, ...]
    timestamps: tuple[datetime, ...]
    config: ScalpBacktestConfig
    halted: bool
    halt_reason: str | None
    peak_bankroll: float

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_skips(self) -> int:
        return len(self.skips)

    @property
    def final_bankroll(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.config.bankroll_start

    @property
    def total_pnl(self) -> float:
        return float(sum(t.net_pnl for t in self.trades))

    @property
    def total_fees(self) -> float:
        return float(sum(t.fees for t in self.trades))

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if not t.flip_close or t.hit_sl or t.hit_tp]
        # All recorded trades are closed; count wins over the lot.
        if not self.trades:
            return float("nan")
        wins = sum(1 for t in self.trades if t.is_win)
        return wins / len(self.trades)

    @property
    def n_hit_tp(self) -> int:
        return sum(1 for t in self.trades if t.hit_tp)

    @property
    def n_hit_sl(self) -> int:
        return sum(1 for t in self.trades if t.hit_sl)

    @property
    def n_flip_close(self) -> int:
        return sum(1 for t in self.trades if t.flip_close and not t.hit_sl and not t.hit_tp)

    @property
    def avg_rr(self) -> float:
        if not self.trades:
            return float("nan")
        return float(sum(t.realized_rr for t in self.trades) / len(self.trades))

    @property
    def expectancy(self) -> float:
        """Average net PnL per trade."""
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        """Peak-to-trough drawdown of the equity curve as a fraction."""
        peak = -float("inf")
        max_dd = 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak)
        return max_dd

    @property
    def bankroll_multiple(self) -> float:
        return self.final_bankroll / self.config.bankroll_start

    def to_table(self) -> pa.Table:
        """Serialize trades to a ``pa.Table`` for parquet / analysis logging."""
        return pa.table(
            {
                "symbol": [t.symbol for t in self.trades],
                "side": [t.side.value for t in self.trades],
                "setup_id": [t.setup_id for t in self.trades],
                "entry_ts": pa.array([t.entry_ts for t in self.trades], type=pa.timestamp("us", tz="UTC")),
                "exit_ts": pa.array([t.exit_ts for t in self.trades], type=pa.timestamp("us", tz="UTC")),
                "entry_price": [t.entry_price for t in self.trades],
                "exit_price": [t.exit_price for t in self.trades],
                "qty": [t.qty for t in self.trades],
                "sl_price": [t.sl_price for t in self.trades],
                "tp_price": [t.tp_price for t in self.trades],
                "hit_sl": [t.hit_sl for t in self.trades],
                "hit_tp": [t.hit_tp for t in self.trades],
                "flip_close": [t.flip_close for t in self.trades],
                "net_pnl": [t.net_pnl for t in self.trades],
                "realized_rr": [t.realized_rr for t in self.trades],
                "confidence": [t.confidence for t in self.trades],
            }
        )


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def run_scalp_backtest(
    *,
    bars: pa.Table,
    strategy: Any,
    symbol: str,
    config: ScalpBacktestConfig | None = None,
) -> ScalpBacktestResult:
    """Run a scalping backtest over ``bars`` (OHLCV_SCHEMA, ascending by ts).

    Parameters
    ----------
    bars
        OHLCV table (``kairon.data.io.OHLCV_SCHEMA``), sorted ascending by ``ts``.
    strategy
        A ``SignalStrategy`` with ``predict(bars, symbol) -> LivePrediction`` and
        ``last_indicator_snapshot`` exposing ``sl_price``/``tp_price``.
    symbol
        Symbol label for the emitted trades.
    config
        Run configuration; defaults to the medium-risk scalping profile.
    """
    cfg = config or ScalpBacktestConfig()
    n = bars.num_rows
    ts_col = bars.column("ts").to_pylist()
    high_col = bars.column("high").to_pylist()
    low_col = bars.column("low").to_pylist()
    close_col = bars.column("close").to_pylist()

    trades: list[ScalpTrade] = []
    skips: list[ScalpSkip] = []
    equity_curve: list[float] = []
    timestamps: list[datetime] = []

    bankroll = cfg.bankroll_start
    peak_bankroll = bankroll

    # Open position state (None when flat).
    pos: dict[str, Any] | None = None
    cooldown_until = -1  # bar index before which no new entry
    halted = False
    halt_reason: str | None = None

    warmup = max(strategy.warmup_bars, cfg.buffer_bars)

    def _snapshot_sl_tp(snap: dict[str, Any], close: float) -> tuple[float, float, float]:
        sl = snap.get("sl_price")
        tp = snap.get("tp_price")
        if sl is None or tp is None:
            return 0.0, 0.0, 0.0
        sl_f = float(sl)
        tp_f = float(tp)
        sl_distance = abs(sl_f - close)
        return sl_f, tp_f, sl_distance

    for i in range(n):
        ts = ts_col[i]
        close = float(close_col[i])
        high = float(high_col[i])
        low = float(low_col[i])

        # ---- Mark-to-market equity for this bar ----
        if pos is not None:
            unreal = _unrealized_pnl(pos["side"], pos["qty"], pos["entry_price"], close)
            equity = bankroll + unreal
        else:
            equity = bankroll
        equity_curve.append(equity)
        timestamps.append(ts)

        # ---- Manage an open position first (stops/flip) ----
        if pos is not None:
            side = pos["side"]
            sl_price = pos["sl_price"]
            tp_price = pos["tp_price"]
            long_ = side is Side.LONG
            if long_:
                sl_hit = low <= sl_price
                tp_hit = high >= tp_price
            else:
                sl_hit = high >= sl_price
                tp_hit = low <= tp_price
            # If both crossed in one bar, assume the (closer) SL hit first — the
            # conservative worst case for a scalp.
            hit_sl = sl_hit
            hit_tp = tp_hit and not sl_hit
            if hit_sl or hit_tp:
                exit_price = stop_exit_price(
                    side_is_long=long_, sl_price=sl_price, tp_price=tp_price,
                    hit_sl=hit_sl, hit_tp=hit_tp,
                )
                _close_trade(
                    pos, trades, ts, exit_price, cfg, bankroll,
                    hit_sl=hit_sl, hit_tp=hit_tp, flip_close=False, end_bar=i,
                )
                bankroll += trades[-1].net_pnl
                peak_bankroll = max(peak_bankroll, bankroll)
                pos = None
                cooldown_until = i + cfg.cooldown_bars
            elif cfg.flip_to_flat and cfg.attach_stops:
                # Need the current signal to detect a flip; predict below sets it.
                pred = _predict_at(strategy, bars, i, warmup, symbol)
                direction = float(pred.direction)
                flipped = (
                    direction != 0.0
                    and ((direction > 0) != long_)
                )
                if flipped:
                    _close_trade(
                        pos, trades, ts, close, cfg, bankroll,
                        hit_sl=False, hit_tp=False, flip_close=True, end_bar=i,
                    )
                    bankroll += trades[-1].net_pnl
                    peak_bankroll = max(peak_bankroll, bankroll)
                    pos = None
                    cooldown_until = i + cfg.cooldown_bars
            # else: hold through this bar; equity already marked.

            # Halts are evaluated after a close (below). If still open and a halt
            # condition is met on equity, force-close at the current close.
            if pos is not None and _should_halt(equity, bankroll, peak_bankroll, cfg):
                reason = _halt_reason(equity, bankroll, peak_bankroll, cfg)
                _close_trade(
                    pos, trades, ts, close, cfg, bankroll,
                    hit_sl=False, hit_tp=False, flip_close=True, end_bar=i,
                )
                bankroll += trades[-1].net_pnl
                peak_bankroll = max(peak_bankroll, bankroll)
                pos = None
                halted = True
                halt_reason = reason
            continue  # one action per bar: manage the open position

        # ---- Flat: check halts, then look for an entry ----
        if halted:
            continue
        if _should_halt(equity, bankroll, peak_bankroll, cfg):
            halted = True
            halt_reason = _halt_reason(equity, bankroll, peak_bankroll, cfg)
            continue
        if i < cooldown_until:
            continue
        if i + 1 < warmup:
            continue  # indicators not warm yet

        pred = _predict_at(strategy, bars, i, warmup, symbol)
        direction = float(pred.direction)
        if direction == 0.0:
            continue
        snap = _attr_or_call(strategy, "last_indicator_snapshot") or {}
        sl_price, tp_price, sl_distance = _snapshot_sl_tp(snap, close)
        if sl_distance <= 0:
            continue

        side = Side.LONG if direction > 0 else Side.SHORT
        notional_cap_qty = (bankroll * cfg.leverage * cfg.allocation) / close if close > 0 else 0.0
        raw_qty = risk_size_qty(
            bankroll=bankroll, risk_per_trade=cfg.risk_per_trade,
            sl_distance=sl_distance, notional_cap_qty=notional_cap_qty,
        )
        effective_qty, breach = post_rounding_guard(
            raw_qty=raw_qty, min_qty=cfg.min_qty, sl_distance=sl_distance,
            bankroll=bankroll, risk_per_trade=cfg.risk_per_trade, tol=cfg.risk_cap_tol,
            enforce_risk_cap=cfg.enforce_risk_cap,
            allow_min_lot_overshoot=cfg.allow_min_lot_overshoot,
        )
        if breach is not None:
            justs = _attr_or_call(strategy, "last_justifications") or ()
            skips.append(ScalpSkip(
                ts=ts, side=side, close=close, sl_price=sl_price, tp_price=tp_price,
                sl_distance=sl_distance, intended_qty=raw_qty, reason=breach,
                setup_id=_resolve_setup_id(snap, justs),
                justifications=justs,
                confidence=float(pred.confidence),
            ))
            continue
        qty = _round_lot(effective_qty, cfg.qty_step)
        if qty <= 0:
            continue

        entry_cost = cfg.cost.total_cost(qty * close, "entry")
        justs = _attr_or_call(strategy, "last_justifications") or ()
        pos = {
            "symbol": symbol,
            "side": side,
            "entry_ts": ts,
            "entry_price": close,
            "qty": qty,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_distance": sl_distance,
            "entry_bar": i,
            "entry_cost": entry_cost,
            "entry_bankroll": bankroll,
            "confidence": float(pred.confidence),
            "justifications": justs,
            "setup_id": _resolve_setup_id(snap, justs),
        }

    # ---- Close any still-open position at the last bar ----
    if pos is not None:
        last_ts = ts_col[-1]
        last_close = float(close_col[-1])
        _close_trade(
            pos, trades, last_ts, last_close, cfg, bankroll,
            hit_sl=False, hit_tp=False, flip_close=True, end_bar=n - 1,
        )
        bankroll += trades[-1].net_pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        equity_curve[-1] = bankroll  # final bar settles to realized bankroll

    return ScalpBacktestResult(
        symbol=symbol,
        trades=tuple(trades),
        skips=tuple(skips),
        equity_curve=tuple(equity_curve),
        timestamps=tuple(timestamps),
        config=cfg,
        halted=halted,
        halt_reason=halt_reason,
        peak_bankroll=peak_bankroll,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _attr_or_call(strategy: Any, name: str) -> Any:
    """Read a strategy attribute that may be a property or a zero-arg method.

    The real ``ScalpingStrategy`` exposes ``last_indicator_snapshot`` /
    ``last_justifications`` / ``last_confluence_scores`` as ``@property``
    descriptors; some test doubles expose them as methods. Handle both so the
    engine is robust to either contract (mirrors the orchestrator's
    ``hasattr`` guard, which reads them as properties).
    """
    value = getattr(strategy, name, None)
    if value is None:
        return None
    return value() if callable(value) else value


def _predict_at(strategy: Any, bars: pa.Table, i: int, warmup: int, symbol: str) -> Any:
    """Call ``strategy.predict`` on the rolling window ending at bar ``i``."""
    start = max(0, i + 1 - warmup)
    length = i + 1 - start
    window = bars.slice(start, length)
    return strategy.predict(window, symbol)


def _unrealized_pnl(side: Side, qty: float, entry: float, price: float) -> float:
    if side is Side.LONG:
        return qty * (price - entry)
    return qty * (entry - price)


def _close_trade(
    pos: dict[str, Any],
    trades: list[ScalpTrade],
    exit_ts: datetime,
    exit_price: float,
    cfg: ScalpBacktestConfig,
    bankroll_before: float,
    *,
    hit_sl: bool,
    hit_tp: bool,
    flip_close: bool,
    end_bar: int,
) -> None:
    side = pos["side"]
    qty = pos["qty"]
    entry_price = pos["entry_price"]
    if side is Side.LONG:
        gross = qty * (exit_price - entry_price)
    else:
        gross = qty * (entry_price - exit_price)
    exit_cost = cfg.cost.total_cost(qty * exit_price, "exit")
    fees = pos["entry_cost"] + exit_cost
    net = gross - fees
    exit_bankroll = bankroll_before + net
    trades.append(ScalpTrade(
        symbol=pos["symbol"],
        side=side,
        setup_id=pos["setup_id"],
        entry_ts=pos["entry_ts"],
        exit_ts=exit_ts,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=qty,
        sl_price=pos["sl_price"],
        tp_price=pos["tp_price"],
        sl_distance=pos["sl_distance"],
        hit_sl=hit_sl,
        hit_tp=hit_tp,
        flip_close=flip_close,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        entry_bankroll=pos["entry_bankroll"],
        exit_bankroll=exit_bankroll,
        duration_bars=end_bar - pos["entry_bar"],
        confidence=pos["confidence"],
        justifications=pos["justifications"],
    ))


def _round_lot(qty: float, qty_step: float) -> float:
    if qty_step <= 0:
        return qty
    return float(qty_step * round(qty / qty_step))


def _should_halt(equity: float, bankroll: float, peak: float, cfg: ScalpBacktestConfig) -> bool:
    reason = _halt_reason(equity, bankroll, peak, cfg)
    return reason is not None


def _halt_reason(
    equity: float, bankroll: float, peak: float, cfg: ScalpBacktestConfig,
) -> str | None:
    if cfg.stop_at is not None and bankroll >= cfg.stop_at:
        return "target_reached"
    if cfg.max_drawdown is not None and peak > 0:
        if (peak - bankroll) / peak >= cfg.max_drawdown:
            return "max_drawdown"
    return None
