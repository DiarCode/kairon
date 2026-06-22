"""TradingLoop: the orchestrator that wires feed → strategy → sizer → guardian → broker.

The :class:`TradingLoop` is the main runtime loop for live trading. On each
tick it:

1. Checks the kill switch (``LiveStore.is_halted()``).
2. Reads a closed bar from the candle feed.
3. Updates mark prices on PaperBroker (for simulation).
4. Runs the signal strategy (or model predictor) to get a prediction.
5. Computes position sizing via :func:`size_position_vol_aware`.
6. Diffs against current positions (direction-aware, supporting both long and short).
7. Runs the :class:`Guardian` risk checks (synchronous, blocking).
8. If no CRITICAL alerts, submits the order through the broker.
9. Persists the order and writes a heartbeat.

The loop is fully async and designed to run inside an ``asyncio`` event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa

from kairon.live.broker.base import (
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from kairon.live.broker.paper import PaperBroker
from kairon.live.config import BankrollConfig, LiveConfig
from kairon.live.drift_killswitch import DriftKillSwitch
from kairon.live.guardian import Guardian
from kairon.live.journal import IndicatorSnapshot, RiskSnapshot, TradeDecision
from kairon.live.predictor import LivePrediction
from kairon.live.pure_fns import post_rounding_guard
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore

logger = logging.getLogger(__name__)


def _uuid7() -> str:
    """Generate a short unique ID for orders and traces."""
    return uuid.uuid4().hex[:24]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _unpack_stops(stops: tuple[float, ...] | None) -> tuple[float, float, str | None]:
    """Unpack a ``_position_stops`` entry into (sl, tp, entry_ts).

    The stops dict historically stored a 2-tuple ``(sl, tp)``. It now stores a
    3-tuple ``(sl, tp, entry_ts)`` so the software-stop close (and the rejected-
    close reconcile path) can record the *true* holding duration instead of
    reading ``filled.ts`` (which on the reconcile path is ``_utc_now_iso()``,
    producing a near-zero duration). The 2-tuple form is still accepted for
    backward compatibility with callers/tests that set stops directly.
    """
    if stops is None:
        return (float("nan"), float("nan"), None)
    if len(stops) >= 3:
        return (float(stops[0]), float(stops[1]), stops[2])
    return (float(stops[0]), float(stops[1]), None)


class TradingLoop:
    """Async trading loop orchestrator.

    Wires together the candle feed, signal strategy (or model predictor),
    position sizer, risk guardian, and order broker into a single tick loop.

    Args:
        config: Live trading configuration.
        broker: The broker to use (PaperBroker for dry_run, BybitBroker for live).
        predictor: The live model predictor (optional, used if no strategy).
        strategy: The signal strategy (optional, takes priority over predictor).
        guardian: The risk guardian.
        reconciler: The position reconciler.
        store: The live store for persistence.
        feed: An async feed object with a ``queue`` attribute yielding
            pyarrow Tables of closed bars.
    """

    def __init__(
        self,
        *,
        config: LiveConfig,
        broker: Broker,
        predictor: Any | None = None,
        strategy: Any | None = None,
        guardian: Guardian,
        reconciler: Reconciler,
        store: LiveStore,
        feed: Any,
        bankroll: BankrollConfig | None = None,
        attach_stops: bool = False,
        drift_killswitch: DriftKillSwitch | None = None,
        orderflow_provider: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        self._broker = broker
        self._predictor = predictor
        self._strategy = strategy
        self._guardian = guardian
        self._reconciler = reconciler
        self._store = store
        self._feed = feed
        # When True, attach the strategy's ATR-based sl_price/tp_price to entry
        # orders as native exchange-side TP/SL (scalping mode). Default False
        # leaves the legacy (no attached stops) behaviour unchanged.
        self._attach_stops: bool = attach_stops
        # Opt-in performance drift kill-switch (Phase 3). When set, every
        # bankroll-mode close feeds a rolling win-rate/expectancy window; if live
        # performance drifts below the research edge the loop halts. ``None``
        # preserves the legacy behaviour (no drift kill-switch).
        self._drift_killswitch: DriftKillSwitch | None = drift_killswitch
        # Opt-in order-flow provider (Phase 4b). When set, the loop calls it
        # per symbol before each predict to refresh the strategy's order-book
        # snapshot (``strategy.last_orderflow``). ``None`` preserves the legacy
        # behaviour (no order-flow input). The provider is a *sync* callable
        # returning a cached snapshot — the runner polls the book async and
        # stashes the latest result, so this call never blocks the loop.
        self._orderflow_provider: Callable[[str], Any] | None = orderflow_provider
        # Running tally of realized PnL since session start, fed to the guardian's
        # daily-loss kill switch (was hard-coded 0.0, so the switch never fired).
        self._session_realized_pnl: float = 0.0

        self._tick_count: int = 0
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._fill_task: asyncio.Task | None = None

        # Track current positions for diffing
        self._positions: dict[str, Position] = {}

        # Track which order opened each position (for decision outcome tracking)
        self._position_entry_orders: dict[str, str] = {}
        # Track the entry timestamp of each open position (for trade duration)
        self._position_entry_ts: dict[str, str] = {}
        # Intended (re-anchored) SL/TP per open position, stored when an opening
        # order with attached stops is accepted. Drives the software-side stop
        # monitor so loss-capping does not depend on Bybit's attached TP/SL
        # triggering reliably (on testnet it sometimes does not).
        self._position_stops: dict[str, tuple[float, ...]] = {}
        # Symbols whose software-stop close order is in flight (awaiting fill
        # reconciliation). Guards against re-firing the close / re-entering
        # while the broker has not yet reported the close fill.
        self._closing: set[str] = set()

        # Rolling bar buffer for the strategy (stores raw bar dicts)
        self._bar_buffer: dict[str, deque[dict]] = {}

        # --- Growth / synthetic-bankroll mode (opt-in) ---------------------
        # When bankroll is set, sizing uses the tracked bankroll (not broker
        # equity), compounds on realized PnL, logs to growth_ledger, and halts
        # at stop_at. Default None keeps the legacy broker-equity path intact.
        self._bankroll_cfg: BankrollConfig | None = bankroll
        self._bankroll: float = bankroll.start if bankroll is not None else 0.0
        self._bankroll_start: float = self._bankroll
        self._bankroll_peak: float = self._bankroll
        self._bankroll_milestones_hit: set[float] = set()
        if bankroll is not None:
            self._store.write_ledger(
                kind="start",
                bankroll=self._bankroll,
                note=f"start={bankroll.start} lev={bankroll.leverage} "
                     f"alloc={bankroll.allocation} stop_at={bankroll.stop_at}",
            )

    @property
    def bankroll(self) -> float:
        """Current synthetic bankroll (0.0 when bankroll mode is off)."""
        return self._bankroll

    @property
    def bankroll_peak(self) -> float:
        """Peak synthetic bankroll this session (for drawdown reporting)."""
        return self._bankroll_peak

    @property
    def session_realized_pnl(self) -> float:
        """Realized PnL accumulated since session start (for the kill switch)."""
        return self._session_realized_pnl

    @property
    def bankroll_config(self) -> BankrollConfig | None:
        """The active BankrollConfig, or None when growth mode is off."""
        return self._bankroll_cfg

    @property
    def tick_count(self) -> int:
        """Number of ticks processed so far."""
        return self._tick_count

    @property
    def is_running(self) -> bool:
        """Whether the loop is currently running."""
        return self._running

    # --- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the trading loop as an async task."""
        if self._running:
            logger.warning("TradingLoop already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="trading-loop")
        if hasattr(self._broker, "stream_fills"):
            self._fill_task = asyncio.create_task(
                self._drain_fills(), name="fill-drain"
            )
        logger.info(
            "TradingLoop started (symbols=%s, dry_run=%s, warmup=%d)",
            self._config.symbols,
            self._config.dry_run,
            self._config.warmup_bars,
        )

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        if self._fill_task is not None:
            self._fill_task.cancel()
            try:
                await self._fill_task
            except asyncio.CancelledError:
                pass
        self._fill_task = None
        logger.info("TradingLoop stopped after %d ticks", self._tick_count)

    async def finalize_open_positions(self) -> None:
        """Record a ``manual_close`` outcome for any position still open at shutdown.

        Positions closed by the session-end flatten bypass ``_update_position``
        (the fill-drain task has already stopped by the time ``_flatten`` runs),
        so their entry decision would otherwise never receive an outcome and the
        journal would report 0 closed decisions forever. We approximate the
        realized PnL with the broker's current unrealized PnL on the open
        position and mark the outcome ``manual_close``.

        We also write a ``closed_trade`` row (exit price derived from the
        unrealized PnL) so the session report counts this as a round-trip trade
        with a locally-journaled realized PnL, instead of relying solely on
        Bybit's closed-PnL endpoint.

        Call after :meth:`stop` and *before* flattening, while the position still
        exists on the broker.

        Iterates the UNION of ``_position_entry_orders`` (positions with a
        journaled entry decision) and ``_positions`` (the local mirror) so a
        position whose entry-order tracking drifted — but which is still open on
        the broker and tracked in the local mirror — is still journaled as a
        ``manual_close`` round trip with a bankroll debit. The decision-outcome
        update only runs when an entry order id exists for the symbol.
        """
        symbols = set(self._position_entry_orders) | set(self._positions)
        if not symbols:
            return
        now_ts = _utc_now_iso()
        for symbol in sorted(symbols):
            order_id = self._position_entry_orders.get(symbol)
            try:
                positions = await self._broker.get_positions(symbol)
            except Exception as e:
                logger.warning("finalize_open_positions: get_positions failed for %s: %s", symbol, e)
                continue
            pos = next((p for p in positions if p.symbol == symbol), None)
            realized = 0.0
            if pos is not None and pos.qty > 1e-9:
                realized = float(pos.unrealized_pnl or 0.0)
                # Approximate the exit price from the unrealized PnL so the
                # session report records this as a round-trip trade. The actual
                # close happens moments later in _flatten; this mark-price
                # approximation matches the decision-outcome PnL we record.
                entry_price = pos.avg_entry
                if pos.side == OrderSide.BUY:
                    exit_price = entry_price + realized / pos.qty
                else:
                    exit_price = entry_price - realized / pos.qty
                entry_ts = self._position_entry_ts.get(symbol, now_ts)
                try:
                    duration_s = (
                        datetime.fromisoformat(now_ts) - datetime.fromisoformat(entry_ts)
                    ).total_seconds()
                except (ValueError, TypeError):
                    duration_s = None
                try:
                    self._store.write_closed_trade(
                        symbol=symbol,
                        side=pos.side.value,
                        entry_qty=pos.qty,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        realized_pnl=round(realized, 8),
                        fee=0.0,
                        entry_ts=entry_ts,
                        exit_ts=now_ts,
                        duration_seconds=duration_s,
                    )
                except Exception:
                    logger.debug("finalize_open_positions: could not write closed trade for %s", symbol)
                # Compound the session-end close into the synthetic bankroll so
                # the growth curve reflects every realized PnL, not just the
                # mid-session round-trips captured in _update_position. No-op
                # when bankroll mode is off.
                self._apply_bankroll_close(symbol, realized)
                self._session_realized_pnl += realized
            if order_id is not None:
                try:
                    self._store.update_decision_outcome(
                        order_id=order_id,
                        outcome="manual_close",
                        pnl=round(realized, 8),
                        ts=now_ts,
                    )
                except Exception:
                    logger.debug("finalize_open_positions: could not update outcome for %s", order_id)
                logger.info(
                    "Finalized open decision %s for %s: outcome=manual_close pnl=%.6f",
                    order_id, symbol, realized,
                )
            else:
                logger.info(
                    "Finalized open position for %s (no entry order tracked): "
                    "manual_close pnl=%.6f",
                    symbol, realized,
                )
            # Manual close at shutdown: cancel orphan attached TP/SL so they do
            # not fire after the flatten on a symbol with no position.
            await self._cancel_orphan_stops(symbol)

    # --- Main loop -----------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main loop: consume bars from the feed and process ticks."""
        try:
            while self._running:
                # 1. Check kill switch
                if self._store.is_halted():
                    logger.info("Kill switch active; skipping tick")
                    await asyncio.sleep(self._config.cadence_seconds)
                    continue

                # 2. Get next closed bar from the feed
                try:
                    bar_table = await asyncio.wait_for(
                        self._feed.queue.get(),
                        timeout=self._config.cadence_seconds * 2,
                    )
                except TimeoutError:
                    logger.debug("No bar received within timeout; retrying")
                    continue
                except asyncio.CancelledError:
                    break

                # 3. Process the bar
                await self._process_tick(bar_table)

        except asyncio.CancelledError:
            logger.info("TradingLoop cancelled")
        except Exception:
            logger.exception("TradingLoop crashed")
            raise

    async def _process_tick(self, bar: Any) -> None:  # noqa: PLR0911
        """Process a single closed bar (one tick).

        The return count is high by design: each guard (extract failure, warm-up,
        no equity, cooldown, no delta, guardian block) is a clear early-exit.
        Merging them would bury the control flow, so the complexity-threshold
        rule is suppressed here rather than refactored.
        """
        self._tick_count += 1

        # Extract symbol and close price from the bar
        try:
            symbol = self._extract_symbol(bar)
            close_price = self._extract_close(bar)
        except Exception as e:
            logger.warning("Failed to extract bar data: %s", e)
            self._store.write_heartbeat(
                mode="error",
                n_positions=len(self._positions),
            )
            return

        # 2a. Update mark price on PaperBroker for fill simulation
        if isinstance(self._broker, PaperBroker):
            self._broker.set_mark_price(symbol, close_price)
            new_fills = self._broker.tick()
            for fill in new_fills:
                self._store.write_fill(fill)
                logger.info("PaperBroker fill: %s %s qty=%.6f @ %.2f",
                            fill.side.value, fill.symbol, fill.qty, fill.price)

        # 2b. Accumulate bar into rolling buffer for strategy
        self._accumulate_bar(symbol, bar)

        # 2c. Software-side stop monitor (scalping): cap losses independent of
        # the exchange's attached TP/SL trigger reliability. Closes any open
        # position whose intended SL/TP has been crossed. If the current
        # symbol's position was closed (or is mid-close), skip the normal signal
        # path for this tick so we never re-enter on top of a just-stopped-out
        # position. See :meth:`_check_software_stops`.
        if await self._check_software_stops(symbol, close_price):
            return

        # 3. Warm-up: skip trading for the first N bars
        warmup = self._strategy.warmup_bars if self._strategy else self._config.warmup_bars
        if self._tick_count <= warmup:
            logger.debug(
                "Warm-up tick %d/%d for %s",
                self._tick_count,
                warmup,
                symbol,
            )
            self._store.write_heartbeat(
                mode="warmup",
                n_positions=len(self._positions),
                last_signal_ts=_utc_now_iso(),
            )
            return

        # 4. Get current equity from broker
        try:
            balances = await self._broker.get_balances()
            equity = self._get_usdt_equity(balances)
        except Exception as e:
            logger.error("Failed to get balances: %s", e)
            equity = 0.0

        if equity <= 0:
            logger.warning("No equity available; skipping tick")
            self._store.write_heartbeat(
                mode="no_equity",
                n_positions=len(self._positions),
            )
            return

        # 5. Check cooldown for this symbol
        if self._guardian.is_cooling_down(symbol):
            logger.info("Symbol %s is in cooldown; skipping", symbol)
            self._store.write_heartbeat(
                mode="cooldown",
                equity=equity,
                n_positions=len(self._positions),
            )
            return

        # 6. Build prediction from strategy or model
        prediction = self._make_prediction(symbol)

        # 7. Compute target position size (direction-aware)
        from kairon.policy.sizer import size_position_vol_aware

        if self._bankroll_cfg is not None:
            # Growth/scalping mode: size off the tracked synthetic bankroll,
            # not the vol-scaled broker-equity fraction. Two sizing modes:
            #   * risk_per_trade > 0 (scalping): fixed-fractional risk sizing —
            #     qty = (risk_per_trade * bankroll) / sl_distance, capped by the
            #     leverage notional. Bounds the loss per trade to a known
            #     fraction of the bankroll when the SL is hit.
            #   * risk_per_trade == 0 (growth): fixed notional sizing —
            #     bankroll * leverage * allocation / price.
            # Symbols whose floored qty is below the broker min are skipped with
            # a logged heartbeat + ledger row rather than silently dropped.
            cfg = self._bankroll_cfg
            snapshot = (
                self._strategy.last_indicator_snapshot
                if self._strategy is not None and hasattr(self._strategy, "last_indicator_snapshot")
                else {}
            )
            sl_price = snapshot.get("sl_price")
            min_qty = getattr(self._broker, "min_qty_for", lambda _: 1e-9)(symbol)
            notional_cap_qty = (self._bankroll * cfg.sizing_notional_factor) / close_price

            if cfg.risk_per_trade > 0 and sl_price is not None and math.isfinite(float(sl_price)):
                # Fixed-fractional risk sizing: risk a fraction of the bankroll
                # over the ATR-based stop distance, capped by the leverage
                # notional so a tight stop can't blow past the position cap.
                sl_distance = abs(close_price - float(sl_price))
                if sl_distance <= 0:
                    sl_distance = None
            else:
                sl_distance = None
            if cfg.risk_per_trade > 0 and sl_distance is not None:
                risk_qty = (cfg.risk_per_trade * self._bankroll) / sl_distance
                raw_magnitude = min(risk_qty, notional_cap_qty)
            else:
                # Notional sizing (growth mode, or scalping with no usable SL).
                raw_magnitude = notional_cap_qty

            raw_qty = raw_magnitude if prediction.direction >= 0 else -raw_magnitude

            # Risk-cap guard (Phase 0.2): after determining the quantity that
            # will actually reach the broker, recompute the implied risk and
            # skip if it would exceed risk_per_trade * (1 + tol). The broker
            # floors quantity DOWN to the lot step (so plain risk sizing only
            # shrinks risk), but (a) min-lot overshoot (allow_min_lot_overshoot)
            # bumps qty UP to the min lot, and (b) confidence-scaled sizing
            # (Phase 2.6) inflates the intended qty — both can breach the cap.
            # This guard is the runtime guarantee the cap stays inviolable.
            effective_magnitude, breach_reason = post_rounding_guard(
                raw_qty=raw_qty,
                min_qty=min_qty,
                sl_distance=sl_distance,
                bankroll=self._bankroll,
                risk_per_trade=cfg.risk_per_trade,
                tol=cfg.risk_cap_tol,
                enforce_risk_cap=cfg.enforce_risk_cap,
                allow_min_lot_overshoot=cfg.allow_min_lot_overshoot,
            )
            if breach_reason == "below_min_lot":
                logger.info(
                    "Bankroll too small for %s: target_qty=%.6f < min_qty=%.6f "
                    "(bankroll=%.2f notional_cap=%.2f)",
                    symbol, abs(raw_qty), min_qty, self._bankroll,
                    notional_cap_qty * close_price,
                )
                self._store.write_ledger(
                    kind="skip",
                    bankroll=self._bankroll,
                    symbol=symbol,
                    note=f"target_qty={abs(raw_qty):.6f} < min_qty={min_qty:.6f} "
                         f"(notional_cap={notional_cap_qty * close_price:.2f})",
                )
                self._store.write_heartbeat(
                    mode="bankroll_too_small",
                    equity=equity,
                    n_positions=len(self._positions),
                    last_signal_ts=_utc_now_iso(),
                )
                return
            if breach_reason is not None:
                # risk_cap_breach or risk_cap_breach_overshoot: keep the cap
                # inviolable by skipping with a distinct ledger row.
                logger.info(
                    "Risk-cap breach skip for %s: reason=%s effective_qty=%.6f "
                    "implied_risk would exceed %.4f (cap=%.4f tol=%.2f)",
                    symbol, breach_reason, effective_magnitude,
                    cfg.risk_per_trade * (1 + cfg.risk_cap_tol),
                    cfg.risk_per_trade, cfg.risk_cap_tol,
                )
                self._store.write_ledger(
                    kind="risk_cap_breach_skip",
                    bankroll=self._bankroll,
                    symbol=symbol,
                    note=f"reason={breach_reason} effective_qty={effective_magnitude:.6f} "
                         f"min_qty={min_qty:.6f}",
                )
                self._store.write_heartbeat(
                    mode="risk_cap_breach",
                    equity=equity,
                    n_positions=len(self._positions),
                    last_signal_ts=_utc_now_iso(),
                )
                return
            # Apply the (possibly overshoot-bumped) effective magnitude.
            raw_qty = effective_magnitude if prediction.direction >= 0 else -effective_magnitude
        else:
            max_equity_frac = getattr(self._guardian, "max_position_equity_fraction", 0.20)
            snapshot = (
                self._strategy.last_indicator_snapshot
                if self._strategy is not None and hasattr(self._strategy, "last_indicator_snapshot")
                else {}
            )
            if snapshot.get("regime_prob_stressed", 0.0) > 0.3:
                max_equity_frac = max_equity_frac / 2.0

            raw_qty = size_position_vol_aware(
                equity=equity,
                price=close_price,
                predicted_magnitude=abs(prediction.magnitude),
                realized_vol_target=max(prediction.volatility, 0.001),
                max_position_equity_fraction=max_equity_frac,
                direction=prediction.direction,
            )

        # 8. Current signed position
        current_pos = self._positions.get(symbol)
        if current_pos is not None:
            current_signed = current_pos.qty if current_pos.side == OrderSide.BUY else -current_pos.qty
        else:
            current_signed = 0.0

        # A neutral prediction means "hold current position", not "flatten".
        direction = prediction.direction
        if direction == 0.0:
            target_qty = current_signed
        else:
            target_qty = raw_qty

        # Position-flip protection (scalping, attach_stops only): when the
        # signal flips AGAINST an open position, close to flat rather than
        # reversing in a single market order. A reversal rides the wrong way
        # until the next flip and is the main source of whipsaw churn and
        # oversized losses (a 2.5%-risk trade previously lost 12.8% this way).
        # Flattening lets the move exhaust; the per-symbol SL cooldown (engaged
        # on a losing close) blocks immediate re-entry. The mirror long->short
        # is symmetric. Default path (attach_stops=False) reverses as before.
        if (
            self._attach_stops
            and current_signed != 0.0
            and direction != 0.0
            and (direction > 0) != (current_signed > 0)
        ):
            target_qty = 0.0

        delta = target_qty - current_signed

        # An order that moves against an open position (opposite side) reduces
        # or flattens it; such close orders carry no attached TP/SL and are
        # marked reduce-only. Same-side adds and fresh opens attach stops.
        is_reducing = current_signed != 0.0 and (delta > 0) != (current_signed > 0)

        # Skip if delta is below the broker's minimum lot size or negligible.
        min_delta = getattr(self._broker, "min_qty_for", lambda _: 1e-9)(symbol)
        if abs(delta) < max(1e-9, min_delta):
            logger.debug("No position change needed for %s (delta=%.6f)", symbol, delta)
            self._store.write_heartbeat(
                mode="hold",
                equity=equity,
                n_positions=len(self._positions),
                last_signal_ts=_utc_now_iso(),
            )
            return

        # 9. Build order
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order_qty = abs(delta)

        # Scalping: attach the strategy's ATR-based sl_price/tp_price as native
        # exchange-side TP/SL so Bybit manages the exits server-side. Disabled
        # by default (attach_stops=False) to keep legacy behaviour unchanged.
        attach_sl: float | None = None
        attach_tp: float | None = None
        if self._attach_stops and not is_reducing:
            snap = (
                self._strategy.last_indicator_snapshot
                if self._strategy is not None and hasattr(self._strategy, "last_indicator_snapshot")
                else {}
            )
            sp = snap.get("sl_price")
            tp = snap.get("tp_price")
            snap_close = snap.get("close")
            if (
                sp is not None and math.isfinite(float(sp))
                and tp is not None and math.isfinite(float(tp))
                and snap_close is not None and math.isfinite(float(snap_close))
            ):
                # Re-anchor the ATR-based stops to the live last price. The
                # strategy computes sl/tp relative to the 1m bar close, but
                # Bybit validates attached TP/SL against the live last price at
                # submission — and the actual entry fills near that live price,
                # not the bar close. On a volatile book the two can differ enough
                # to put the TP on the wrong side of entry (e.g. a short whose
                # bar-close-anchored TP ends up above the live fill), which Bybit
                # rejects. Anchoring to the live price keeps the stops on the
                # correct side of entry while preserving the strategy's ATR
                # sl/tp *distances* (and thus the per-trade risk).
                get_last = getattr(self._broker, "get_last_price", None)
                live = None
                if get_last is not None:
                    try:
                        live = await get_last(symbol)
                    except Exception as e:
                        logger.debug("get_last_price failed for %s: %s", symbol, e)
                anchor = (
                    float(live) if live is not None and math.isfinite(float(live))
                    else float(snap_close)
                )
                sl_dist = abs(float(sp) - float(snap_close))
                tp_dist = abs(float(tp) - float(snap_close))
                if delta > 0:  # long: SL below entry, TP above
                    attach_sl = anchor - sl_dist
                    attach_tp = anchor + tp_dist
                else:  # short: SL above entry, TP below
                    attach_sl = anchor + sl_dist
                    attach_tp = anchor - tp_dist
            else:
                if sp is not None and math.isfinite(float(sp)):
                    attach_sl = float(sp)
                if tp is not None and math.isfinite(float(tp)):
                    attach_tp = float(tp)

        order = Order(
            id=_uuid7(),
            intent_id=_uuid7(),
            trace_id=_uuid7(),
            symbol=symbol,
            side=side,
            qty=order_qty,
            order_type=OrderType.MARKET,
            sl=attach_sl,
            tp=attach_tp,
            reduce_only=is_reducing,
            status=OrderStatus.PENDING,
            ts=_utc_now_iso(),
        )

        # 10. Persist order intent BEFORE sending to broker
        self._store.write_order(order)

        # 10a. Write the full decision context to the trade journal
        self._write_decision(prediction, order, equity, close_price)

        # 11. Run guardian checks
        positions_tuple = tuple(self._positions.values())
        position_alerts = self._guardian.check_positions(positions_tuple, equity)
        daily_alert = self._guardian.check_daily_loss(self._session_realized_pnl, equity)

        critical_alerts = [a for a in position_alerts if a.severity == "critical"]
        if daily_alert is not None and daily_alert.severity == "critical":
            critical_alerts.append(daily_alert)

        if critical_alerts:
            logger.warning(
                "Guardian blocked order for %s: %d critical alerts",
                symbol,
                len(critical_alerts),
            )
            self._store.update_order_status(order.id, OrderStatus.REJECTED)
            self._store.write_event(
                kind="guardian_block",
                severity="warning",
                payload_json=json.dumps({
                    "symbol": symbol,
                    "order_id": order.id,
                    "alerts": [a.message for a in critical_alerts],
                }),
            )
            self._store.write_heartbeat(
                mode="blocked",
                equity=equity,
                n_positions=len(self._positions),
            )
            return

        # 12. Submit order through broker
        try:
            filled = await self._broker.place_order(order)
            self._store.update_order_status(
                filled.id, filled.status, filled.broker_id
            )
            logger.info(
                "Order %s: %s %s qty=%.6f status=%s",
                filled.id,
                filled.side.value,
                filled.symbol,
                filled.qty,
                filled.status.value,
            )

            # 12b. Mirror the attached stops onto the open position so the
            # software-side stop monitor can cap losses regardless of whether
            # the exchange's attached TP/SL actually triggers. Only opening
            # orders carry stops; reducing/close orders (is_reducing) do not.
            if attach_sl is not None and attach_tp is not None:
                self._position_stops[symbol] = (attach_sl, attach_tp)

            # 12a. Persist fills from PaperBroker (market order fills are immediate)
            if isinstance(self._broker, PaperBroker):
                for fill in self._broker.get_fills():
                    if fill.order_id == filled.id:
                        existing = self._store.get_order(fill.order_id)
                        if existing is not None:
                            self._store.write_fill(fill)
                            logger.info(
                                "Fill: %s %s qty=%.6f @ %.2f fee=%.4f",
                                fill.side.value, fill.symbol, fill.qty,
                                fill.price, fill.fee,
                            )

            # If the broker rejected the order, do not update local positions.
            if filled.status == OrderStatus.REJECTED:
                self._store.write_heartbeat(
                    mode="rejected",
                    equity=equity,
                    n_positions=len(self._positions),
                )
                return

        except Exception as e:
            logger.error("Order submission failed: %s", e)
            self._store.update_order_status(order.id, OrderStatus.REJECTED)
            self._store.write_event(
                kind="broker_error",
                severity="critical",
                payload_json=f'{{"order_id": "{order.id}", "error": "{e}"}}',
            )
            return

        # 13. Update positions using the broker-accepted quantity.
        # For live brokers that stream fills, the fill-drain task applies the
        # actual executed quantity and price; for PaperBroker we update here
        # because fills are synchronous.
        if not hasattr(self._broker, "stream_fills"):
            self._update_position(symbol, side, filled.qty, close_price, filled)

        # 14. Run reconciler periodically
        reconcile_interval = max(1, self._config.reconcile_interval_seconds // self._config.cadence_seconds)
        if self._tick_count % reconcile_interval == 0:
            try:
                alerts = await self._reconciler.reconcile()
                for alert in alerts:
                    logger.info("Reconciler alert: %s", alert.message)
                    self._store.write_event(
                        kind="reconciler_alert",
                        severity=alert.severity.value,
                        payload_json=json.dumps({
                            "rule": alert.rule_name,
                            "message": alert.message,
                            "source": alert.source,
                        }),
                    )
            except Exception as e:
                logger.error("Reconciler failed: %s", e)

        # 15. Write heartbeat
        self._store.write_heartbeat(
            mode="live" if not self._config.dry_run else "dry_run",
            equity=equity,
            n_positions=len(self._positions),
            last_signal_ts=_utc_now_iso(),
        )

    # --- Fill streaming ------------------------------------------------------

    async def _drain_fills(self) -> None:
        """Drain the broker's fill queue and persist incremental fills.

        Launched as a background task when the broker exposes
        ``stream_fills()``. Each fill updates the local order status and the
        tracked position using the actual fill price and quantity.
        """
        try:
            fill_queue = await self._broker.stream_fills()
        except Exception as e:
            logger.warning("Could not start fill stream: %s", e)
            return

        try:
            while self._running:
                try:
                    fill = await asyncio.wait_for(fill_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

                self._store.write_fill(fill)
                existing_order = self._store.get_order(fill.order_id)
                total_filled = sum(
                    f.qty for f in self._store.get_fills_for_order(fill.order_id)
                )
                status = (
                    OrderStatus.FILLED
                    if existing_order is not None
                    and total_filled >= existing_order.qty - 1e-9
                    else OrderStatus.PARTIAL
                )
                self._store.update_order_status(fill.order_id, status)
                self._update_position(
                    fill.symbol, fill.side, fill.qty, fill.price,
                    Order(
                        id=fill.order_id,
                        intent_id="",
                        trace_id="",
                        symbol=fill.symbol,
                        side=fill.side,
                        qty=fill.qty,
                        order_type=OrderType.MARKET,
                        status=status,
                        ts=fill.ts,
                    ),
                )
                # If this fill closed the position, cancel any orphan attached
                # TP/SL conditional orders (the un-triggered leg of the pair, or
                # both legs if the close was a reduce-only market order).
                if fill.symbol not in self._positions:
                    await self._cancel_orphan_stops(fill.symbol)
                logger.info(
                    "Fill drain: %s %s qty=%.6f @ %.2f fee=%.4f",
                    fill.side.value, fill.symbol, fill.qty, fill.price, fill.fee,
                )
        except asyncio.CancelledError:
            logger.info("Fill drain task cancelled")
        except Exception:
            logger.exception("Fill drain task crashed")
            raise

    # --- Software-side stop monitoring --------------------------------------
    # The scalping engine attaches ATR-based TP/SL to entry orders so the
    # exchange manages exits server-side. On Bybit testnet, attached stops
    # have been observed NOT to trigger (price ran straight through an SL
    # without the position closing), which let a 2.5%-risk trade lose 12.8%.
    # This monitor is the robust, self-contained loss cap: each tick it checks
    # every open position against its intended SL/TP using the freshest price
    # available and closes (reduce-only) any that have been crossed. The
    # resulting fill flows through the normal close path, so realized PnL, the
    # per-symbol SL cooldown, bankroll compounding, and the trade journal all
    # update exactly as an exchange-triggered stop would.

    async def _fetch_stop_price(
        self, symbol: str, fallback: float | None
    ) -> float | None:
        """Freshest price for ``symbol`` for stop evaluation.

        Prefers the broker's live last price (fresher than the bar close);
        falls back to ``fallback`` (the current bar's close, available only for
        the tick's own symbol). Returns None when no price is available — the
        caller skips that position this tick and re-evaluates next tick.
        """
        get_last = getattr(self._broker, "get_last_price", None)
        if get_last is not None:
            try:
                live = await get_last(symbol)
            except Exception as e:
                logger.debug("get_last_price failed for %s: %s", symbol, e)
                live = None
            if live is not None and math.isfinite(float(live)) and float(live) > 0:
                return float(live)
        return fallback

    async def _cancel_orphan_stops(self, symbol: str) -> None:
        """Cancel any orphan attached TP/SL conditional orders after a close.

        When a position closes by any path OTHER than its attached SL/TP
        triggering (software-stop market close, signal flip to flat, manual
        flatten, or the rejected-close reconcile), the attached conditional stop
        orders can remain live on Bybit as orphans — they would fire on a future
        price move with no position to close. Even when the attached SL/TP DID
        trigger, the *other* leg (the TP when the SL hit, or vice-versa) is left
        dangling. Cancelling all conditional orders for the symbol after every
        close keeps the book clean so a stale orphan cannot interfere with the
        next position's freshly-attached stops.

        Active only in scalping/attach_stops mode (where attached stops exist);
        a no-op otherwise and on brokers without ``cancel_all`` (e.g.
        PaperBroker). Cheap and safe — a cancel on a symbol with no conditional
        orders is a no-op on the exchange. Default ON in attach_stops mode
        (a risk-correctness fix, not an opt-in).
        """
        if not self._attach_stops:
            return
        cancel = getattr(self._broker, "cancel_all", None)
        if cancel is None:
            return
        try:
            await cancel(symbol)
        except Exception as e:
            logger.debug("cancel_all (orphan stops) failed for %s: %s", symbol, e)

    async def _check_software_stops(
        self, current_symbol: str, current_close: float
    ) -> bool:
        """Close any open position whose intended SL/TP has been crossed.

        Returns True if the current symbol's position was closed (or is
        already mid-close) so the caller can skip the normal signal path for
        this tick — never re-enter on top of a just-stopped-out position.
        Positions on other symbols are also monitored (their fills reconcile
        asynchronously), but those do not block the current symbol's signal.
        """
        if not self._attach_stops:
            return False
        if current_symbol in self._closing:
            return True

        closed_current = False
        for symbol, pos in list(self._positions.items()):
            if symbol in self._closing:
                continue
            stops = self._position_stops.get(symbol)
            if stops is None:
                continue
            sl_price, tp_price, _entry_ts = _unpack_stops(stops)
            if not (math.isfinite(sl_price) and math.isfinite(tp_price)):
                continue

            fallback = current_close if symbol == current_symbol else None
            price = await self._fetch_stop_price(symbol, fallback)
            if price is None or price <= 0:
                continue

            hit_sl = (pos.side == OrderSide.BUY and price <= sl_price) or (
                pos.side == OrderSide.SELL and price >= sl_price
            )
            hit_tp = (pos.side == OrderSide.BUY and price >= tp_price) or (
                pos.side == OrderSide.SELL and price <= tp_price
            )
            if not (hit_sl or hit_tp):
                continue

            reason = "sl" if hit_sl else "tp"
            logger.info(
                "Software stop (%s) triggered for %s %s qty=%.6f: "
                "price=%.4f sl=%.4f tp=%.4f",
                reason.upper(), symbol, pos.side.value, pos.qty,
                price, sl_price, tp_price,
            )
            # The stop executes at the stop LEVEL (where the attached SL/TP sits),
            # not at the crossing price — using the crossing price would overstate
            # the realized loss when the exchange already filled at the stop.
            exit_price = sl_price if hit_sl else tp_price
            await self._close_position_software(symbol, pos, reason, exit_price)
            if symbol == current_symbol:
                closed_current = True

        return closed_current

    async def _close_position_software(
        self, symbol: str, pos: Position, reason: str, exit_price: float
    ) -> None:
        """Close an open position with a reduce-only market order.

        Mirrors the normal order path's post-submit fill handling so the close
        reconciles identically for PaperBroker (synchronous fills) and live
        brokers (fill-drain task). ``_closing`` guards against re-firing until
        the position is reconciled away in :meth:`_update_position`.
        ``exit_price`` is the stop LEVEL (where the attached SL/TP sits); for
        PaperBroker it is used as the close fill price, and for the rejected-
        close reconciliation it is the local close price.
        """
        self._closing.add(symbol)
        stops = self._position_stops.pop(symbol, None)
        close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        order = Order(
            id=_uuid7(),
            intent_id=_uuid7(),
            trace_id=_uuid7(),
            symbol=symbol,
            side=close_side,
            qty=pos.qty,
            order_type=OrderType.MARKET,
            sl=None,
            tp=None,
            reduce_only=True,
            status=OrderStatus.PENDING,
            ts=_utc_now_iso(),
        )
        self._store.write_order(order)
        self._store.write_event(
            kind="software_stop",
            severity="info",
            payload_json=json.dumps({
                "symbol": symbol,
                "side": pos.side.value,
                "qty": pos.qty,
                "reason": reason,
                "order_id": order.id,
            }),
        )

        try:
            filled = await self._broker.place_order(order)
        except Exception:
            logger.exception("Software close order failed for %s", symbol)
            self._closing.discard(symbol)
            # Restore stops so the monitor retries on the next tick.
            if stops is not None and symbol not in self._position_stops:
                self._position_stops[symbol] = stops
            return

        self._store.update_order_status(filled.id, filled.status, filled.broker_id)
        logger.info(
            "Software close order %s: %s %s qty=%.6f status=%s",
            filled.id, filled.side.value, filled.symbol, filled.qty,
            filled.status.value,
        )

        if filled.status == OrderStatus.REJECTED:
            # A reduce-only close that the broker rejects is almost always the
            # reconciliation race: the exchange's attached SL/TP fired and
            # closed the position first (the WS close fill then often does not
            # drain on testnet), so Bybit reports position=0 and refuses the
            # redundant reduce-only order (ErrCode 110017). Reconcile the local
            # mirror to flat at the trigger price so realized PnL, the SL
            # cooldown, and bankroll compounding all record. If the position is
            # genuinely still open on the broker, restore stops and retry.
            await self._reconcile_software_close(symbol, pos, stops, exit_price)
            return

        # PaperBroker fills synchronously: persist + apply the close now.
        if isinstance(self._broker, PaperBroker):
            for fill in self._broker.get_fills():
                if fill.order_id == filled.id:
                    existing = self._store.get_order(fill.order_id)
                    if existing is not None:
                        self._store.write_fill(fill)
        if not hasattr(self._broker, "stream_fills"):
            self._update_position(
                symbol, close_side, filled.qty, exit_price, filled
            )
            # PaperBroker / sync-fill path: the close is already applied. Cancel
            # orphan attached stops for the now-closed position.
            if symbol not in self._positions:
                await self._cancel_orphan_stops(symbol)

    async def _reconcile_software_close(
        self,
        symbol: str,
        pos: Position,
        stops: tuple[float, ...] | None,
        exit_price: float,
    ) -> None:
        """Reconcile after a rejected software close.

        Confirms against the broker whether the position is truly gone. If it
        is (the attached SL/TP closed it first), mirrors the close locally at
        ``exit_price`` so the trade journal, SL cooldown, and bankroll
        compounding update exactly as a drained close would. If the position is
        still open on the broker (a rejection for some other reason), restores
        the stops so the monitor retries on the next tick.

        The true entry timestamp is carried through the ``stops`` tuple so the
        reconciled closed-trade row records the real holding duration rather
        than a near-zero value (the synthetic close Order's ``ts`` is
        ``_utc_now_iso()``). If the entry ts is already tracked on
        ``_position_entry_ts`` (the normal case), that value is authoritative.
        """
        self._closing.discard(symbol)
        local = self._positions.get(symbol)
        if local is None:
            return  # already reconciled via a drained fill

        # Ensure the true entry timestamp is available for the duration calc in
        # _update_position. Prefer the tracked entry ts; fall back to the one
        # carried in the stops tuple; last resort the open order's ts.
        if symbol not in self._position_entry_ts:
            _sl, _tp, stops_entry_ts = _unpack_stops(stops)
            if stops_entry_ts is not None:
                self._position_entry_ts[symbol] = stops_entry_ts

        broker_qty = local.qty
        if hasattr(self._broker, "get_positions"):
            try:
                broker_positions = await self._broker.get_positions(symbol)
            except Exception as e:
                logger.debug(
                    "get_positions failed during software-close reconcile "
                    "for %s: %s", symbol, e,
                )
                broker_positions = []
            broker_qty = next(
                (p.qty for p in broker_positions if p.symbol == symbol), 0.0
            )

        if broker_qty > 1e-9:
            # Still open on the broker — restore stops and retry next tick.
            if stops is not None and symbol not in self._position_stops:
                self._position_stops[symbol] = stops
            return

        # Position gone on the broker: close the local mirror at the stop level
        # (exit_price), recording the round trip.
        close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        logger.info(
            "Reconciling %s to flat at %.4f (exchange closed the position; "
            "no fill drained).", symbol, exit_price,
        )
        self._store.write_event(
            kind="software_stop_reconcile",
            severity="info",
            payload_json=json.dumps({
                "symbol": symbol, "side": pos.side.value, "qty": local.qty,
                "exit_price": exit_price,
            }),
        )
        self._update_position(
            symbol, close_side, local.qty, exit_price,
            Order(
                id=_uuid7(), intent_id="", trace_id="",
                symbol=symbol, side=close_side, qty=local.qty,
                order_type=OrderType.MARKET, status=OrderStatus.FILLED,
                ts=_utc_now_iso(),
            ),
        )
        # The exchange SL/TP closed the position; the un-triggered leg (and any
        # other conditional) is now an orphan — cancel it so it cannot fire on a
        # future move with no position to close.
        await self._cancel_orphan_stops(symbol)

    # --- Position tracking -------------------------------------------------------

    def _update_position(
        self, symbol: str, side: OrderSide, qty: float, price: float, filled: Order
    ) -> None:
        """Update the position tracking dict, persist to store, and record closed trades."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            if side == pos.side:
                # Same-side add: weighted average entry
                new_qty = pos.qty + qty
                new_avg = (pos.avg_entry * pos.qty + price * qty) / new_qty
                updated = Position(
                    symbol=symbol, side=pos.side, qty=new_qty,
                    avg_entry=new_avg, unrealized_pnl=0.0, ts=_utc_now_iso(),
                )
            else:
                # Opposite side: close or reduce — compute realized PnL
                close_qty = min(qty, pos.qty)
                remaining = pos.qty - close_qty

                # Realized PnL for the closed portion
                realized_pnl = close_qty * abs(price - pos.avg_entry)
                if pos.side == OrderSide.BUY:
                    realized_pnl = close_qty * (price - pos.avg_entry)
                else:
                    realized_pnl = close_qty * (pos.avg_entry - price)

                # Accumulate session realized PnL for the daily-loss kill switch.
                self._session_realized_pnl += realized_pnl

                # Scalping: engage the per-symbol SL cooldown on a losing close so
                # a symbol that just stopped out isn't immediately re-entered.
                if realized_pnl < 0 and self._attach_stops:
                    record_sl = getattr(self._guardian, "record_sl", None)
                    if callable(record_sl):
                        record_sl(symbol)

                # Growth mode: compound the synthetic bankroll with this close.
                self._apply_bankroll_close(symbol, realized_pnl)

                # Compute duration
                entry_ts = self._position_entry_ts.get(symbol, filled.ts)
                now_ts = _utc_now_iso()
                try:
                    entry_dt = datetime.fromisoformat(entry_ts)
                    now_dt = datetime.fromisoformat(now_ts)
                    duration_s = (now_dt - entry_dt).total_seconds()
                except (ValueError, TypeError):
                    duration_s = None

                # Record the closed trade
                self._store.write_closed_trade(
                    symbol=symbol,
                    side=pos.side.value,
                    entry_qty=close_qty,
                    entry_price=pos.avg_entry,
                    exit_price=price,
                    realized_pnl=round(realized_pnl, 8),
                    fee=0.0,
                    entry_ts=entry_ts,
                    exit_ts=now_ts,
                    duration_seconds=duration_s,
                )

                # Update the trade decision outcome
                # Find the original order for this position's entry
                if symbol in self._position_entry_orders:
                    outcome = "hit_tp" if realized_pnl > 0 else "hit_sl"
                    try:
                        self._store.update_decision_outcome(
                            order_id=self._position_entry_orders[symbol],
                            outcome=outcome,
                            pnl=round(realized_pnl, 8),
                            ts=now_ts,
                        )
                    except Exception:
                        logger.debug("Could not update decision outcome for %s", symbol)
                logger.info(
                    "Closed trade: %s %s qty=%.6f entry=%.2f exit=%.2f PnL=%.4f",
                    symbol, pos.side.value, close_qty, pos.avg_entry, price, realized_pnl,
                )

                if remaining <= 0:
                    del self._positions[symbol]
                    self._position_entry_ts.pop(symbol, None)
                    self._position_entry_orders.pop(symbol, None)
                    self._position_stops.pop(symbol, None)
                    self._closing.discard(symbol)
                    self._store.delete_position(symbol)
                    return
                updated = Position(
                    symbol=symbol, side=pos.side, qty=remaining,
                    avg_entry=pos.avg_entry, unrealized_pnl=0.0, ts=_utc_now_iso(),
                )
        else:
            # New position
            updated = Position(
                symbol=symbol, side=side, qty=qty,
                avg_entry=price, unrealized_pnl=0.0, ts=_utc_now_iso(),
            )
            self._position_entry_ts[symbol] = filled.ts
            self._position_entry_orders[symbol] = filled.id
            # Upgrade an attached-stops entry from a 2-tuple (sl, tp) to a
            # 3-tuple (sl, tp, entry_ts) using the TRUE fill timestamp, so a
            # later software-stop / reconcile close records the real holding
            # duration instead of a near-zero value from _utc_now_iso().
            existing_stops = self._position_stops.get(symbol)
            if existing_stops is not None and len(existing_stops) == 2:
                self._position_stops[symbol] = (
                    existing_stops[0], existing_stops[1], filled.ts,
                )

        self._positions[symbol] = updated
        self._store.write_position(updated)

    # --- Growth / bankroll compounding -------------------------------------

    def _apply_bankroll_close(self, symbol: str, realized_pnl: float) -> None:
        """Compound the synthetic bankroll by a closed trade's realized PnL.

        No-op when growth mode is off. Logs the close to ``growth_ledger``,
        records any newly-crossed milestone, and halts the loop when the
        bankroll reaches ``stop_at`` (or depletes to zero).
        """
        cfg = self._bankroll_cfg
        if cfg is None:
            return
        prev = self._bankroll
        # Clamp at zero: a synthetic bankroll cannot go negative (the real
        # testnet account still covers the loss as margin).
        self._bankroll = max(0.0, prev + realized_pnl)
        self._store.write_ledger(
            kind="close",
            bankroll=self._bankroll,
            delta=realized_pnl,
            symbol=symbol,
            note=f"realized={realized_pnl:.4f}",
        )
        logger.info(
            "Bankroll %s -> %.4f (realized=%.4f on %s)",
            prev, self._bankroll, realized_pnl, symbol,
        )

        # Track the peak for the drawdown halt (scalping risk control).
        self._bankroll_peak = max(self._bankroll_peak, self._bankroll)

        # Feed the drift kill-switch (Phase 3): record this close as a
        # bankroll-fraction so the rolling window stays comparable as the
        # bankroll compounds, tagged with the setup that opened the position.
        if self._drift_killswitch is not None:
            # Skip when the bankroll is dust (near-depleted) so a near-zero
            # ``prev`` cannot blow the fraction to inf and poison the rolling
            # window; the loop is halting on depletion anyway. Non-finite
            # realized PnL (a bad testnet fill) is coerced to a full-loss
            # inside the kill-switch.
            if prev > cfg.start * 1e-6:
                frac = realized_pnl / prev
            else:
                frac = 0.0
            entry_order = self._position_entry_orders.get(symbol)
            setup_id = (
                self._store.decision_setup_id(entry_order) if entry_order else None
            )
            self._drift_killswitch.record(frac, setup_id)

        # Milestones: log the first time the bankroll crosses each upward.
        for m in cfg.milestones:
            if m not in self._bankroll_milestones_hit and self._bankroll >= m:
                self._bankroll_milestones_hit.add(m)
                self._store.write_ledger(
                    kind="milestone",
                    bankroll=self._bankroll,
                    delta=self._bankroll - self._bankroll_start,
                    note=f"crossed {m} USDT",
                )
                logger.info("Bankroll milestone reached: %.2f USDT (from %.2f)", m, self._bankroll_start)
                self._store.write_event(
                    kind="growth_milestone",
                    severity="info",
                    payload_json=json.dumps({"milestone": m, "bankroll": self._bankroll}),
                )

        # Halt conditions: reached profit target, or bankroll depleted.
        if cfg.stop_at is not None and self._bankroll >= cfg.stop_at:
            logger.info(
                "Bankroll reached stop_at=%.2f (current=%.4f); halting loop.",
                cfg.stop_at, self._bankroll,
            )
            self._store.write_ledger(
                kind="halt",
                bankroll=self._bankroll,
                delta=self._bankroll - self._bankroll_start,
                note=f"stop_at={cfg.stop_at} reached",
            )
            self._store.halt(f"bankroll reached stop_at={cfg.stop_at}")
            self._running = False
        elif self._bankroll <= 0.0:
            logger.warning("Bankroll depleted to 0; halting loop.")
            self._store.write_ledger(
                kind="halt", bankroll=self._bankroll, note="bankroll depleted"
            )
            self._store.halt("bankroll depleted")
            self._running = False
        elif (
            cfg.max_drawdown is not None
            and self._bankroll_peak > 0
            and (self._bankroll_peak - self._bankroll) / self._bankroll_peak >= cfg.max_drawdown
        ):
            drawdown = (self._bankroll_peak - self._bankroll) / self._bankroll_peak
            logger.warning(
                "Bankroll drawdown %.1f%% >= max_drawdown %.2f; halting loop.",
                drawdown * 100, cfg.max_drawdown,
            )
            self._store.write_ledger(
                kind="halt",
                bankroll=self._bankroll,
                delta=self._bankroll - self._bankroll_start,
                note=f"drawdown {drawdown:.1%} >= {cfg.max_drawdown:.1%}",
            )
            self._store.halt(f"bankroll drawdown {drawdown:.2%} >= {cfg.max_drawdown:.2%}")
            self._running = False

        # Drift kill-switch (Phase 3): halt when live win-rate / expectancy
        # drifts below the research edge on fresh bars — the out-of-sample
        # guardrail against the in-sample matrix being overfit. Independent of
        # the bankroll halts above so it trips on a slow edge-bleed that has not
        # (yet) hit the drawdown limit.
        if self._running and self._drift_killswitch is not None:
            verdict = self._drift_killswitch.check()
            if verdict.halt and verdict.reason:
                logger.warning("Drift kill-switch tripped: %s; halting loop.", verdict.reason)
                self._store.write_ledger(
                    kind="halt",
                    bankroll=self._bankroll,
                    delta=self._bankroll - self._bankroll_start,
                    note=verdict.reason,
                )
                self._store.write_event(
                    kind="drift_killswitch",
                    severity="warning",
                    payload_json=json.dumps({
                        "reason": verdict.reason,
                        "win_rate": verdict.win_rate,
                        "expectancy": verdict.expectancy,
                        "setup_id": verdict.setup_id,
                    }),
                )
                self._store.halt(verdict.reason)
                self._running = False

    # --- Decision journaling ---------------------------------------------------

    def _write_decision(
        self,
        prediction: LivePrediction,
        order: Order,
        equity: float,
        close_price: float,
    ) -> None:
        """Persist a TradeDecision snapshot for the journal.

        Reads the indicator snapshot from the strategy (if available)
        and writes a complete decision record to the LiveStore.
        """
        strategy_name = type(self._strategy).__name__ if self._strategy else "None"
        snapshot: dict[str, float | None] = {}
        justifications: tuple[str, ...] = ()
        confluence: dict[str, float] = {}

        if self._strategy is not None and hasattr(self._strategy, "last_indicator_snapshot"):
            snapshot = self._strategy.last_indicator_snapshot
        if self._strategy is not None and hasattr(self._strategy, "last_justifications"):
            justifications = self._strategy.last_justifications
        if self._strategy is not None and hasattr(self._strategy, "last_confluence_scores"):
            confluence = self._strategy.last_confluence_scores

        # Build indicator snapshot from strategy output
        indicators = IndicatorSnapshot(
            ema_fast=snapshot.get("ema_fast"),
            ema_slow=snapshot.get("ema_slow"),
            rsi_14=snapshot.get("rsi_14"),
            atr_14=snapshot.get("atr_14"),
            macd_line=snapshot.get("macd_line"),
            macd_signal=snapshot.get("macd_signal"),
            macd_histogram=snapshot.get("macd_histogram"),
            adx=snapshot.get("adx"),
            bollinger_upper=snapshot.get("bollinger_upper"),
            bollinger_mid=snapshot.get("bollinger_mid"),
            bollinger_lower=snapshot.get("bollinger_lower"),
            stochastic_k=snapshot.get("stochastic_k"),
            stochastic_d=snapshot.get("stochastic_d"),
            cci=snapshot.get("cci"),
            williams_r=snapshot.get("williams_r"),
            obv=snapshot.get("obv"),
            vwap=snapshot.get("vwap"),
            volume_vs_avg=snapshot.get("volume_vs_avg"),
            regime_prob_trending=snapshot.get("regime_prob_trending"),
            regime_prob_ranging=snapshot.get("regime_prob_ranging"),
            regime_prob_volatile=snapshot.get("regime_prob_volatile"),
            regime_prob_stressed=snapshot.get("regime_prob_stressed"),
            bos_direction=snapshot.get("bos_direction"),
            swing_high=snapshot.get("swing_high"),
            swing_low=snapshot.get("swing_low"),
            close=snapshot.get("close", close_price),
            high=snapshot.get("high"),
            low=snapshot.get("low"),
            volume=snapshot.get("volume"),
            setup_id=snapshot.get("setup_id"),
            regime=snapshot.get("regime"),
        )

        # Build risk snapshot
        atr_val = snapshot.get("atr_14", close_price * 0.01)
        risk = RiskSnapshot(
            sl_price=snapshot.get("sl_price"),
            tp_price=snapshot.get("tp_price"),
            position_size_fraction=None,  # filled later by sizer
            equity_at_signal=equity,
            atr_distance_pct=(atr_val / close_price) if atr_val and close_price > 0 else None,
        )

        decision = TradeDecision(
            order_id=order.id,
            symbol=order.symbol,
            timestamp=order.ts,
            strategy_name=strategy_name,
            direction=prediction.direction,
            confidence=prediction.confidence,
            magnitude=prediction.magnitude,
            volatility=prediction.volatility,
            horizon=prediction.horizon,
            trend_score=confluence.get("trend"),
            momentum_score=confluence.get("momentum"),
            structure_score=confluence.get("structure"),
            volume_score=confluence.get("volume"),
            indicators=indicators,
            risk=risk,
            justifications=prediction.justifications or justifications,
        )

        try:
            self._store.write_decision(decision)
        except Exception as e:
            logger.warning("Failed to write trade decision: %s", e)

    def _accumulate_bar(self, symbol: str, bar: Any) -> None:
        """Accumulate a bar into the rolling buffer for the strategy."""
        if symbol not in self._bar_buffer:
            self._bar_buffer[symbol] = deque(maxlen=100)
        self._bar_buffer[symbol].append(self._bar_to_dict(bar))

    def prewarm(self, symbol: str, bars: Any) -> int:
        """Seed the rolling bar buffer with history WITHOUT placing orders.

        Fills ``_bar_buffer[symbol]`` with each historical bar and advances
        ``_tick_count`` past the warm-up threshold, but never reaches the order
        path (no equity fetch, no guardian, no broker call). After prewarm, the
        first LIVE bar from the feed trades immediately and the strategy acts on
        the live bar's close — not a stale historical close — so attached TP/SL
        land on the correct side of the current market price (the stale-history
        replay that previously made Bybit reject TP/SL on the wrong side). The
        seeded bars are buffer-only; they are never journaled as decisions.

        Returns the number of bars seeded.
        """
        n = int(bars.num_rows)
        for i in range(n):
            self._accumulate_bar(symbol, bars.slice(i, 1))
        warmup = self._strategy.warmup_bars if self._strategy else self._config.warmup_bars
        if self._tick_count <= warmup:
            self._tick_count = warmup + 1
        logger.info(
            "Prewarmed %s with %d history bars (tick_count=%d, warmup=%d)",
            symbol, n, self._tick_count, warmup,
        )
        return n

    def _make_prediction(self, symbol: str) -> LivePrediction:
        """Create a LivePrediction using the strategy or fallback to neutral."""
        if self._strategy is not None:
            # Phase 4b: refresh the strategy's order-book snapshot from the
            # provider's cache before predict (no-op when no provider is set,
            # preserving the legacy behaviour byte-for-byte). The provider is
            # expected to return an OrderFlowSnapshot or None; a failure must
            # never crash the loop — fall back to "no order-flow signal".
            if self._orderflow_provider is not None and getattr(
                self._strategy, "use_orderflow", False
            ):
                try:
                    self._strategy.last_orderflow = self._orderflow_provider(symbol)
                except Exception as e:
                    logger.debug("orderflow provider failed for %s: %s", symbol, e)
                    self._strategy.last_orderflow = None
            bar_table = self._buffer_to_table(symbol)
            if bar_table is not None and bar_table.num_rows >= self._strategy.warmup_bars:
                return self._strategy.predict(bar_table, symbol)

        # Fallback: neutral prediction
        return LivePrediction(
            symbol=symbol,
            direction=0.0,
            magnitude=0.0,
            volatility=0.01,
            confidence=0.0,
            horizon=self._config.horizon,
            ts=_utc_now_iso(),
        )

    def _bar_to_dict(self, bar: Any) -> dict:
        """Convert a bar table to a dict for the buffer."""
        with contextlib.suppress(Exception):
            if hasattr(bar, "column"):
                close = float(bar.column("close")[0].as_py()) if hasattr(bar.column("close")[0], "as_py") else float(bar.column("close")[0])
                return {
                    "open": float(bar.column("open")[0].as_py()) if hasattr(bar.column("open")[0], "as_py") else float(bar.column("open")[0]),
                    "high": float(bar.column("high")[0].as_py()) if hasattr(bar.column("high")[0], "as_py") else float(bar.column("high")[0]),
                    "low": float(bar.column("low")[0].as_py()) if hasattr(bar.column("low")[0], "as_py") else float(bar.column("low")[0]),
                    "close": close,
                    "volume": float(bar.column("volume")[0].as_py()) if hasattr(bar.column("volume")[0], "as_py") else float(bar.column("volume")[0]),
                }
        return {}

    def _buffer_to_table(self, symbol: str) -> pa.Table | None:
        """Convert the rolling bar buffer to a pyarrow Table for the strategy."""
        from kairon.data.io import OHLCV_SCHEMA

        if symbol not in self._bar_buffer or len(self._bar_buffer[symbol]) == 0:
            return None

        bars = self._bar_buffer[symbol]
        try:
            # Synthetic monotonically-increasing timestamps (the strategy
            # ignores ts). Built via timedelta so a buffer longer than 60 bars
            # does not overflow the minute field — ``datetime(2026,1,1,0,i)``
            # raised ValueError for i>=60, which was swallowed by the bare
            # ``except`` below, silently returning None and forcing every tick
            # to the neutral fallback (zero live trades despite valid signals).
            base_ts = datetime(2026, 1, 1, tzinfo=UTC)
            return pa.table(
                {
                    "ts": [base_ts + timedelta(minutes=i) for i in range(len(bars))],
                    "open": [b["open"] for b in bars],
                    "high": [b["high"] for b in bars],
                    "low": [b["low"] for b in bars],
                    "close": [b["close"] for b in bars],
                    "volume": [b["volume"] for b in bars],
                },
                schema=OHLCV_SCHEMA,
            )
        except Exception:
            logger.exception("Failed to reconstruct bar table for %s", symbol)
            return None

    # --- Helpers ---------------------------------------------------------------

    def _extract_symbol(self, bar: Any) -> str:
        """Extract the symbol from a bar table. Falls back to first configured symbol."""
        with contextlib.suppress(Exception):
            if hasattr(bar, "column") and "symbol" in bar.column_names:
                return str(bar.column("symbol")[0])
        return self._config.symbols[0] if self._config.symbols else "BTC-USDT-PERP"

    def _extract_close(self, bar: Any) -> float:
        """Extract the close price from a bar table."""
        with contextlib.suppress(Exception):
            if hasattr(bar, "column"):
                for col_name in ("close", "Close", "c"):
                    if col_name in bar.column_names:
                        val = bar.column(col_name)[0]
                        return float(val.as_py() if hasattr(val, "as_py") else val)
        raise ValueError("Cannot extract close price from bar")

    def _get_usdt_equity(self, balances: list[Any]) -> float:
        """Extract USDT equity from broker balances."""
        for b in balances:
            if b.currency == "USDT":
                return b.total
        return sum(b.total for b in balances) if balances else 0.0


__all__ = ["TradingLoop"]
