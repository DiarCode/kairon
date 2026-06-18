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
import json
import logging
import uuid
from collections import deque
from datetime import UTC, datetime
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
from kairon.live.config import LiveConfig
from kairon.live.guardian import Guardian
from kairon.live.journal import IndicatorSnapshot, RiskSnapshot, TradeDecision
from kairon.live.predictor import LivePrediction
from kairon.live.reconciler import Reconciler
from kairon.live.store import LiveStore

logger = logging.getLogger(__name__)


def _uuid7() -> str:
    """Generate a short unique ID for orders and traces."""
    return uuid.uuid4().hex[:24]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    ) -> None:
        self._config = config
        self._broker = broker
        self._predictor = predictor
        self._strategy = strategy
        self._guardian = guardian
        self._reconciler = reconciler
        self._store = store
        self._feed = feed

        self._tick_count: int = 0
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._fill_task: asyncio.Task | None = None

        # Track current positions for diffing
        self._positions: dict[str, Position] = {}

        # Track which order opened each position (for decision outcome tracking)
        self._position_entry_orders: dict[str, str] = {}

        # Rolling bar buffer for the strategy (stores raw bar dicts)
        self._bar_buffer: dict[str, deque[dict]] = {}

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

        Call after :meth:`stop` and *before* flattening, while the position still
        exists on the broker.
        """
        if not self._position_entry_orders:
            return
        now_ts = _utc_now_iso()
        for symbol, order_id in list(self._position_entry_orders.items()):
            try:
                positions = await self._broker.get_positions(symbol)
            except Exception as e:
                logger.warning("finalize_open_positions: get_positions failed for %s: %s", symbol, e)
                continue
            pos = next((p for p in positions if p.symbol == symbol), None)
            realized = float(pos.unrealized_pnl or 0.0) if pos and pos.qty > 1e-9 else 0.0
            try:
                self._store.update_decision_outcome(
                    order_id=order_id,
                    outcome="manual_close",
                    pnl=round(realized, 8),
                    ts=now_ts,
                )
            except Exception:
                logger.debug("finalize_open_positions: could not update outcome for %s", order_id)
                continue
            logger.info(
                "Finalized open decision %s for %s: outcome=manual_close pnl=%.6f",
                order_id, symbol, realized,
            )

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

    async def _process_tick(self, bar: Any) -> None:
        """Process a single closed bar (one tick)."""
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

        delta = target_qty - current_signed

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

        order = Order(
            id=_uuid7(),
            intent_id=_uuid7(),
            trace_id=_uuid7(),
            symbol=symbol,
            side=side,
            qty=order_qty,
            order_type=OrderType.MARKET,
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
        daily_alert = self._guardian.check_daily_loss(0.0, equity)

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
                logger.info(
                    "Fill drain: %s %s qty=%.6f @ %.2f fee=%.4f",
                    fill.side.value, fill.symbol, fill.qty, fill.price, fill.fee,
                )
        except asyncio.CancelledError:
            logger.info("Fill drain task cancelled")
        except Exception:
            logger.exception("Fill drain task crashed")
            raise

    # --- Position tracking -------------------------------------------------------

    def _update_position(
        self, symbol: str, side: OrderSide, qty: float, price: float, filled: Order
    ) -> None:
        """Update the position tracking dict, persist to store, and record closed trades."""
        # Track entry timestamp for duration calculation
        if not hasattr(self, '_position_entry_ts'):
            self._position_entry_ts: dict[str, str] = {}

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

        self._positions[symbol] = updated
        self._store.write_position(updated)

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

    def _make_prediction(self, symbol: str) -> LivePrediction:
        """Create a LivePrediction using the strategy or fallback to neutral."""
        if self._strategy is not None:
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
        try:
            if hasattr(bar, "column"):
                close = float(bar.column("close")[0].as_py()) if hasattr(bar.column("close")[0], "as_py") else float(bar.column("close")[0])
                return {
                    "open": float(bar.column("open")[0].as_py()) if hasattr(bar.column("open")[0], "as_py") else float(bar.column("open")[0]),
                    "high": float(bar.column("high")[0].as_py()) if hasattr(bar.column("high")[0], "as_py") else float(bar.column("high")[0]),
                    "low": float(bar.column("low")[0].as_py()) if hasattr(bar.column("low")[0], "as_py") else float(bar.column("low")[0]),
                    "close": close,
                    "volume": float(bar.column("volume")[0].as_py()) if hasattr(bar.column("volume")[0], "as_py") else float(bar.column("volume")[0]),
                }
        except Exception:
            pass
        return {}

    def _buffer_to_table(self, symbol: str) -> pa.Table | None:
        """Convert the rolling bar buffer to a pyarrow Table for the strategy."""
        from kairon.data.io import OHLCV_SCHEMA

        if symbol not in self._bar_buffer or len(self._bar_buffer[symbol]) == 0:
            return None

        bars = self._bar_buffer[symbol]
        try:
            return pa.table(
                {
                    "ts": [datetime(2026, 1, 1, 0, i, 0, tzinfo=UTC) for i in range(len(bars))],
                    "open": [b["open"] for b in bars],
                    "high": [b["high"] for b in bars],
                    "low": [b["low"] for b in bars],
                    "close": [b["close"] for b in bars],
                    "volume": [b["volume"] for b in bars],
                },
                schema=OHLCV_SCHEMA,
            )
        except Exception:
            return None

    # --- Helpers ---------------------------------------------------------------

    def _extract_symbol(self, bar: Any) -> str:
        """Extract the symbol from a bar table. Falls back to first configured symbol."""
        try:
            if hasattr(bar, "column") and "symbol" in bar.column_names:
                return str(bar.column("symbol")[0])
        except Exception:
            pass
        return self._config.symbols[0] if self._config.symbols else "BTC-USDT-PERP"

    def _extract_close(self, bar: Any) -> float:
        """Extract the close price from a bar table."""
        try:
            if hasattr(bar, "column"):
                for col_name in ("close", "Close", "c"):
                    if col_name in bar.column_names:
                        val = bar.column(col_name)[0]
                        return float(val.as_py() if hasattr(val, "as_py") else val)
        except Exception:
            pass
        raise ValueError("Cannot extract close price from bar")

    def _get_usdt_equity(self, balances: list[Any]) -> float:
        """Extract USDT equity from broker balances."""
        for b in balances:
            if b.currency == "USDT":
                return b.total
        return sum(b.total for b in balances) if balances else 0.0


__all__ = ["TradingLoop"]
