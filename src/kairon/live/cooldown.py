"""Order-cooldown wrapper and trading loop.

Suppresses rapid rebalancing during short sessions: after an order is
accepted for a symbol, new signals for that symbol are flattened for
``cooldown_seconds``. Extracted from ``scripts/run_testnet_symbol.py`` so
the in-process :class:`~kairon.live.host.SessionHost` can reuse it without
importing from the ``scripts`` package (which is not an installed module).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from kairon.live.broker.base import Broker, Order, OrderStatus
from kairon.live.orchestrator import TradingLoop
from kairon.live.predictor import LivePrediction

DEFAULT_COOLDOWN_SECONDS = 5 * 60  # one trade every 5 minutes per symbol


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class CooldownBrokerWrapper:
    """Wrap a :class:`Broker` and record the last accepted order timestamp per symbol.

    Used by :class:`CooledTradingLoop` to suppress rapid rebalancing during
    short sessions. Generic over the broker protocol so it works with both
    :class:`~kairon.live.broker.bybit.BybitBroker` and
    :class:`~kairon.live.broker.paper.PaperBroker`.
    """

    def __init__(
        self,
        inner: Broker,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._inner = inner
        self.cooldown_seconds = cooldown_seconds
        self._last_order_ts: dict[str, float] = {}

    async def place_order(self, order: Order) -> Order:
        result = await self._inner.place_order(order)
        if result.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
            self._last_order_ts[order.symbol] = time.time()
        return result

    def is_cooling_down(self, symbol: str, now: float | None = None) -> bool:
        """True if ``symbol`` is within its post-order cooldown window."""
        last = self._last_order_ts.get(symbol)
        if last is None:
            return False
        return (now if now is not None else time.time()) - last < self.cooldown_seconds

    def __getattr__(self, name: str) -> Any:
        # Proxy attribute access to the wrapped broker; return type is
        # genuinely arbitrary because it forwards any broker attribute.
        return getattr(self._inner, name)


class CooledTradingLoop(TradingLoop):
    """:class:`TradingLoop` that ignores new signals while a symbol is in cooldown."""

    def _make_prediction(self, symbol: str) -> LivePrediction:
        broker = self._broker
        if isinstance(broker, CooldownBrokerWrapper) and broker.is_cooling_down(symbol):
            return LivePrediction(
                symbol=symbol,
                direction=0.0,
                magnitude=0.0,
                volatility=0.01,
                confidence=0.0,
                horizon=self._config.horizon,
                ts=_utc_now_iso(),
            )
        return super()._make_prediction(symbol)


__all__ = ["DEFAULT_COOLDOWN_SECONDS", "CooldownBrokerWrapper", "CooledTradingLoop"]
