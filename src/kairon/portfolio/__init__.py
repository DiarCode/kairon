"""Risk & portfolio layer.

This module provides:

- **Position sizing**: fixed-fraction, Kelly-fraction, and vol-targeted.
- **Exposure limits**: per-position and total-leverage caps.
- **Portfolio aggregation**: combine per-symbol signal strength into a
  single ``PortfolioSignal`` (long / short / flat per symbol + a
  target portfolio weight vector).

It is *not* a broker. It just decides *how much* to trade given a
signal; the :class:`kairon.paper.PaperTrader` (or a real broker) is
responsible for actually placing the order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SizingConfig:
    """Configuration for position sizing.

    ``method`` selects the sizing strategy:

    - ``"fixed_fraction"``: a constant fraction of equity per trade.
    - ``"kelly"``: a *fractional* Kelly bet — ``fraction * (p - q) / b``
      where ``p`` is the win rate, ``q = 1 - p``, and ``b`` is the
      pay-off ratio (avg_win / avg_loss).
    - ``"vol_target"``: scale position so that ``size * vol_forecast ==
      vol_target_annual``. Requires ``vol_forecast`` per call.
    """

    method: str = "fixed_fraction"
    fraction: float = 0.10  # for fixed_fraction and Kelly
    kelly_cap: float = 0.25  # never bet more than this fraction of equity
    vol_target_annual: float = 0.10  # for vol_target
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in {"fixed_fraction", "kelly", "vol_target"}:
            raise ValueError(f"unsupported method: {self.method!r}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if not 0.0 < self.kelly_cap <= 1.0:
            raise ValueError(f"kelly_cap must be in (0, 1], got {self.kelly_cap}")
        if self.vol_target_annual <= 0:
            raise ValueError(f"vol_target_annual must be > 0, got {self.vol_target_annual}")


def fixed_fraction_size(*, equity: float, price: float, fraction: float) -> float:
    """Return ``equity * fraction / price`` units."""
    if equity <= 0:
        raise ValueError(f"equity must be > 0, got {equity}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return (equity * fraction) / price


def kelly_size(
    *,
    equity: float,
    price: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    cap: float = 0.25,
) -> float:
    """Fractional Kelly sizing.

    ``kelly_fraction = (p - q) / b`` where ``p = win_rate``, ``q = 1 - p``,
    and ``b = avg_win / avg_loss``. The result is *clipped* to
    ``[-cap, cap]`` (long-only: clipped to ``[0, cap]``).
    """
    if equity <= 0:
        raise ValueError(f"equity must be > 0, got {equity}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")
    if not 0.0 < win_rate < 1.0:
        raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
    if avg_win <= 0 or avg_loss <= 0:
        raise ValueError(f"avg_win and avg_loss must be > 0, got {avg_win=}, {avg_loss=}")
    if not 0.0 < cap <= 1.0:
        raise ValueError(f"cap must be in (0, 1], got {cap}")
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly = (win_rate - q) / b
    clipped = max(0.0, min(cap, kelly))
    return (equity * clipped) / price


def vol_target_size(
    *,
    equity: float,
    price: float,
    vol_forecast_annual: float,
    vol_target_annual: float = 0.10,
) -> float:
    """Size so that the position's annualised vol matches the target."""
    if vol_forecast_annual <= 0:
        raise ValueError(f"vol_forecast_annual must be > 0, got {vol_forecast_annual}")
    if vol_target_annual <= 0:
        raise ValueError(f"vol_target_annual must be > 0, got {vol_target_annual}")
    notional = equity * (vol_target_annual / vol_forecast_annual)
    return notional / price


def size_position(
    *,
    equity: float,
    price: float,
    config: SizingConfig,
    win_rate: float | None = None,
    avg_win: float | None = None,
    avg_loss: float | None = None,
    vol_forecast_annual: float | None = None,
) -> float:
    """Dispatch to the configured sizing method."""
    if config.method == "fixed_fraction":
        return fixed_fraction_size(equity=equity, price=price, fraction=config.fraction)
    if config.method == "kelly":
        if win_rate is None or avg_win is None or avg_loss is None:
            raise ValueError("kelly sizing needs win_rate, avg_win, avg_loss")
        return kelly_size(
            equity=equity,
            price=price,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            cap=config.kelly_cap,
        )
    if config.method == "vol_target":
        if vol_forecast_annual is None:
            raise ValueError("vol_target sizing needs vol_forecast_annual")
        return vol_target_size(
            equity=equity,
            price=price,
            vol_forecast_annual=vol_forecast_annual,
            vol_target_annual=config.vol_target_annual,
        )
    raise ValueError(f"unsupported method: {config.method!r}")


# ---------------------------------------------------------------------------
# Exposure limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExposureLimits:
    """Per-position and total-leverage caps."""

    max_position_equity_fraction: float = 0.20  # max single position as % of equity
    max_total_leverage: float = 1.0  # max sum(|position notional|) / equity
    max_positions: int = 10
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_position_equity_fraction <= 1.0:
            raise ValueError(
                f"max_position_equity_fraction must be in (0, 1], got {self.max_position_equity_fraction}"
            )
        if self.max_total_leverage <= 0:
            raise ValueError(f"max_total_leverage must be > 0, got {self.max_total_leverage}")
        if self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")


def check_exposure(
    *,
    candidate_symbol: str,
    candidate_size: float,
    candidate_price: float,
    existing: dict[str, tuple[float, float]],  # symbol -> (size, price)
    equity: float,
    limits: ExposureLimits,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``.

    The check enforces three things:

    1. Single position size: ``|size * price| / equity <= max_position_equity_fraction``.
    2. Total leverage: ``sum(|size * price|) / equity <= max_total_leverage``.
    3. Number of positions: ``len(existing) + 1 <= max_positions``.
    """
    if equity <= 0:
        return False, f"equity must be > 0, got {equity}"
    if candidate_size == 0 or candidate_price <= 0:
        return False, "candidate_size must be non-zero, candidate_price > 0"
    candidate_notional = abs(candidate_size * candidate_price)
    if candidate_notional / equity > limits.max_position_equity_fraction:
        return False, (
            f"position notional {candidate_notional:.2f} exceeds "
            f"{limits.max_position_equity_fraction:.0%} of equity"
        )
    total = candidate_notional + sum(abs(s * p) for s, p in existing.values())
    if total / equity > limits.max_total_leverage:
        return False, (
            f"total exposure {total:.2f} exceeds leverage cap "
            f"{limits.max_total_leverage:.2f}x equity"
        )
    if candidate_symbol not in existing and len(existing) >= limits.max_positions:
        return False, f"max_positions ({limits.max_positions}) reached"
    return True, "ok"


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PortfolioSignal:
    """A single point-in-time view of the model's portfolio view."""

    weights: dict[str, float]  # symbol -> weight in [-1, 1]
    gross: float  # sum(|w|)
    net: float  # sum(w) — sign indicates bias
    n_long: int
    n_short: int
    n_flat: int
    extras: dict[str, Any] = field(default_factory=dict)


def aggregate_signals(
    signals: dict[str, float],
    *,
    confidence_floor: float = 0.0,
) -> PortfolioSignal:
    """Convert raw per-symbol signals to portfolio weights.

    Each signal is clipped to ``[-1, 1]``. Signals whose absolute value
    is below ``confidence_floor`` are treated as flat. The result is a
    :class:`PortfolioSignal` with the per-symbol weights and a few
    summary stats.
    """
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError(f"confidence_floor must be in [0, 1], got {confidence_floor}")
    weights: dict[str, float] = {}
    n_long = n_short = n_flat = 0
    for sym, raw in signals.items():
        w = max(-1.0, min(1.0, float(raw)))
        if abs(w) < confidence_floor:
            w = 0.0
        weights[sym] = w
        if w > 0:
            n_long += 1
        elif w < 0:
            n_short += 1
        else:
            n_flat += 1
    arr = np.array(list(weights.values()), dtype=np.float64)
    return PortfolioSignal(
        weights=weights,
        gross=float(np.abs(arr).sum()),
        net=float(arr.sum()),
        n_long=n_long,
        n_short=n_short,
        n_flat=n_flat,
    )


__all__ = [
    "ExposureLimits",
    "PortfolioSignal",
    "SizingConfig",
    "aggregate_signals",
    "check_exposure",
    "fixed_fraction_size",
    "kelly_size",
    "size_position",
    "vol_target_size",
]
