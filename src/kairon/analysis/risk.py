"""Risk level calculation — stop loss, take profit, position sizing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskLevels:
    """Stop-loss, take-profit, and position sizing for the current bar."""

    stop_loss_long: float  # 2x ATR below current price
    stop_loss_short: float  # 2x ATR above current price
    stop_loss_long_tight: float  # 1.5x ATR below current price
    stop_loss_short_tight: float  # 1.5x ATR above current price
    take_profit_long_1: float  # 2x ATR above (1:1 R:R for longs)
    take_profit_long_2: float  # 3x ATR above (1:1.5 R:R for longs)
    take_profit_short_1: float  # 2x ATR below (1:1 R:R for shorts)
    take_profit_short_2: float  # 3x ATR below (1:1.5 R:R for shorts)
    fib_tp_long: float  # Fibonacci 1.618 extension above
    fib_tp_short: float  # Fibonacci 1.618 extension below
    position_size_pct: float  # As fraction of equity (0.005-0.02)
    atr: float  # ATR(14) value used for calculations


def calculate_risk_levels(
    current_price: float,
    *,
    atr: float,
    fib_tp_long: float = 0.0,
    fib_tp_short: float = 0.0,
    garch_vol: float = 0.0,
    equity: float = 10000.0,
    risk_pct: float = 0.01,
) -> RiskLevels:
    """Calculate stop-loss, take-profit, and position sizing levels.

    Parameters
    ----------
    current_price : float
        Current close price.
    atr : float
        ATR(14) value.
    fib_tp_long : float
        Fibonacci 1.618 extension above (from pivots). 0 if not available.
    fib_tp_short : float
        Fibonacci 1.618 extension below (from pivots). 0 if not available.
    garch_vol : float
        GARCH conditional volatility (used to adjust position size).
    equity : float
        Portfolio equity for position sizing.
    risk_pct : float
        Risk per trade as fraction of equity (default 0.01 = 1%).

    Returns
    -------
    RiskLevels
        All SL/TP levels and position sizing.
    """
    # ATR-based levels
    sl_long = current_price - 2 * atr
    sl_short = current_price + 2 * atr
    sl_long_tight = current_price - 1.5 * atr
    sl_short_tight = current_price + 1.5 * atr
    tp_long_1 = current_price + 2 * atr
    tp_long_2 = current_price + 3 * atr
    tp_short_1 = current_price - 2 * atr
    tp_short_2 = current_price - 3 * atr

    # Fib extensions (use ATR-based as fallback if not available)
    fib_long = fib_tp_long if fib_tp_long > 0 else current_price + 1.618 * atr
    fib_short = fib_tp_short if fib_tp_short > 0 else current_price - 1.618 * atr

    # Position sizing: risk = equity * risk_pct / (2 * ATR)
    # This means a 2*ATR stop loss risks exactly risk_pct of equity
    if atr > 0:
        position_size_pct = min(0.05, risk_pct * equity / (2 * atr * 100))
    else:
        position_size_pct = 0.01

    # Adjust for volatility: reduce position in high-vol regimes
    if garch_vol > 0.05:
        position_size_pct *= 0.75  # Reduce by 25% in high volatility
    elif garch_vol > 0.03:
        position_size_pct *= 0.9  # Reduce by 10% in moderate volatility

    return RiskLevels(
        stop_loss_long=sl_long,
        stop_loss_short=sl_short,
        stop_loss_long_tight=sl_long_tight,
        stop_loss_short_tight=sl_short_tight,
        take_profit_long_1=tp_long_1,
        take_profit_long_2=tp_long_2,
        take_profit_short_1=tp_short_1,
        take_profit_short_2=tp_short_2,
        fib_tp_long=fib_long,
        fib_tp_short=fib_short,
        position_size_pct=position_size_pct,
        atr=atr,
    )
