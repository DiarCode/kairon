"""Performance metrics: Sharpe, Sortino, drawdown, win-rate, profit factor.

All metrics are computed from an *equity curve* (a 1-D array of mark-to-
market equity values, one per bar). We assume the user is honest about
the time-step — pass ``bars_per_year`` for the annualisation factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Standard performance summary from an equity curve."""

    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_trades: int
    n_bars: int
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_vol": self.annualized_vol,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "n_trades": float(self.n_trades),
            "n_bars": float(self.n_bars),
        }


# Sensible annualisation factors
BARS_PER_YEAR_5M: Final[int] = 365 * 24 * 12  # 105,120
BARS_PER_YEAR_1H: Final[int] = 365 * 24  # 8,760
BARS_PER_YEAR_1D: Final[int] = 365


def sharpe_ratio(
    returns: np.ndarray,
    *,
    bars_per_year: int = BARS_PER_YEAR_1D,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    ``returns`` is a 1-D array of per-bar simple returns. The risk-free
    rate is given as an *annualised* number; we convert it to a per-bar
    rate via ``risk_free / bars_per_year`` before subtracting.
    """
    if returns.size < 2:
        return float("nan")
    rf_per_bar = risk_free / bars_per_year
    excess = returns - rf_per_bar
    if excess.std(ddof=0) == 0:
        return float("nan")
    return float(excess.mean() / excess.std(ddof=0) * math.sqrt(bars_per_year))


def sortino_ratio(
    returns: np.ndarray,
    *,
    bars_per_year: int = BARS_PER_YEAR_1D,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sortino ratio (penalises only downside deviation)."""
    if returns.size < 2:
        return float("nan")
    rf_per_bar = risk_free / bars_per_year
    excess = returns - rf_per_bar
    downside = excess[excess < 0]
    if downside.size == 0 or downside.std(ddof=0) == 0:
        return float("nan")
    return float(excess.mean() / downside.std(ddof=0) * math.sqrt(bars_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a *negative* fraction of the
    peak (e.g. -0.20 == 20% drawdown)."""
    if equity.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def win_rate(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return float("nan")
    return float((pnl > 0).mean())


def profit_factor(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return float("nan")
    gross_wins = pnl[pnl > 0].sum()
    gross_losses = -pnl[pnl < 0].sum()
    if gross_losses == 0:
        return float("inf") if gross_wins > 0 else float("nan")
    return float(gross_wins / gross_losses)


def summarize(
    equity: np.ndarray,
    *,
    bars_per_year: int = BARS_PER_YEAR_1D,
    risk_free: float = 0.0,
    trade_pnl: np.ndarray | None = None,
) -> PerformanceReport:
    """Compute a full :class:`PerformanceReport` from an equity curve.

    ``trade_pnl`` (optional) is the per-trade realised PnL. If absent,
    ``win_rate`` and ``profit_factor`` are NaN.
    """
    if equity.size < 2:
        return PerformanceReport(
            total_return=float("nan"),
            annualized_return=float("nan"),
            annualized_vol=float("nan"),
            sharpe=float("nan"),
            sortino=float("nan"),
            max_drawdown=float("nan"),
            calmar=float("nan"),
            win_rate=float("nan") if trade_pnl is None else win_rate(trade_pnl),
            profit_factor=float("nan") if trade_pnl is None else profit_factor(trade_pnl),
            n_trades=0 if trade_pnl is None else int(trade_pnl.size),
            n_bars=int(equity.size),
        )
    rets = np.diff(equity) / equity[:-1]
    total = float(equity[-1] / equity[0] - 1.0)
    years = max(equity.size / bars_per_year, 1e-9)
    ann_return = float((1.0 + total) ** (1.0 / years) - 1.0)
    ann_vol = float(rets.std(ddof=0) * math.sqrt(bars_per_year))
    sharpe = sharpe_ratio(rets, bars_per_year=bars_per_year, risk_free=risk_free)
    sortino = sortino_ratio(rets, bars_per_year=bars_per_year, risk_free=risk_free)
    mdd = max_drawdown(equity)
    calmar = ann_return / abs(mdd) if mdd < 0 else float("nan")
    n_trades = 0 if trade_pnl is None else int(trade_pnl.size)
    return PerformanceReport(
        total_return=total,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=win_rate(trade_pnl) if trade_pnl is not None else float("nan"),
        profit_factor=profit_factor(trade_pnl) if trade_pnl is not None else float("nan"),
        n_trades=n_trades,
        n_bars=int(equity.size),
    )


__all__ = [
    "BARS_PER_YEAR_1D",
    "BARS_PER_YEAR_1H",
    "BARS_PER_YEAR_5M",
    "PerformanceReport",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize",
    "win_rate",
]
