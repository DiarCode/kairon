"""One-shot backtest + performance + DSR evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairon.backtest.engine import (
    BacktestResult,
    BacktestSpec,
    run_backtest,
)
from kairon.backtest.metrics import (
    BARS_PER_YEAR_1D,
    PerformanceReport,
    summarize,
)
from kairon.backtest.statistics import (
    DSRResult,
    DSRSpec,
    deflated_sharpe_ratio,
)


@dataclass(frozen=True, slots=True)
class BacktestEvaluation:
    """The full output of :func:`backtest_and_evaluate`."""

    backtest: BacktestResult
    performance: PerformanceReport
    dsr: DSRResult
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backtest": {
                "symbol": self.backtest.symbol,
                "n_trades": self.backtest.n_trades,
                "total_pnl": self.backtest.total_pnl,
                "win_rate": self.backtest.win_rate,
                "final_equity": self.backtest.final_equity,
            },
            "performance": self.performance.to_dict(),
            "dsr": {
                "sharpe": self.dsr.sharpe,
                "dsr": self.dsr.dsr,
                "p_value": self.dsr.p_value,
                "sr_star": self.dsr.sr_star,
            },
        }


def backtest_and_evaluate(
    *,
    symbol: str,
    timestamps: np.ndarray,
    close: np.ndarray,
    signals: np.ndarray,
    backtest_spec: BacktestSpec | None = None,
    dsr_spec: DSRSpec | None = None,
    bars_per_year: int = BARS_PER_YEAR_1D,
) -> BacktestEvaluation:
    """Run a backtest and compute performance + DSR in one pass.

    The DSR's ``n_trials`` is set to 1 (single backtest) by default;
    the caller should override it if the backtest is one of many
    variants tried in the same study.
    """
    backtest_spec = backtest_spec or BacktestSpec()
    dsr_spec = dsr_spec or DSRSpec()
    result = run_backtest(
        symbol=symbol,
        timestamps=timestamps,
        close=close,
        signals=signals,
        spec=backtest_spec,
    )
    equity = np.asarray(result.equity_curve, dtype=np.float64)
    trade_pnl = (
        np.asarray([t.pnl for t in result.trades], dtype=np.float64)
        if result.trades
        else np.zeros(0, dtype=np.float64)
    )
    perf = summarize(
        equity,
        bars_per_year=bars_per_year,
        trade_pnl=trade_pnl if trade_pnl.size else None,
    )
    rets = np.diff(equity) / equity[:-1] if equity.size > 1 else np.zeros(0)
    dsr = deflated_sharpe_ratio(rets, spec=dsr_spec)
    return BacktestEvaluation(
        backtest=result,
        performance=perf,
        dsr=dsr,
    )


__all__ = ["BacktestEvaluation", "backtest_and_evaluate"]
