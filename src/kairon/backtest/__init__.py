"""Backtest subpackage."""

from kairon.backtest.cost import (
    DEFAULT_CRYPTO_COSTS,
    DEFAULT_STOCK_COSTS,
    CostModel,
)
from kairon.backtest.engine import (
    BacktestResult,
    BacktestSpec,
    Trade,
    run_backtest,
    signals_to_target,
)
from kairon.backtest.evaluate import BacktestEvaluation, backtest_and_evaluate
from kairon.backtest.metrics import (
    BARS_PER_YEAR_1D,
    BARS_PER_YEAR_1H,
    BARS_PER_YEAR_5M,
    PerformanceReport,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    win_rate,
)
from kairon.backtest.position import Position, Side
from kairon.backtest.statistics import (
    DSRResult,
    DSRSpec,
    PBOResult,
    deflated_sharpe_ratio,
    probability_of_backtest_overfit,
)

__all__ = [
    "BARS_PER_YEAR_1D",
    "BARS_PER_YEAR_1H",
    "BARS_PER_YEAR_5M",
    "DEFAULT_CRYPTO_COSTS",
    "DEFAULT_STOCK_COSTS",
    "BacktestEvaluation",
    "BacktestResult",
    "BacktestSpec",
    "CostModel",
    "DSRResult",
    "DSRSpec",
    "PBOResult",
    "PerformanceReport",
    "Position",
    "Side",
    "Trade",
    "backtest_and_evaluate",
    "deflated_sharpe_ratio",
    "max_drawdown",
    "probability_of_backtest_overfit",
    "profit_factor",
    "run_backtest",
    "sharpe_ratio",
    "signals_to_target",
    "sortino_ratio",
    "summarize",
    "win_rate",
]
