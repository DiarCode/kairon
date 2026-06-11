# ADR-0006: Cost-aware backtest by default

**Status:** Accepted
**Date:** 2026-06-05

## Context
R-001 V.F shows that a 60% accurate S&P 500 model has **negative expected return** with naive daily trading. Costs are not optional. Bhand & Joshi 2026 show leakage inflates Sharpe by 6-10x; cost realism is the other half of the honesty story.

## Decision
- Every backtest applies a **typed `CostModel`**: commission, spread, market impact (square-root of `order_size / ADV`), funding (perps), borrow (shorts), slippage.
- `kairon.backtest.cost_model.CostModel` is a pydantic schema; the same schema is used in paper trading and live inference.
- Headline metrics are reported **post-cost**. Pre-cost is shown as a secondary line for diagnostic purposes only.
- The "default" cost model is conservative: commission 0.05% spot, spread 1 tick, market impact square-root with conservative ADV factor, slippage 1 tick.

## Consequences
- Lower headline Sharpe numbers, but realistic.
- Forces realistic position sizing.
- Prevents a class of "profitable in backtest, ruinous in production" mistakes.

## Alternatives considered
- Zero-cost backtest: rejected (R-001 evidence).
- Per-exchange real-time cost: deferred to v2 (requires live L2 order book + historical reconstructions).
