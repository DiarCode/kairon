# ADR-0008: No LLM for numeric prediction or numeric evaluation

**Status:** Accepted
**Date:** 2026-06-05

## Context
This is a clarification of ADR-0005. The user explicitly required the LLM layer; we have to use it. The risk is that the LLM silently becomes a "smart-sounding" numeric oracle. This ADR is the explicit rule and the test.

## Decision
- LLMs cannot:
  - Predict prices, returns, vol, or probabilities.
  - Compute Sharpe, Sortino, PnL, hit rate, Brier, ECE.
  - Judge whether a model is "good" or "bad".
  - Choose hyperparameters.
- LLMs can:
  - Summarize backtest results in plain English.
  - Explain why a driver is high or low.
  - Compare two signals and articulate the difference.
  - Plan research (read-only agent).
- A typed guard in `kairon.llm.guardrails.assert_no_numeric_authority` raises if a prompt asks for the forbidden outputs. The CI test suite enforces this on a corpus of bad prompts.

## Consequences
- Hard line in the code.
- Easy to explain in user-facing docs.
- One explicit class of bugs eliminated.

## Alternatives considered
- "Just be careful" — rejected; not enforceable.
- Sandbox the LLM to a separate process — considered for v2.
