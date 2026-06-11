# ADR-0005: LLM is a reasoning layer, never a numeric oracle

**Status:** Accepted
**Date:** 2026-06-05

## Context
The user requires Ollama cloud for research synthesis, hypothesis generation, evidence-grounded explanation, and agent planning. LLMs hallucinate numerics, especially in finance. Letting an LLM "predict the price" would silently inject noise.

## Decision
- The LLM (`gpt-oss:120b-cloud`) is used only for: research synthesis, hypothesis generation, summarization, evidence-grounded explanation, anomaly commentary, decision support.
- The LLM **never** produces a numeric prediction, a Sharpe, a PnL, or a probability that drives a trade.
- Every LLM call is wrapped in a typed `LLMRequest` / `LLMResponse`.
- `citations_required: bool` is True for any user-facing call. Responses without citations are **rejected**.
- The LLM has no tools that mutate state.

## Consequences
- Lower LLM attack surface.
- A consistent, auditable boundary between ML numerics and language reasoning.
- The product can show "no LLM explanation available" without compromising the numeric signal.

## Alternatives considered
- Use the LLM as a numeric oracle: rejected (hallucination risk).
- Skip LLMs entirely: rejected (user requirement + value for explanation/synthesis).
- Use a self-hosted open model: deferred; cloud gives fast iteration.
