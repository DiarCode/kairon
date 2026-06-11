"""LLM-layer typed schemas.

The LLM is treated as an *untrusted narrator*. We give it a structured
input (prediction + market context) and a list of allowed claim
categories. The response is parsed, post-filtered against the
allow-list, and returned in a strictly-typed envelope so the rest of
the codebase can never accidentally pass an ungrounded string into a
decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ClaimCategory = str  # "regime" | "signal" | "risk" | "sizing" | "hedge"


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Structured input describing the ensemble's current prediction."""

    symbol: str
    horizon: str
    backend: str
    predicted_class: int
    probabilities: dict[int, float] | None = None
    ensemble_agreement: float | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Structured market context for the LLM."""

    regime: str | None = None
    volatility: float | None = None
    drift: float | None = None
    n_bars: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReasoningRequest:
    """A single LLM reasoning call."""

    prediction: PredictionContext
    context: MarketContext
    evaluation: dict[str, Any] | None = None
    allowed_claim_categories: frozenset[str] = field(
        default_factory=lambda: frozenset({"regime", "signal", "risk", "sizing", "hedge"})
    )


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    """A single claim, post-filtered against the allow-list."""

    text: str
    category: str


@dataclass(frozen=True, slots=True)
class LLMReasoning:
    """The structured LLM output."""

    summary: str
    claims: tuple[GroundedClaim, ...]
    confidence: float  # 0..1


@dataclass(frozen=True, slots=True)
class ReasoningResponse:
    """Envelope around the LLM call: raw text + parsed reasoning + meta."""

    raw_text: str
    reasoning: LLMReasoning
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


__all__ = [
    "GroundedClaim",
    "LLMReasoning",
    "MarketContext",
    "PredictionContext",
    "ReasoningRequest",
    "ReasoningResponse",
]
