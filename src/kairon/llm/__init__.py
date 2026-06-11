"""LLM reasoning layer — Ollama cloud + a no-op fallback."""

from kairon.llm.client import (
    OllamaClient,
    OllamaConfig,
    confidence_from_agreement,
    make_reasoning_request,
)
from kairon.llm.schemas import (
    GroundedClaim,
    LLMReasoning,
    MarketContext,
    PredictionContext,
    ReasoningRequest,
    ReasoningResponse,
)

__all__ = [
    "GroundedClaim",
    "LLMReasoning",
    "MarketContext",
    "OllamaClient",
    "OllamaConfig",
    "PredictionContext",
    "ReasoningRequest",
    "ReasoningResponse",
    "confidence_from_agreement",
    "make_reasoning_request",
]
