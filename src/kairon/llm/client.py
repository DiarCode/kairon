"""LLM reasoning layer — Ollama cloud.

The LLM is the *narrator*, not the trader. Its job is to turn the
ensemble's structured output + market context into a human-readable
explanation. The LLM never sees the raw price series; it only sees:

- The current ensemble's probability distribution over classes.
- The walk-forward evaluation summary (accuracy, DSR, max DD).
- The recent feature summary (regime, volatility).
- A list of *allowed claim categories* (so we can post-filter
  hallucinated ones).

Everything the LLM produces is logged and reproducible — we never
make a trading decision based on the LLM's output alone. The LLM
output is treated as commentary.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairon.llm.schemas import (
    GroundedClaim,
    LLMReasoning,
    MarketContext,
    PredictionContext,
    ReasoningRequest,
    ReasoningResponse,
)


# ---------------------------------------------------------------------------
# Optional ollama dependency
# ---------------------------------------------------------------------------
def _has_ollama() -> bool:
    return importlib.util.find_spec("ollama") is not None  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Configuration for the Ollama client."""

    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "https://ollama.com"))
    model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud"))
    api_key: str = field(default_factory=lambda: os.environ.get("OLLAMA_API_KEY", ""))
    timeout_seconds: float = 60.0
    temperature: float = 0.2
    max_retries: int = 3
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class OllamaClient:
    """Thin wrapper over the ``ollama`` SDK.

    If ``ollama`` is not installed, all methods raise :class:`RuntimeError`
    with an actionable message. This keeps the LLM layer a hard opt-in.
    """

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self._ollama: Any = None
        if _has_ollama():
            self._ollama = importlib.import_module("ollama")

    @property
    def is_available(self) -> bool:
        return self._ollama is not None

    def reason(
        self,
        request: ReasoningRequest,
    ) -> ReasoningResponse:
        """Send a reasoning request to the LLM and return a structured
        response.

        Returns a :class:`ReasoningResponse` containing the raw LLM
        text, the list of *grounded* claims (after post-filtering), and
        any error information. Never raises — failures are returned in
        the response so the calling pipeline can degrade gracefully.
        """
        prompt = _build_prompt(request)
        if self._ollama is None:
            return ReasoningResponse(
                raw_text="",
                reasoning=LLMReasoning(
                    summary=(
                        "LLM layer unavailable (ollama SDK not installed). "
                        "No natural-language reasoning was generated."
                    ),
                    claims=(),
                    confidence=0.0,
                ),
                prompt_tokens=0,
                completion_tokens=0,
                error="ollama SDK not installed",
            )
        try:
            auth_headers: dict[str, str] = (
                {"Authorization": f"Bearer {self.config.api_key}"}
                if self.config.api_key
                else {}
            )
            client = self._ollama.Client(
                host=self.config.host,
                headers=auth_headers,
                timeout=self.config.timeout_seconds,
            )
            response = client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": self.config.temperature},
            )
            if isinstance(response, dict):
                text = response.get("message", {}).get("content", "")
            else:
                text = str(response)
        except Exception as e:
            return ReasoningResponse(
                raw_text="",
                reasoning=LLMReasoning(
                    summary=f"LLM call failed: {e}",
                    claims=(),
                    confidence=0.0,
                ),
                prompt_tokens=0,
                completion_tokens=0,
                error=str(e),
            )
        reasoning = _parse_response(text, request)
        return ReasoningResponse(
            raw_text=text,
            reasoning=reasoning,
            prompt_tokens=0,
            completion_tokens=0,
        )


# ---------------------------------------------------------------------------
# Prompt + parse
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a financial-narrative assistant. You will be \
given a structured prediction from an ML ensemble plus a market context. \
Your job is to:

1. Write a 2-3 sentence summary of what the model is *currently* saying.
2. List the *grounded* claims you can make, each one tagged with a \
category from the allowed set.

Rules:
- Never invent numbers that are not in the input. If a number is not \
present, omit that claim.
- Categorise each claim as one of: regime, signal, risk, sizing, or hedge.
- Do NOT make buy/sell recommendations. You are commentary only.
- Be concise. Avoid filler."""


def _build_prompt(req: ReasoningRequest) -> str:
    parts: list[str] = []
    parts.append("## Prediction\n")
    parts.append(_format_prediction(req.prediction))
    parts.append("\n## Market context\n")
    parts.append(_format_context(req.context))
    if req.allowed_claim_categories:
        cats = ", ".join(sorted(req.allowed_claim_categories))
        parts.append(f"\n## Allowed claim categories\n{cats}\n")
    if req.evaluation:
        parts.append("\n## Walk-forward evaluation\n")
        parts.append(_format_evaluation(req.evaluation))
    parts.append(
        "\nRespond in this JSON format:\n"
        "{\n"
        '  "summary": "string (2-3 sentences)",\n'
        '  "claims": [{"text": "...", "category": "regime|signal|risk|sizing|hedge"}],\n'
        '  "confidence": 0.0..1.0\n'
        "}\n"
    )
    return "".join(parts)


def _format_prediction(p: PredictionContext) -> str:
    bits: list[str] = []
    bits.append(f"- Backend: {p.backend}")
    bits.append(f"- Symbol: {p.symbol}")
    bits.append(f"- Horizon: {p.horizon}")
    bits.append(f"- Predicted class: {p.predicted_class}")
    if p.probabilities is not None:
        probs = ", ".join(f"P({c})={p:.3f}" for c, p in p.probabilities.items())
        bits.append(f"- Class probabilities: {probs}")
    if p.ensemble_agreement is not None:
        bits.append(f"- Ensemble agreement: {p.ensemble_agreement:.3f}")
    if p.notes:
        bits.append(f"- Notes: {p.notes}")
    return "\n".join(bits)


def _format_context(c: MarketContext) -> str:
    bits: list[str] = []
    if c.regime:
        bits.append(f"- Regime: {c.regime}")
    if c.volatility:
        bits.append(f"- Recent realised volatility: {c.volatility:.4f}")
    if c.drift:
        bits.append(f"- Recent drift: {c.drift:.4f}")
    if c.n_bars:
        bits.append(f"- Bars in window: {c.n_bars}")
    if c.extras:
        bits.append(f"- Other: {c.extras}")
    return "\n".join(bits) or "(no context)"


def _format_evaluation(ev: dict[str, Any]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in ev.items())


def _parse_response(text: str, request: ReasoningRequest) -> LLMReasoning:
    """Parse the LLM's JSON response into a :class:`LLMReasoning`."""
    # Best-effort JSON extraction (strip ```json fences if present)
    cleaned = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1)
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        # Treat the whole text as a summary, no claims
        return LLMReasoning(
            summary=cleaned or "(empty response)",
            claims=(),
            confidence=0.0,
        )
    summary = str(data.get("summary", ""))
    raw_claims = data.get("claims", [])
    allowed = set(request.allowed_claim_categories) if request.allowed_claim_categories else None
    claims: list[GroundedClaim] = []
    for c in raw_claims:
        if not isinstance(c, dict):
            continue
        text_claim = str(c.get("text", "")).strip()
        cat = str(c.get("category", "")).strip()
        if not text_claim:
            continue
        if allowed and cat not in allowed:
            continue
        claims.append(GroundedClaim(text=text_claim, category=cat))
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    return LLMReasoning(summary=summary, claims=tuple(claims), confidence=confidence)


# ---------------------------------------------------------------------------
# Convenience: high-level helper
# ---------------------------------------------------------------------------
def make_reasoning_request(
    *,
    symbol: str,
    horizon: str,
    backend: str,
    predicted_class: int,
    probabilities: dict[int, float] | None = None,
    regime: str | None = None,
    recent_volatility: float | None = None,
    ensemble_agreement: float | None = None,
    evaluation: dict[str, Any] | None = None,
) -> ReasoningRequest:
    """Convenience builder for the common case."""
    return ReasoningRequest(
        prediction=PredictionContext(
            symbol=symbol,
            horizon=horizon,
            backend=backend,
            predicted_class=predicted_class,
            probabilities=probabilities,
            ensemble_agreement=ensemble_agreement,
        ),
        context=MarketContext(
            regime=regime,
            volatility=recent_volatility,
        ),
        evaluation=evaluation,
        allowed_claim_categories=frozenset({"regime", "signal", "risk", "sizing", "hedge"}),
    )


def confidence_from_agreement(agreement: float) -> float:
    """Map an ensemble-agreement score (0..1) to a confidence number (0..1).

    Pure-Python helper, no LLM call. Useful for pre-filling the
    ``confidence`` field of :class:`LLMReasoning` when the LLM is
    unavailable but you still want a number.
    """
    if not np.isfinite(agreement):
        return 0.5
    # Soft mapping: 0.5 agreement → 0.5 confidence; 1.0 → ~0.85; 0.0 → ~0.15
    return float(0.5 + 0.7 * (agreement - 0.5))


__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "confidence_from_agreement",
    "make_reasoning_request",
]
