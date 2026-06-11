"""Tests for the LLM reasoning layer."""

from __future__ import annotations

import importlib  # noqa: F401 — used by name inside a string
from typing import Any

import pytest

from kairon.llm.client import (
    OllamaClient,
    OllamaConfig,
    _has_ollama,
    confidence_from_agreement,
    make_reasoning_request,
)
from kairon.llm.schemas import (
    LLMReasoning,
    ReasoningRequest,
    ReasoningResponse,
)


# --- pure helpers ---------------------------------------------------------
def test_confidence_from_agreement_center() -> None:
    assert confidence_from_agreement(0.5) == pytest.approx(0.5)


def test_confidence_from_agreement_high() -> None:
    assert confidence_from_agreement(1.0) == pytest.approx(0.85)


def test_confidence_from_agreement_low() -> None:
    assert confidence_from_agreement(0.0) == pytest.approx(0.15)


def test_confidence_from_agreement_nan() -> None:
    assert confidence_from_agreement(float("nan")) == 0.5


def test_make_reasoning_request_defaults() -> None:
    req = make_reasoning_request(
        symbol="BTC/USDT",
        horizon="1h",
        backend="logreg",
        predicted_class=1,
        probabilities={0: 0.3, 1: 0.7},
        regime="trending",
    )
    assert isinstance(req, ReasoningRequest)
    assert req.prediction.symbol == "BTC/USDT"
    assert req.context.regime == "trending"
    assert "regime" in req.allowed_claim_categories


# --- config validation ----------------------------------------------------
def test_ollama_config_validates() -> None:
    with pytest.raises(ValueError):
        OllamaConfig(timeout_seconds=0)
    with pytest.raises(ValueError):
        OllamaConfig(temperature=-0.1)
    with pytest.raises(ValueError):
        OllamaConfig(temperature=3.0)
    with pytest.raises(ValueError):
        OllamaConfig(max_retries=-1)


# --- client behaviour when ollama is absent --------------------------------
def test_client_works_without_ollama(monkeypatch) -> None:
    """When the SDK isn't installed, the client returns a graceful
    'unavailable' response instead of raising.
    """
    if _has_ollama():
        pytest.skip("ollama installed; can't test no-op path")
    client = OllamaClient()
    assert client.is_available is False
    req = make_reasoning_request(
        symbol="BTC/USDT", horizon="1h", backend="logreg", predicted_class=1
    )
    resp = client.reason(req)
    assert isinstance(resp, ReasoningResponse)
    assert resp.reasoning.confidence == 0.0
    assert "unavailable" in resp.reasoning.summary.lower()
    assert resp.error is not None


# --- response parsing ------------------------------------------------------
def test_parse_response_json_block(monkeypatch) -> None:
    """End-to-end: parse a synthetic LLM response via a stubbed SDK."""
    from kairon.llm import client as client_mod

    fake_module = type("mod", (), {})
    fake_module.Client = lambda **kw: _FakeClient(  # type: ignore[attr-type]
        text=(
            "```json\n"
            "{\n"
            '  "summary": "The model is bullish on the 1h horizon.",\n'
            '  "claims": [\n'
            '    {"text": "Regime is trending", "category": "regime"},\n'
            '    {"text": "Signal agreement is 0.85", "category": "signal"},\n'
            '    {"text": "Surprise claim", "category": "surprise"}\n'
            "  ],\n"
            '  "confidence": 0.78\n'
            "}\n"
            "```"
        )
    )
    monkeypatch.setattr(client_mod, "_has_ollama", lambda: True)
    monkeypatch.setattr(client_mod, "importlib", _StubImportlib(fake_module))
    # Use the parser directly to avoid monkeypatching the module-level import
    req = make_reasoning_request(
        symbol="BTC/USDT", horizon="1h", backend="logreg", predicted_class=1
    )
    reasoning = client_mod._parse_response(
        fake_module.Client().text, req  # type: ignore[attr-defined]
    )
    assert isinstance(reasoning, LLMReasoning)
    assert "bullish" in reasoning.summary
    assert reasoning.confidence == 0.78
    # 'surprise' is not in the allowed set, so it must be filtered
    cats = [c.category for c in reasoning.claims]
    assert "regime" in cats
    assert "signal" in cats
    assert "surprise" not in cats


def test_parse_response_handles_garbage() -> None:
    from kairon.llm import client as client_mod

    req = make_reasoning_request(
        symbol="BTC/USDT", horizon="1h", backend="logreg", predicted_class=1
    )
    reasoning = client_mod._parse_response("not json at all", req)
    assert reasoning.summary == "not json at all"
    assert reasoning.claims == ()


def test_prompt_includes_allowed_categories() -> None:
    from kairon.llm import client as client_mod

    req = make_reasoning_request(
        symbol="BTC/USDT", horizon="1h", backend="logreg", predicted_class=1
    )
    prompt = client_mod._build_prompt(req)
    assert "regime" in prompt
    assert "Allowed claim categories" in prompt
    assert "BTC/USDT" in prompt
    assert "logreg" in prompt


# --- test stubs ------------------------------------------------------------
class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubImportlib:
    def __init__(self, mod: Any) -> None:
        self._mod = mod

    def import_module(self, name: str) -> Any:
        return self._mod


# --- smoke test for module imports ----------------------------------------
def test_llm_public_api_imports() -> None:
    from kairon.llm import (
        OllamaClient,
        OllamaConfig,
    )

    assert OllamaClient is not None
    assert OllamaConfig is not None
