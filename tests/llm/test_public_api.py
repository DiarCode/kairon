"""Public API smoke test for the llm subpackage."""
from __future__ import annotations


def test_llm_public_api_imports() -> None:
    from kairon.llm import (
        OllamaClient,
        OllamaConfig,
    )

    assert OllamaClient is not None
    assert OllamaConfig is not None
