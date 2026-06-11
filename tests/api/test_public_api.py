"""Public API smoke test for the api subpackage."""
from __future__ import annotations


def test_api_public_api_imports() -> None:
    from kairon.api import (
        HealthResponse,
    )

    assert HealthResponse(status="ok", version="0.1.0", uptime_seconds=1.0)
