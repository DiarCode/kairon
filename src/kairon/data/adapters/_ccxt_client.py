"""Lazy ccxt client factory.

Kept in its own module so that the heavy ``ccxt`` import is only
paid when an actual exchange client is constructed (and so it can
be patched from tests without touching the rest of the adapter).
"""

from __future__ import annotations

import importlib
from typing import Any


def make_client(venue: str) -> Any:  # noqa: ANN401 - ccxt is not well-typed
    """Build an async ccxt client with rate limiting enabled.

    The return type is ``Any`` because ccxt's Python types are
    not well-typed in pyright strict mode; the public adapter
    API is typed.
    """
    ccxt_async = importlib.import_module("ccxt.async_support")
    klass = getattr(ccxt_async, venue)
    return klass({"enableRateLimit": True})
