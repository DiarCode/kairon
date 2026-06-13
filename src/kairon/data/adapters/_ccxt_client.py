"""Lazy ccxt client factory.

Kept in its own module so that the heavy ``ccxt`` import is only
paid when an actual exchange client is constructed (and so it can
be patched from tests without touching the rest of the adapter).

Notes
-----
On Windows, ``aiodns`` may fail to resolve DNS (``pycares`` can't
contact DNS servers). We work around this by creating a custom
``aiohttp.TCPConnector`` with ``ThreadedResolver``, which delegates
to the OS resolver — the same one that ``curl`` and ``requests``
use successfully.
"""

from __future__ import annotations

import importlib
from typing import Any


def _make_threaded_connector() -> Any:
    """Create an aiohttp TCPConnector that uses the OS DNS resolver.

    Falls back to a default connector if ``aiohttp`` is not available
    (shouldn't happen since ccxt depends on it).
    """
    try:
        import aiohttp

        return aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
    except ImportError:
        return None


def make_client(
    venue: str,
    *,
    testnet: bool = False,
    api_key: str = "",
    api_secret: str = "",
) -> Any:  # noqa: ANN401 - ccxt is not well-typed
    """Build an async ccxt client with rate limiting enabled.

    The return type is ``Any`` because ccxt's Python types are
    not well-typed in pyright strict mode; the public adapter
    API is typed.

    Parameters
    ----------
    venue:
        ccxt exchange class name (e.g. ``"bybit"``, ``"binance"``).
    testnet:
        If True, enable sandbox/testnet mode (routes requests to
        the exchange's testnet URLs).
    api_key:
        API key for authenticated endpoints.
    api_secret:
        API secret for authenticated endpoints.
    """
    ccxt_async = importlib.import_module("ccxt.async_support")
    klass = getattr(ccxt_async, venue)
    options: dict[str, Any] = {"enableRateLimit": True}
    if testnet:
        options["sandbox"] = True
    if api_key:
        options["apiKey"] = api_key
    if api_secret:
        options["secret"] = api_secret
    exchange = klass(options)

    # Fix DNS resolution on Windows: replace the default aiohttp
    # session with one that uses ThreadedResolver (OS DNS resolver)
    # instead of aiodns (which can fail on some Windows setups).
    connector = _make_threaded_connector()
    if connector is not None:
        import aiohttp

        exchange.session = aiohttp.ClientSession(connector=connector)

    return exchange