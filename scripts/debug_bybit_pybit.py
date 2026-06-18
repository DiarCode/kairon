"""Direct pybit diagnostic for the active Bybit account.

Loads credentials from .env, then tests the pybit HTTP path end-to-end:
server time, wallet balance, and a non-executable limit order probe.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _env() -> None:
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


def _safe_print(label: str, value: object) -> None:
    """Print without triggering Windows cp1252 encoding errors."""
    text = f"{label}: {value}"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "ignore").decode("ascii"))


def _try(label: str, fn: Callable[[], Any]) -> None:
    """Run ``fn`` and print the outcome."""
    print(f"\n--- {label} ---")
    try:
        result = fn()
        _safe_print("OK", result)
    except Exception as e:
        msg = str(e).encode("ascii", "ignore").decode("ascii")
        print(f"FAILED: {type(e).__name__} {msg}")


def main() -> None:
    _env()

    api_key = os.getenv("KAIRON_BYBIT_API_KEY", "")
    api_secret = os.getenv("KAIRON_BYBIT_API_SECRET", "")
    tld = os.getenv("KAIRON_BYBIT_TLD", "com")

    print(f"tld={tld}")

    from pybit.unified_trading import HTTP  # noqa: PLC0415

    http = HTTP(
        testnet=True,
        api_key=api_key,
        api_secret=api_secret,
        tld=tld,
        log_requests=False,
    )

    _try("server time", http.get_server_time)
    _try("wallet balance", lambda: http.get_wallet_balance(accountType="UNIFIED"))
    _try("linear limit order", lambda: http.place_order(
        category="linear",
        symbol="BTCUSDT",
        side="Buy",
        orderType="Limit",
        qty="0.001",
        price="1.0",
    ))
    _try("inverse limit order", lambda: http.place_order(
        category="inverse",
        symbol="BTCUSD",
        side="Buy",
        orderType="Limit",
        qty="1",
        price="1.0",
    ))
    _try("spot limit order", lambda: http.place_order(
        category="spot",
        symbol="BTCUSDT",
        side="Buy",
        orderType="Limit",
        qty="0.001",
        price="1.0",
    ))


if __name__ == "__main__":
    main()
