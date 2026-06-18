"""Direct ccxt diagnostic for Bybit testnet.kz keys.

This script tests connectivity, balance, and tiny orders via ccxt
against testnet.bybit.kz for multiple product types (linear perp, inverse
perp, spot) to isolate the exact permission/routing issue.
"""

import os
from pathlib import Path

import ccxt
from dotenv import load_dotenv


def _env() -> None:
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


def _try(exchange: ccxt.bybit, label: str, fn) -> None:
    print(f"\n--- {label} ---")
    try:
        result = fn()
        print("OK:", result)
    except Exception as e:
        print("FAILED:", type(e).__name__, e)


def main() -> None:
    _env()

    api_key = os.getenv("KAIRON_BYBIT_API_KEY", "")
    api_secret = os.getenv("KAIRON_BYBIT_API_SECRET", "")

    print(f"API key present: {bool(api_key)} (len={len(api_key)})")
    print(f"API secret present: {bool(api_secret)} (len={len(api_secret)})")

    # ---------- testnet.bybit.kz ----------
    exchange = ccxt.bybit({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "sandbox": True,
        "hostname": "bybit.kz",
    })
    exchange.load_markets()

    print("\n\n===== TESTNET.BYBIT.KZ =====")
    _try(
        exchange,
        "linear balance",
        lambda: exchange.fetch_balance(params={"type": "unified"}),
    )
    _try(
        exchange,
        "linear limit order",
        lambda: exchange.create_order(
            "BTC/USDT:USDT",
            "limit",
            "buy",
            0.001,
            1.0,
            params={"category": "linear"},
        ),
    )
    _try(
        exchange,
        "inverse limit order",
        lambda: exchange.create_order(
            "BTC/USD:BTC",
            "limit",
            "buy",
            1,
            1.0,
            params={"category": "inverse"},
        ),
    )
    _try(
        exchange,
        "spot balance",
        lambda: exchange.fetch_balance(params={"type": "spot"}),
    )
    _try(
        exchange,
        "spot limit order",
        lambda: exchange.create_order(
            "BTC/USDT",
            "limit",
            "buy",
            0.001,
            1.0,
            params={"category": "spot"},
        ),
    )

    # ---------- demo trading (paper on mainnet) ----------
    print("\n\n===== DEMO TRADING =====")
    demo = ccxt.bybit({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        # Demo trading is a separate Bybit environment; try the public hostname
        # first to see if the same key works there.
        "hostname": "bybit.com",
        "options": {"demoTrading": True},
    })
    demo.set_sandbox_mode(False)
    _try(demo, "demo server time", lambda: demo.publicGetV5MarketTime())
    _try(
        demo,
        "demo linear balance",
        lambda: demo.fetch_balance(params={"type": "unified"}),
    )


if __name__ == "__main__":
    main()
