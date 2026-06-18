"""Diagnostic script for BybitRawBroker against testnet.bybit.kz.

Loads credentials from .env, prints server time and wallet balance, then
attempts a tiny non-executable limit order. The full request/response is
printed for transparency.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from kairon.live.broker.bybit_raw import BybitRawBroker
from kairon.live.broker.bybit_shared import BybitAPIError, BybitPermissionError


def _load_env() -> None:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


def _safe_print(label: str, value: object) -> None:
    """Print without triggering Windows cp1252 encoding errors."""
    text = f"{label}: {value}"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "ignore").decode("ascii"))


async def main() -> None:  # noqa: PLR0915
    _load_env()

    api_key = os.getenv("KAIRON_BYBIT_API_KEY", "")
    api_secret = os.getenv("KAIRON_BYBIT_API_SECRET", "")
    tld = os.getenv("KAIRON_BYBIT_TLD", "kz")

    if not api_key or not api_secret:
        _safe_print("ERROR", "KAIRON_BYBIT_API_KEY/SECRET not set")
        return

    broker = BybitRawBroker(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True,
        tld=tld,
    )

    _safe_print("Broker", f"BybitRawBroker(testnet=True, tld={tld})")

    try:
        server_time = await broker._request("GET", "/v5/market/time", signed=False)
        _safe_print("Server time", server_time.get("result", {}))
    except Exception as e:
        _safe_print("Server time failed", e)

    try:
        balances = await broker.get_balances()
        _safe_print("Balances", [b.model_dump() for b in balances])
    except Exception as e:
        _safe_print("Balances failed", e)

    # Pick a price slightly below the current best bid so the probe order is
    # valid but very unlikely to fill.
    probe_price = "1.0"
    try:
        ticker = await broker._request(
            "GET",
            "/v5/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            signed=False,
        )
        tick = ticker.get("result", {}).get("list", [{}])[0]
        bid = float(tick.get("bid1Price", "0") or "0")
        if bid > 0:
            probe_price = f"{bid * 0.995:.2f}"
    except Exception as e:
        _safe_print("Could not fetch ticker, using fallback price", e)

    order_link_id = f"debug-{int(os.times().system * 1000)}"
    _safe_print("OrderLinkId", order_link_id)
    _safe_print("Probe price", probe_price)

    try:
        response = await broker._request(
            "POST",
            "/v5/order/create",
            payload={
                "category": "linear",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "qty": "0.001",
                "price": probe_price,
                "orderLinkId": order_link_id,
            },
        )
        _safe_print("Order accepted", response)
        # Cancel immediately so we do not leave an orphan order.
        try:
            cancel = await broker._request(
                "POST",
                "/v5/order/cancel",
                payload={
                    "category": "linear",
                    "symbol": "BTCUSDT",
                    "orderLinkId": order_link_id,
                },
            )
            _safe_print("Cancelled", cancel)
        except Exception as e:
            _safe_print("Cancel failed", e)
    except BybitPermissionError as e:
        _safe_print("RESTRICTED", e)
        _safe_print(
            "Note",
            "Account blocked by Bybit error 10024. Complete KYC/eligibility "
            "prompt in Bybit app or on testnet.bybit.com.",
        )
    except BybitAPIError as e:
        msg = str(e)
        if "170131" in msg or "Insufficient balance" in msg or "110007" in msg:
            _safe_print(
                "Permission OK",
                "Account can trade, but has insufficient balance/margin",
            )
        else:
            _safe_print("API error", e)
    except Exception as e:
        _safe_print("Order failed", e)

    await broker.aclose()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
