"""Close a residual open position on the Bybit testnet account.

Uses BybitBroker.close_position() (chunked reduce-only market + limit fallback),
which is the exact path built for thin testnet books. Safe: reduce-only orders
cannot increase exposure.

Usage:
    uv run python scripts/close_residual_position.py ETH-USDT-PERP
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _load_env(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


async def _run(symbol: str) -> int:
    from kairon.live.broker.bybit import BybitBroker

    broker = BybitBroker(
        api_key=os.environ["KAIRON_BYBIT_API_KEY"],
        api_secret=os.environ["KAIRON_BYBIT_API_SECRET"],
        testnet=os.environ.get("KAIRON_BYBIT_TESTNET", "true").lower()
        in ("1", "true", "yes"),
        tld=os.environ.get("KAIRON_BYBIT_TLD", "com"),
    )

    positions = await broker.get_positions(symbol)
    pos = next((p for p in positions if p.symbol == symbol and p.qty > 1e-9), None)
    if pos is None:
        print(f"No open position for {symbol}; nothing to close.")
        return 0

    print(f"Open position: {symbol} side={pos.side} qty={pos.qty} avg_entry={pos.avg_entry}")
    print("Closing via chunked reduce-only + limit fallback ...")

    order = await broker.close_position(symbol)
    print(f"Close order submitted: id={order.id} side={order.side} "
          f"qty={order.qty} type={order.order_type} status={order.status}")

    # Verify it actually closed.
    await asyncio.sleep(2.0)
    positions2 = await broker.get_positions(symbol)
    pos2 = next((p for p in positions2 if p.symbol == symbol), None)
    remaining = pos2.qty if pos2 else 0.0
    if remaining <= 1e-9:
        print(f"OK: {symbol} position fully closed (remaining={remaining}).")
        return 0
    print(f"WARN: residual {remaining} {symbol} still open. Retry or close via UI.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="close_residual_position")
    parser.add_argument("symbol", nargs="?", default="ETH-USDT-PERP")
    args = parser.parse_args(argv)
    _load_env(Path(__file__).resolve().parent.parent)
    if not os.environ.get("KAIRON_BYBIT_API_KEY"):
        print("FAIL: KAIRON_BYBIT_API_KEY missing from .env")
        return 2
    return asyncio.run(_run(args.symbol))


if __name__ == "__main__":
    raise SystemExit(main())