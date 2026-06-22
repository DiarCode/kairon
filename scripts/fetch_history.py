"""Fetch Bybit TESTNET historical bars into the research parquet store.

Pulls N weeks (default 8) of 1m/5m/15m bars for the candidate scalping universe
into ``data/history/<symbol>/<tf>.parquet`` via
:mod:`kairon.data.history_fetch`. Idempotent + incremental: re-running only
appends bars that arrived since the last stored timestamp.

Usage:
    uv run python scripts/fetch_history.py
    uv run python scripts/fetch_history.py --symbols ETH-USDT-PERP SOL-USDT-PERP \\
        --timeframes 1m 5m 15m --weeks 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kairon.data.history_fetch import (
    DEFAULT_TIMEFRAMES,
    DEFAULT_WEEKS,
    sync_all_sync,
)
from kairon.data.symbols import CryptoVenue, crypto_perp

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYMBOLS = ["ETH-USDT-PERP", "SOL-USDT-PERP", "XRP-USDT-PERP",
                   "BTC-USDT-PERP", "LINK-USDT-PERP"]


def _load_env(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fetch_history", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES))
    p.add_argument("--weeks", type=int, default=DEFAULT_WEEKS)
    p.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    p.add_argument("--chunk-bars", type=int, default=1000)
    p.add_argument("--mainnet", action="store_true",
                   help="Fetch from MAINNET (default is testnet). NOT recommended: "
                        "the research store is testnet-labeled.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _load_env(REPO_ROOT)
    testnet = not args.mainnet
    sym_objs = []
    for s in args.symbols:
        base, quote = s.split("-")[0], s.split("-")[1]
        sym_objs.append(crypto_perp(base, quote, CryptoVenue.BYBIT))
    root = Path(args.data_root)
    print(f"Fetching {args.weeks}w of {args.timeframes} for {args.symbols}")
    print(f"  venue: {'TESTNET' if testnet else 'MAINNET'}  store: {root / 'history'}")
    report = sync_all_sync(
        sym_objs, args.timeframes, root=root, weeks=args.weeks,
        testnet=testnet, chunk_bars=args.chunk_bars,
    )
    print("\nHistory sync report:")
    for (sym, tf), n in sorted(report.items()):
        print(f"  {sym:<18} {tf:<4} {n:>8} bars")
    total = sum(report.values())
    print(f"  {'TOTAL':<23} {total:>8} bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
