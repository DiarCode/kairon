"""Bybit testnet preflight: verify the account is healthy enough to trade.

Checks (no secrets printed):
  1. Connection + server time.
  2. USDT equity and availableToWithdraw > 0.
  3. No residual open positions.
  4. Trading permission: attempt a tiny cancel-able limit order and cancel it,
     reporting whether ErrCode 10024 / 110007 / other is hit.

Usage:
    uv run python scripts/preflight_testnet.py [--symbol BTC-USDT-PERP]

Reads KAIRON_BYBIT_* vars from .env (or the real environment).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# pybit error strings contain unicode arrows; force UTF-8 on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass


def _load_env(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _masked(key: str) -> str:
    v = os.environ.get(key, "")
    if not v:
        return "<unset>"
    if len(v) <= 6:
        return f"{v[:1]}***"
    return f"{v[:3]}...{v[-2:]} (len={len(v)})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preflight_testnet")
    parser.add_argument("--symbol", default="BTC-USDT-PERP")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root)

    api_key = os.environ.get("KAIRON_BYBIT_API_KEY", "")
    api_secret = os.environ.get("KAIRON_BYBIT_API_SECRET", "")
    testnet = os.environ.get("KAIRON_BYBIT_TESTNET", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    tld = os.environ.get("KAIRON_BYBIT_TLD", "com")

    print(f"testnet={testnet} tld={tld} key={_masked('KAIRON_BYBIT_API_KEY')} "
          f"secret={_masked('KAIRON_BYBIT_API_SECRET')}")
    if not api_key or not api_secret:
        print("FAIL: missing KAIRON_BYBIT_API_KEY / KAIRON_BYBIT_API_SECRET in .env")
        return 2

    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        print("FAIL: pybit not installed. Run `uv sync --extra live`.")
        return 2

    session = HTTP(
        testnet=testnet,
        api_key=api_key,
        api_secret=api_secret,
        tld=tld,
        log_requests=False,
    )

    failures: list[str] = []

    # 1. server time
    try:
        t = session.get_server_time()
        print(f"[ok] server time: {t}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] get_server_time: {e}")
        return 1

    # 2. wallet balance
    try:
        w = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows = (w.get("result") or {}).get("list", [])
        if not rows:
            print("[FAIL] wallet balance returned no accounts")
            return 1
        acct = rows[0]
        eq = _coin_field(acct, "totalEquity")
        # availableToWithdraw is often None/blank at top level; the real
        # "can I place orders" signal is totalAvailableBalance (margin).
        avail = _coin_field(acct, "totalAvailableBalance")
        if avail <= 0:
            avail = _coin_field(acct, "availableToWithdraw")
        print(f"[info] USDT totalEquity={eq:.4f} availableBalance={avail:.4f}")
        if eq <= 0:
            failures.append("totalEquity is 0 — account has no funds")
        if avail <= 0:
            failures.append("available balance is 0 — new orders will be rejected (110007)")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] get_wallet_balance: {e}")
        failures.append("wallet balance call failed")

    # 3. positions
    try:
        p = session.get_positions(category="linear", settleCoin="USDT")
        pos_list = (p.get("result") or {}).get("list", [])
        open_pos = [x for x in pos_list if _to_float(x.get("size")) != 0.0]
        print(f"[info] open linear positions: {len(open_pos)}")
        for x in open_pos:
            print(f"   - {x.get('symbol')} size={x.get('size')} side={x.get('side')} "
                  f"unrealisedPnl={x.get('unrealisedPnl')}")
        if open_pos:
            failures.append(f"{len(open_pos)} residual open position(s) still open")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] get_positions: {e}")
        failures.append("positions call failed")

    # 4. permission probe: place + cancel a tiny limit far from market
    try:
        from kairon.live.broker.bybit_shared import symbol_str_to_bybit
        bybit_sym, _category = symbol_str_to_bybit(args.symbol)
    except Exception:  # noqa: BLE001
        bybit_sym = args.symbol.replace("-USDT-PERP", "USDT").replace("-", "")
    try:
        tick = session.get_instruments_info(category="linear", symbol=bybit_sym)
        lists = (tick.get("result") or {}).get("list", [])
        if not lists:
            print(f"[FAIL] instruments_info empty for {bybit_sym}")
            failures.append(f"cannot resolve instrument {bybit_sym}")
        else:
            inst = lists[0]
            min_qty = _to_float(inst.get("lotSizeFilter", {}).get("minOrderQty", "0.001"))
            mark = _to_float(session.get_tickers(category="linear", symbol=bybit_sym)
                             .get("result", {}).get("list", [{}])[0].get("markPrice", "0"))
            # limit price far below mark so it never fills (long side)
            limit_px = round(mark * 0.5, 6) or 1.0
            print(f"[info] {bybit_sym} mark={mark:.4f} minQty={min_qty} "
                  f"probe limit px={limit_px}")
            try:
                r = session.place_order(
                    category="linear",
                    symbol=bybit_sym,
                    side="Buy",
                    orderType="Limit",
                    qty=str(max(min_qty, 0.001)),
                    price=str(limit_px),
                    timeInForce="PostOnly",
                )
                oid = (r.get("result") or {}).get("orderId")
                print(f"[ok] probe order accepted orderId={oid}")
                if oid:
                    try:
                        session.cancel_order(category="linear", symbol=bybit_sym, orderId=oid)
                        print("[ok] probe order cancelled")
                    except Exception as ce:  # noqa: BLE001
                        print(f"[warn] cancel failed: {ce}")
            except Exception as pe:  # noqa: BLE001
                msg = str(pe)
                code = _extract_code(msg)
                if code == "10024":
                    failures.append("10024: account restricted from linear perps")
                elif code == "110007":
                    failures.append("110007: insufficient available balance to place order")
                else:
                    print(f"[FAIL] place_order: {msg}")
                    failures.append(f"place_order failed (code={code})")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] permission probe setup: {e}")
        failures.append("permission probe setup failed")

    print("\n=== PREFLIGHT RESULT ===")
    if failures:
        for f in failures:
            print(f"  - {f}")
        print(f"\n{len(failures)} issue(s); NOT ready to trade.")
        return 1
    print("  all checks passed — account is ready to trade.")
    return 0


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _coin_field(acct: dict, field: str) -> float:
    return _to_float(acct.get(field))


def _extract_code(msg: str) -> str:
    # pybit error strings look like: ... ErrCode: 10024 ...
    for tok in msg.replace(",", " ").split():
        if tok.isdigit():
            return tok
    return "?"


if __name__ == "__main__":
    raise SystemExit(main())