"""Run a 30-minute Bybit testnet live trading session for BTC and ETH.

This orchestrator launches two independent per-symbol workers (each in its
own process with its own SQLite store and log file), waits for them to
finish, then combines their reports into a single markdown summary.

All order, fill, decision, heartbeat, and closed-trade data are persisted to
SQLite under ``data/`` and can be analyzed with ``kairon.live.analytics`` or
used to retrain / upgrade downstream models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# pybit logs contain unicode arrows; force UTF-8 on Windows subprocesses.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

DEFAULT_SYMBOLS = ["BTC-USDT-PERP", "ETH-USDT-PERP"]
DEFAULT_DURATION_SECONDS = 30 * 60


def _safe_num(v) -> float:
    if v is None or isinstance(v, float) and math.isnan(v):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


async def _run_one(
    interpreter: Path,
    script: Path,
    symbol: str,
    ts: str,
    data_dir: Path,
    logs_dir: Path,
    duration_seconds: int,
    env: dict[str, str],
) -> asyncio.subprocess.Process:
    safe = symbol.replace("-", "_")
    db = data_dir / f"runs_testnet_30min_{safe}_{ts}.db"
    log = logs_dir / f"testnet_30min_{safe}_{ts}.log"
    cmd = [
        str(interpreter),
        str(script),
        symbol,
        "--db",
        str(db),
        "--log",
        str(log),
        "--duration",
        str(duration_seconds),
    ]
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )


def _collect_report(logs_dir: Path, symbol: str, ts: str) -> dict | None:
    safe = symbol.replace("-", "_")
    path = logs_dir / f"testnet_30min_{safe}_{ts}.report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _recompute_report(data_dir: Path, symbol: str, ts: str):
    """Recompute a LiveSessionReport directly from the worker's SQLite DB.

    The worker serializes its report with ``json.dumps(..., default=str)``,
    which stringifies the ``LiveSessionReport`` dataclass and makes it
    unusable for ``format_report``. Recomputing from the DB yields a real
    ``LiveSessionReport`` with correct attribute access (n_orders, n_fills,
    n_trades, per-symbol breakdown, etc.).
    """
    from kairon.live.analytics import compute_session_report
    from kairon.live.store import LiveStore

    safe = symbol.replace("-", "_")
    db = data_dir / f"runs_testnet_30min_{safe}_{ts}.db"
    if not db.exists():
        return None
    store = LiveStore(db)
    try:
        return compute_session_report(store, timeframe="1m", session_id=f"testnet-{symbol}")
    finally:
        store.close()


def _format_money(v) -> str:
    return f"${_safe_num(v):,.2f}"


def _safe_print(text: str) -> None:
    """Print to stdout, replacing characters the console cannot encode."""
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_testnet_30min_btc_eth",
        description="Run a 30-minute Bybit testnet live trading session for BTC and ETH.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to trade (default: {' '.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help="Session duration in seconds (default: 1800 = 30 minutes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned configuration and exit without launching workers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    data_dir = repo_root / "data"
    logs_dir = repo_root / "logs"
    reports_dir = repo_root / "reports"
    for d in (data_dir, logs_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Make sure workers can import kairon even when spawned from a bare interpreter.
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if str(src_dir) not in existing_pythonpath:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pythonpath}".rstrip(os.pathsep)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    interpreter = Path(sys.executable)
    worker = repo_root / "scripts" / "run_testnet_symbol.py"
    symbols = args.symbols
    duration_seconds = args.duration

    _safe_print(f"Starting {duration_seconds // 60}-minute Bybit testnet session at {ts}")
    _safe_print(f"Symbols: {symbols}")
    _safe_print(f"Interpreter: {interpreter}")
    _safe_print(f"PYTHONPATH: {env.get('PYTHONPATH')}")
    _safe_print(f"Logs: {logs_dir}")
    _safe_print(f"Databases: {data_dir}")

    if args.dry_run:
        _safe_print("Dry run: no workers launched.")
        return 0

    async def _orchestrate() -> list[asyncio.subprocess.Process]:
        procs = await asyncio.gather(
            *(
                _run_one(interpreter, worker, sym, ts, data_dir, logs_dir, duration_seconds, env)
                for sym in symbols
            )
        )

        async def _stream(
            proc: asyncio.subprocess.Process, symbol: str
        ) -> asyncio.subprocess.Process:
            prefix = f"[{symbol}]"
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                _safe_print(f"{prefix} {decoded}")
            return proc

        streamed = await asyncio.gather(*(_stream(p, sym) for p, sym in zip(procs, symbols)))
        await asyncio.gather(*(p.wait() for p in streamed))
        return streamed

    procs = asyncio.run(_orchestrate())
    codes = [p.returncode for p in procs]

    reports: list[dict] = []
    for sym in symbols:
        rep = _collect_report(logs_dir, sym, ts)
        if rep is not None:
            reports.append(rep)
        else:
            print(f"WARNING: report missing for {sym}; check the per-symbol log.")

    if not reports:
        print("ERROR: no reports were produced. Check the per-symbol logs.")
        return 1

    lines = [
        "# Kairon 30-Minute Bybit Testnet Session Report — BTC + ETH",
        "",
        f"- Session ID: `testnet_30min_btc_eth_{ts}`",
        f"- Duration: {duration_seconds // 60} minutes",
        f"- Symbols: {', '.join(symbols)}",
        f"- Worker return codes: {dict(zip(symbols, codes))}",
        "",
        "## Combined PnL (Bybit account balance)",
        "",
        "| Symbol | Initial | Final | PnL | PnL % | Orders | Fills | Trades |",
        "|---|---|---|---|---|---|---|---|",
    ]
    total_initial = 0.0
    total_final = 0.0
    total_pnl = 0.0
    total_orders = 0
    total_fills = 0
    total_trades = 0
    for rep in reports:
        sym = rep["symbol"]
        initial = _safe_num(rep["initial_balance"])
        final = _safe_num(rep["final_balance"])
        pnl = _safe_num(rep["total_pnl"])
        pct = _safe_num(rep["total_pnl_pct"])
        rpt = _recompute_report(data_dir, sym, ts)
        if rpt is None:
            print(f"WARNING: could not recompute report from DB for {sym}; "
                  f"using the worker's scalar summary only.")
        orders = int(getattr(rpt, "n_orders", 0)) if rpt is not None else 0
        fills = int(getattr(rpt, "n_fills", 0)) if rpt is not None else 0
        trades = int(getattr(rpt, "n_trades", 0)) if rpt is not None else 0
        total_initial += initial
        total_final += final
        total_pnl += pnl
        total_orders += orders
        total_fills += fills
        total_trades += trades
        lines.append(
            f"| {sym} | {_format_money(initial)} | {_format_money(final)} | "
            f"{_format_money(pnl)} ({pct:+.2f}%) | {orders} | {fills} | {trades} |"
        )
    lines.append(
        f"| **TOTAL** | {_format_money(total_initial)} | {_format_money(total_final)} | "
        f"{_format_money(total_pnl)} | {total_orders} | {total_fills} | {total_trades} |"
    )
    lines.append("")

    lines.append("## Bybit closed PnL (per symbol)")
    lines.append("")
    lines.append("| Symbol | Closed entries | Realized PnL |")
    lines.append("|---|---|---|")
    for rep in reports:
        sym = rep["symbol"]
        count = rep.get("closed_pnl_count", 0)
        realized = _safe_num(rep.get("closed_pnl_realized", 0))
        lines.append(f"| {sym} | {count} | {_format_money(realized)} |")
    lines.append("")

    lines.append("## Per-symbol detail")
    lines.append("")
    from kairon.live.analytics import format_report
    for rep in reports:
        sym = rep["symbol"]
        rpt = _recompute_report(data_dir, sym, ts)
        lines.append(f"### {sym}")
        lines.append("")
        lines.append("```")
        lines.append(format_report(rpt) if rpt is not None else "(report unavailable)")
        lines.append("```")
        lines.append("")

    report_path = reports_dir / f"testnet_30min_btc_eth_{ts}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCombined report written to: {report_path}")
    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)
    print(f"Initial balance: {_format_money(total_initial)}")
    print(f"Final balance:   {_format_money(total_final)}")
    print(f"Total PnL:       {_format_money(total_pnl)}")
    print(f"Total orders:    {total_orders}")
    print(f"Total fills:     {total_fills}")
    print(f"Total trades:    {total_trades}")
    print("=" * 60)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
