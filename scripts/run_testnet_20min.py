"""Run a 20-minute Bybit testnet live trading session for BTC, ETH, and XRP.

This orchestrator launches three independent per-symbol workers (each in
its own process with its own SQLite store and log file), waits for them
to finish, then combines their reports into a single markdown summary.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# pybit logs contain unicode arrows; force UTF-8 on Windows subprocesses.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SYMBOLS = ["BTC-USDT-PERP", "ETH-USDT-PERP", "XRP-USDT-PERP"]
DURATION_SECONDS = 20 * 60


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
) -> asyncio.subprocess.Process:
    safe = symbol.replace("-", "_")
    db = data_dir / f"runs_testnet_{safe}_{ts}.db"
    log = logs_dir / f"testnet_20min_{safe}_{ts}.log"
    cmd = [
        str(interpreter),
        str(script),
        symbol,
        "--db",
        str(db),
        "--log",
        str(log),
        "--duration",
        str(DURATION_SECONDS),
    ]
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


async def _collect_report(logs_dir: Path, symbol: str, ts: str) -> dict | None:
    safe = symbol.replace("-", "_")
    path = logs_dir / f"testnet_20min_{safe}_{ts}.report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    logs_dir = repo_root / "logs"
    reports_dir = repo_root / "reports"
    for d in (data_dir, logs_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    interpreter = Path(sys.executable)
    worker = repo_root / "scripts" / "run_testnet_symbol.py"

    _safe_print(f"Starting 20-minute Bybit testnet session at {ts}")
    _safe_print(f"Symbols: {SYMBOLS}")
    _safe_print(f"Interpreter: {interpreter}")
    _safe_print(f"Logs: {logs_dir}")
    _safe_print(f"Databases: {data_dir}")

    async def _orchestrate() -> list[dict]:
        # Start all three workers in parallel.
        procs = await asyncio.gather(
            *(_run_one(interpreter, worker, sym, ts, data_dir, logs_dir) for sym in SYMBOLS)
        )
        # Stream and combine output so the user sees it live.
        async def _stream(proc: asyncio.subprocess.Process, symbol: str) -> asyncio.subprocess.Process:
            prefix = f"[{symbol}]"
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                _safe_print(f"{prefix} {decoded}")
            return proc

        streamed = await asyncio.gather(*(_stream(p, sym) for p, sym in zip(procs, SYMBOLS)))
        await asyncio.gather(*(p.wait() for p in streamed))
        return streamed

    procs = asyncio.run(_orchestrate())
    codes = [p.returncode for p in procs]

    # Collect reports.
    reports: list[dict] = []
    for sym in SYMBOLS:
        rep = _collect_report(logs_dir, sym, ts)
        if rep is not None:
            reports.append(rep)

    if not reports:
        print("ERROR: no reports were produced. Check the per-symbol logs.")
        return 1

    # Build combined markdown report.
    lines = [
        "# Kairon 20-Minute Bybit Testnet Session Report",
        "",
        f"- Session ID: `testnet_20min_{ts}`",
        f"- Duration: {DURATION_SECONDS // 60} minutes",
        f"- Symbols: {', '.join(SYMBOLS)}",
        f"- Worker return codes: {dict(zip(SYMBOLS, codes))}",
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
        rpt = rep.get("report", {})
        orders = int(getattr(rpt, "n_orders", 0))
        fills = int(getattr(rpt, "n_fills", 0))
        trades = int(getattr(rpt, "n_trades", 0))
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
    for rep in reports:
        sym = rep["symbol"]
        rpt = rep.get("report")
        if rpt is None:
            continue
        lines.append(f"### {sym}")
        lines.append("")
        lines.append("```")
        from kairon.live.analytics import format_report
        lines.append(format_report(rpt))
        lines.append("```")
        lines.append("")

    report_path = reports_dir / f"testnet_20min_{ts}.md"
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
