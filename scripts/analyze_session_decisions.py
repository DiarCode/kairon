"""Analyze trade decisions from one or more live-session SQLite stores.

Backfills the Phase 2.4 deliverable from the live-trading fixes plan. For each
session DB (one per symbol — BTC and ETH live in separate stores), it prints
and writes a markdown report covering:

  1. Decisions per symbol (count, directional split, avg confidence).
  2. Average confidence of winning vs losing decisions (outcome_pnl > 0 vs
     <= 0 among closed decisions).
  3. Most common justifications and their hit rate (fraction of closed
     decisions containing the justification that realized a profit).
  4. Confluence score buckets vs realized PnL (bucket by total confluence =
     trend + momentum + structure + volume).

Usage:
    uv run python scripts/analyze_session_decisions.py [DB_PATH ...] [--out reports/decision_analysis_<ts>.md]

With no DB_PATH arguments, auto-detects the latest pair of
``data/runs_testnet_30min_*_<ts>.db`` files (same timestamp = one session).
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass


def _load_decisions(db_path: Path) -> list[dict]:
    """Load all decision rows from a session DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT symbol, timestamp, direction, confidence, "
            "trend_score, momentum_score, structure_score, volume_score, "
            "justifications_json, outcome, outcome_pnl "
            "FROM live_decisions ORDER BY timestamp"
        ).fetchall()
    finally:
        conn.close()

    decisions: list[dict] = []
    for r in rows:
        try:
            justs = json.loads(r["justifications_json"]) if r["justifications_json"] else []
        except (ValueError, TypeError):
            justs = []
        decisions.append({
            "db": db_path.name,
            "symbol": r["symbol"],
            "direction": float(r["direction"]) if r["direction"] is not None else 0.0,
            "confidence": float(r["confidence"]) if r["confidence"] is not None else 0.0,
            "trend": _f(r["trend_score"]),
            "momentum": _f(r["momentum_score"]),
            "structure": _f(r["structure_score"]),
            "volume": _f(r["volume_score"]),
            "justifications": list(justs),
            "outcome": r["outcome"],
            "outcome_pnl": float(r["outcome_pnl"]) if r["outcome_pnl"] is not None else None,
        })
    return decisions


def _f(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def analyze(decisions: list[dict], db_names: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Kairon Decision Analysis")
    lines.append("")
    lines.append(f"- Stores analyzed: {', '.join(db_names) if db_names else '(none)'}")
    lines.append(f"- Total decisions: {len(decisions)}")
    closed = [d for d in decisions if d["outcome_pnl"] is not None]
    lines.append(f"- Closed decisions (with realized PnL): {len(closed)}")
    lines.append("")

    # 1. Decisions per symbol
    lines.append("## 1. Decisions per symbol")
    lines.append("")
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        by_sym[d["symbol"]].append(d)
    rows = []
    for sym, ds in sorted(by_sym.items()):
        n = len(ds)
        longs = sum(1 for d in ds if d["direction"] > 0)
        shorts = sum(1 for d in ds if d["direction"] < 0)
        flats = sum(1 for d in ds if d["direction"] == 0)
        avg_conf = sum(d["confidence"] for d in ds) / n if n else 0.0
        rows.append([sym, n, longs, shorts, flats, f"{avg_conf:.3f}"])
    lines.append(_md_table(
        ["Symbol", "Decisions", "Long", "Short", "Flat", "Avg confidence"], rows))
    lines.append("")

    # 2. Winning vs losing confidence
    lines.append("## 2. Avg confidence: winning vs losing decisions")
    lines.append("")
    winners = [d for d in closed if d["outcome_pnl"] > 0]
    losers = [d for d in closed if d["outcome_pnl"] <= 0]
    w_conf = sum(d["confidence"] for d in winners) / len(winners) if winners else float("nan")
    l_conf = sum(d["confidence"] for d in losers) / len(losers) if losers else float("nan")
    rows = [
        ["Winners (pnl > 0)", len(winners), f"{w_conf:.3f}" if winners else "—"],
        ["Losers (pnl <= 0)", len(losers), f"{l_conf:.3f}" if losers else "—"],
    ]
    lines.append(_md_table(["Group", "N", "Avg confidence"], rows))
    lines.append("")
    if winners and losers:
        sep_ok = w_conf > l_conf
        lines.append(
            f"Confidence separation: winners {w_conf:.3f} vs losers {l_conf:.3f} "
            f"→ {'GOOD (winners more confident)' if sep_ok else 'WEAK/INVERTED'}."
        )
    else:
        lines.append("Not enough closed winners and losers to assess separation.")
    lines.append("")

    # 3. Justification hit rates
    lines.append("## 3. Most common justifications and hit rate")
    lines.append("")
    counts = Counter()
    closed_with: dict[str, list[float]] = defaultdict(list)
    for d in decisions:
        for j in d["justifications"]:
            counts[j] += 1
            if d["outcome_pnl"] is not None:
                closed_with[j].append(d["outcome_pnl"])
    rows = []
    for j, c in counts.most_common(15):
        pnls = closed_with.get(j, [])
        if pnls:
            hits = sum(1 for p in pnls if p > 0)
            hit_rate = hits / len(pnls)
            avg_pnl = sum(pnls) / len(pnls)
            rows.append([j, c, len(pnls), f"{hit_rate:.1%}", f"{avg_pnl:+.4f}"])
        else:
            rows.append([j, c, 0, "—", "—"])
    if rows:
        lines.append(_md_table(
            ["Justification", "Decisions", "Closed", "Hit rate", "Avg realized PnL"], rows))
    else:
        lines.append("(no justifications recorded)")
    lines.append("")

    # 4. Confluence buckets vs realized PnL
    lines.append("## 4. Confluence score buckets vs realized PnL")
    lines.append("")
    buckets = {"0.0–0.3": [], "0.3–0.6": [], "0.6–0.9": [], "0.9+": []}
    for d in closed:
        total = d["trend"] + d["momentum"] + d["structure"] + d["volume"]
        if total < 0.3:
            buckets["0.0–0.3"].append((total, d["outcome_pnl"]))
        elif total < 0.6:
            buckets["0.3–0.6"].append((total, d["outcome_pnl"]))
        elif total < 0.9:
            buckets["0.6–0.9"].append((total, d["outcome_pnl"]))
        else:
            buckets["0.9+"].append((total, d["outcome_pnl"]))
    rows = []
    for label, items in buckets.items():
        if not items:
            rows.append([label, 0, "—", "—", "—"])
            continue
        avg_conf_total = sum(t for t, _ in items) / len(items)
        avg_pnl = sum(p for _, p in items) / len(items)
        wins = sum(1 for _, p in items if p > 0)
        rows.append([label, len(items), f"{avg_conf_total:.3f}", f"{avg_pnl:+.4f}",
                     f"{wins}/{len(items)}"])
    lines.append(_md_table(
        ["Confluence bucket", "N", "Avg total confluence", "Avg realized PnL", "Winners"], rows))
    lines.append("")

    return "\n".join(lines)


def _autodetect_latest_session(data_dir: Path) -> list[Path]:
    """Find the latest pair of testnet 30-min DBs sharing one timestamp."""
    files = sorted(glob.glob(str(data_dir / "runs_testnet_30min_*_*.db")),
                   key=lambda p: Path(p).name, reverse=True)
    if not files:
        return []
    # Group by timestamp suffix (last _YYYYMMDD_HHMMSS.db).
    by_ts: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        p = Path(f)
        # name: runs_testnet_30min_BTC_USDT_PERP_20260618_061548.db
        stem = p.stem  # runs_testnet_30min_BTC_USDT_PERP_20260618_061548
        ts = "_".join(stem.split("_")[-2:])  # 20260618_061548
        by_ts[ts].append(p)
    if not by_ts:
        return []
    latest_ts = max(by_ts.keys())
    return sorted(by_ts[latest_ts])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analyze_session_decisions")
    parser.add_argument("dbs", nargs="*", help="Session SQLite DB path(s) to analyze")
    parser.add_argument("--out", default=None, help="Markdown output path")
    parser.add_argument("--data-dir", default=None, help="Data dir for autodetect (default: repo data/)")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"

    if args.dbs:
        db_paths = [Path(d) for d in args.dbs]
    else:
        db_paths = _autodetect_latest_session(data_dir)

    if not db_paths:
        print("No DB paths given and none auto-detected. Pass one or more DB paths.")
        return 2

    decisions: list[dict] = []
    for p in db_paths:
        if not p.exists():
            print(f"WARNING: {p} does not exist; skipping.")
            continue
        decisions.extend(_load_decisions(p))
        print(f"Loaded {len(decisions)} cumulative decisions after {p.name}")

    if not decisions:
        print("No decisions found in any provided DB.")
        return 1

    report = analyze(decisions, [p.name for p in db_paths])
    print("\n" + report)

    out_path = Path(args.out) if args.out else repo_root / "reports" / "decision_analysis_latest.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())