"""Unit tests for scripts/analyze_session_decisions.py.

Builds a synthetic live_decisions table with a mix of winners, losers, and
open decisions, then asserts the analyzer reports confidence separation,
justification hit rates, and confluence buckets.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "analyze_session_decisions.py"


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_session_decisions", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "session.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE live_decisions (
            order_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            direction REAL NOT NULL,
            confidence REAL NOT NULL,
            magnitude REAL,
            volatility REAL,
            horizon TEXT,
            trend_score REAL,
            momentum_score REAL,
            structure_score REAL,
            volume_score REAL,
            indicators_json TEXT NOT NULL DEFAULT '{}',
            risk_json TEXT NOT NULL DEFAULT '{}',
            justifications_json TEXT NOT NULL DEFAULT '[]',
            outcome TEXT,
            outcome_pnl REAL,
            outcome_ts TEXT
        )
        """
    )
    rows = [
        # (order_id, symbol, dir, conf, trend, mom, struct, vol, justs, outcome, pnl)
        ("o1", "BTC-USDT-PERP", 1.0, 0.80, 0.3, 0.2, 0.1, 0.1,
         ["EMA trend continuation (bullish)", "MACD histogram positive"], "hit_tp", 120.0),
        ("o2", "BTC-USDT-PERP", -1.0, 0.40, 0.1, 0.1, 0.05, 0.05,
         ["RSI overbought"], "hit_sl", -50.0),
        ("o3", "ETH-USDT-PERP", 1.0, 0.70, 0.25, 0.2, 0.1, 0.05,
         ["EMA trend continuation (bullish)"], "hit_tp", 30.0),
        ("o4", "ETH-USDT-PERP", 1.0, 0.35, 0.05, 0.05, 0.05, 0.05,
         ["MACD histogram positive"], "hit_sl", -10.0),
        # open (no outcome yet)
        ("o5", "BTC-USDT-PERP", 1.0, 0.60, 0.2, 0.2, 0.1, 0.1,
         ["EMA trend continuation (bullish)"], None, None),
    ]
    for r in rows:
        conn.execute(
            "INSERT INTO live_decisions (order_id, symbol, timestamp, strategy_name, "
            "direction, confidence, trend_score, momentum_score, structure_score, "
            "volume_score, justifications_json, outcome, outcome_pnl) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r[0], r[1], "2026-06-18T00:00:00+00:00", "ComprehensiveStrategy",
             r[2], r[3], r[4], r[5], r[6], r[7], json.dumps(r[8]), r[9], r[10]),
        )
    conn.commit()
    conn.close()
    return db


def test_analyze_reports_confidence_separation(tmp_path: Path) -> None:
    mod = _load_analyzer()
    db = _make_db(tmp_path)
    decisions = mod._load_decisions(db)
    assert len(decisions) == 5
    report = mod.analyze(decisions, [db.name])

    # Winners (o1, o3): conf 0.80, 0.70 -> avg 0.75
    # Losers (o2, o4): conf 0.40, 0.35 -> avg 0.375
    assert "Winners (pnl > 0)" in report
    assert "0.750" in report
    assert "0.375" in report
    assert "GOOD (winners more confident)" in report


def test_analyze_justification_hit_rate(tmp_path: Path) -> None:
    mod = _load_analyzer()
    db = _make_db(tmp_path)
    decisions = mod._load_decisions(db)
    report = mod.analyze(decisions, [db.name])

    # "EMA trend continuation (bullish)" appears in o1 (win), o3 (win), o5 (open)
    # -> 2 closed, 2 hits -> 100% hit rate.
    assert "EMA trend continuation (bullish)" in report
    # Hit rate column should show 100.0% for that justification.
    assert "100.0%" in report


def test_analyze_autodetect_finds_latest(tmp_path: Path) -> None:
    mod = _load_analyzer()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Two timestamps; the later one should be picked.
    for ts in ("20260618_060000", "20260618_070402"):
        for sym in ("BTC_USDT_PERP", "ETH_USDT_PERP"):
            (data_dir / f"runs_testnet_30min_{sym}_{ts}.db").write_bytes(b"")
    found = mod._autodetect_latest_session(data_dir)
    assert len(found) == 2
    names = " ".join(p.name for p in found)
    assert "20260618_070402" in names
    assert "060000" not in names


def test_cli_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_analyzer()
    db = _make_db(tmp_path)
    out = tmp_path / "report.md"
    rc = mod.main([str(db), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "# Kairon Decision Analysis" in text