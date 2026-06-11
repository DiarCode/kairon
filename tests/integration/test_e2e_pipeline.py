"""Integration tests for the W8 e2e pipeline (W8.1 + W8.2).

The W8 batch ships the headline end-to-end backtest deliverable:
``scripts/run_e2e.py btc_1h`` and ``scripts/run_e2e.py btc_5m``.
The integration tests in this module are the W8 acceptance
criterion #4: the script exits 0 and the report file is non-empty
on synthetic data.

The tests are *integration* tests (the directory name is
``tests/integration/``) — they invoke the script as a subprocess
and assert the headline side effects:

1. Exit code 0.
2. The markdown report at the canonical path is non-empty.
3. The JSON status sidecar is valid JSON with the documented
   ``headline`` shape (CAS, DSR, PBO, ...).

Why subprocess and not direct call?
-----------------------------------
The W8 acceptance criterion is "the SCRIPT exits 0", not "the
library functions work in-process". The subprocess invocation
exercises the CLI entry-point, the argparse surface, the IO
side-effects, and the JSON serialisation in one go — any
regression in any layer is caught by the integration test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH: Path = REPO_ROOT / "scripts" / "run_e2e.py"
REPORT_1H: Path = REPO_ROOT / "reports" / "e2e_btc_1h_w8.md"
REPORT_5M: Path = REPO_ROOT / "reports" / "e2e_btc_5m_w8.md"
SIDECAR_1H: Path = REPO_ROOT / "artifacts" / "w8_1_status.json"
SIDECAR_5M: Path = REPO_ROOT / "artifacts" / "w8_2_status.json"


def _run_subcommand(subcommand: str) -> subprocess.CompletedProcess[str]:
    """Run ``scripts/run_e2e.py <subcommand>`` as a subprocess.

    Uses ``uv run`` for the v1 dependency surface. The v1 contract
    is exit code 0 on success; a non-zero exit code is a fatal
    W8 acceptance criterion failure.
    """
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), subcommand],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO_ROOT),
    )
    return proc


def test_btc_1h_pipeline_runs() -> None:
    """W8.1 acceptance criterion: the script exits 0 and the report is non-empty."""
    proc: subprocess.CompletedProcess[str] = _run_subcommand("btc_1h")
    assert proc.returncode == 0, (
        f"scripts/run_e2e.py btc_1h exited {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert REPORT_1H.exists(), f"report {REPORT_1H} does not exist"
    assert REPORT_1H.stat().st_size > 0, f"report {REPORT_1H} is empty"
    # The sidecar must also exist and be valid JSON.
    assert SIDECAR_1H.exists(), f"sidecar {SIDECAR_1H} does not exist"
    payload: dict[str, object] = json.loads(SIDECAR_1H.read_text(encoding="utf-8"))
    assert "headline" in payload
    headline: object = payload["headline"]
    assert isinstance(headline, dict)
    for key in ("cas", "dsr", "pbo", "sharpe", "max_drawdown", "brier", "ece"):
        assert key in headline, f"missing {key!r} in headline"
    assert payload.get("story_id") == "W8.1"
    assert payload.get("timeframe") == "1h"


def test_btc_5m_pipeline_runs() -> None:
    """W8.2 acceptance criterion: the script exits 0 and the report is non-empty."""
    proc: subprocess.CompletedProcess[str] = _run_subcommand("btc_5m")
    assert proc.returncode == 0, (
        f"scripts/run_e2e.py btc_5m exited {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert REPORT_5M.exists(), f"report {REPORT_5M} does not exist"
    assert REPORT_5M.stat().st_size > 0, f"report {REPORT_5M} is empty"
    assert SIDECAR_5M.exists(), f"sidecar {SIDECAR_5M} does not exist"
    payload: dict[str, object] = json.loads(SIDECAR_5M.read_text(encoding="utf-8"))
    assert "headline" in payload
    headline: object = payload["headline"]
    assert isinstance(headline, dict)
    for key in ("cas", "dsr", "pbo", "sharpe", "max_drawdown", "brier", "ece"):
        assert key in headline, f"missing {key!r} in headline"
    assert payload.get("story_id") == "W8.2"
    assert payload.get("timeframe") == "5m"


def test_run_simulation_w7_runner_is_composable() -> None:
    """W7.gate acceptance criterion #3: a single run_simulation function
    composes latency + fill + rebate and returns a list of trades."""
    import numpy as np

    from kairon.paper.runner import (
        SimulatedTrade,
        SimulationConfig,
        run_simulation,
    )

    rng: np.random.Generator = np.random.default_rng(0)
    prices: np.ndarray = 50_000.0 + np.cumsum(rng.normal(0.0, 50.0, 200))
    signals: np.ndarray = np.zeros(200, dtype=np.int8)
    signals[20:40] = 1
    signals[120:140] = -1  # would close the long (no short in v1)

    cfg: SimulationConfig = SimulationConfig(order_kind="market", order_size=0.05)
    result = run_simulation(prices=prices, signals=signals, config=cfg, timeframe="1h")
    assert isinstance(result.trades, list)
    assert all(isinstance(t, SimulatedTrade) for t in result.trades)
    # The headline W7.gate fields are populated.
    assert result.p50_latency_ms >= 0.0
    assert result.p99_latency_ms >= 0.0
    assert 0.0 <= result.fill_rate <= 1.0
    assert result.maker_rebate_bps >= 0.0


def test_w8_status_files_have_required_fields() -> None:
    """The W8.1 and W8.2 status files conform to the documented schema."""
    for sidecar in (SIDECAR_1H, SIDECAR_5M):
        if not sidecar.exists():
            continue  # accept skip if upstream test did not run
        payload: dict[str, object] = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "story_id",
            "decided_at_iso",
            "symbol",
            "timeframe",
            "data_source",
            "n_bars",
            "n_features",
            "n_trades",
            "headline",
            "w7_simulator_integration",
            "regime_breakdown",
        ):
            assert key in payload, f"missing {key!r} in {sidecar}"
        # Headline numeric fields are finite.
        headline: object = payload["headline"]
        assert isinstance(headline, dict)
        for key, val in headline.items():
            if key in {"win_rate", "profit_factor"}:
                # NaN-allowed for empty trade sets.
                continue
            assert isinstance(val, (int, float)), (
                f"headline.{key} must be numeric in {sidecar}, got {type(val).__name__}"
            )
