"""Analyze screen (US-007).

A progress ring + the current stage label, plus a small underlined
"cancel" text link. No header, no sidebar, no persistent chrome.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from kairon.analysis.contracts import HorizonName, ProvenanceBlock

router = APIRouter()


@router.get("/analyze", response_class=HTMLResponse)
async def analyze_screen(request: Request) -> HTMLResponse:
    """Render the Analyze screen with the first stage active."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "analyze.html",
        {"request": request, "stage": "parsing", "stage_label": "Parsing CSV"},
    )


# Module-level allowlist of horizons; kept in sync with HorizonName.
# We don't reach for ``cast`` because src/kairon/ui/web/ has a CI guard
# against Any/cast/type:ignore per AGENTS.md.
_VALID_HORIZONS: frozenset[str] = frozenset(("day", "swing", "long"))

# Canonical base directory for run data. Mirrors the convention in
# ``upload_csv`` (which writes to ``<CWD>/runs/<run_id>/input.csv``).
_RUNS_DIR = Path("runs")


def _resolve_csv_path(run_id: str) -> Path:
    """Return the canonical CSV path for *run_id*.

    The upload handler writes to ``<CWD>/runs/<run_id>/input.csv``.
    Deriving the path server-side avoids the client needing to know
    the filesystem layout and eliminates the leading-slash bug where
    ``/runs/<id>/input.csv`` resolved against the filesystem root.
    """
    return _RUNS_DIR / run_id / "input.csv"


@router.post("/api/runs")
async def start_run(request: Request) -> JSONResponse:
    """Start a run for the given run_id + horizon. Returns the status URL.

    The actual analysis runs synchronously in this v1 (we re-use
    ``run_analysis`` from the CLI pipeline). A future iteration may
    background this via BackgroundTasks.
    """
    import json

    from kairon.analysis.engine import build_run_result, run_analysis
    from kairon.analysis.loader import load_csv
    from kairon.store.runs import RunStore

    body = await request.body()
    payload = json.loads(body or b"{}")
    run_id = str(payload.get("run_id", ""))
    raw_horizon = str(payload.get("horizon", "day"))

    if not run_id:
        return JSONResponse({"error": "missing run_id"}, status_code=400)

    csv_path = _resolve_csv_path(run_id)
    if not csv_path.exists():
        return JSONResponse(
            {"error": f"csv not found for run_id {run_id}"}, status_code=400
        )

    # The horizon comes from the wire as a plain string; build_run_result
    # requires a HorizonName literal. Validate it before passing through.
    if raw_horizon not in _VALID_HORIZONS:
        return JSONResponse(
            {"error": f"unknown horizon {raw_horizon!r}; expected day|swing|long"},
            status_code=400,
        )
    # Membership in _VALID_HORIZONS (= HorizonName's literal set) is the
    # narrowing gate; we annotate the local explicitly so the rest of the
    # function sees HorizonName and not ``str``.
    horizon: HorizonName
    if raw_horizon == "day":
        horizon = "day"
    elif raw_horizon == "swing":
        horizon = "swing"
    else:
        horizon = "long"

    # Persist the run via RunStore (idempotent if user retries)
    store: RunStore = request.app.state.run_store
    try:
        result = load_csv(csv_path)
        analysis = run_analysis(
            result.table,
            symbol=result.symbol,
            timeframe=result.timeframe.name,
            horizon=horizon,
        )
        prov = ProvenanceBlock(
            config_hash="web-v1", data_hash=run_id, model_version="kairon-0.1.0", seed=42
        )
        run = build_run_result(
            analysis,
            horizon=horizon,
            run_id=run_id,
            csv_path=csv_path,
            provenance=prov,
        )
        store.create(run, csv_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"run_id": run_id, "status_url": f"/api/runs/{run_id}", "status": "done"})


@router.get("/api/runs/{run_id}")
async def run_status(run_id: str, request: Request) -> JSONResponse:
    """Return the current stage of a run. Always 'done' in v1 (sync)."""
    store = request.app.state.run_store
    run = store.get(run_id)
    if run is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"run_id": run_id, "status": "done"})


@router.post("/api/runs/{run_id}/save")
async def save_run(run_id: str, request: Request) -> JSONResponse:
    """Pin/unpin a run. The web UI's [Save] button wires here."""
    import json

    body = await request.body()
    payload = json.loads(body or b"{}")
    pinned = bool(payload.get("pinned", True))
    store = request.app.state.run_store
    store.set_pinned(run_id, pinned)
    return JSONResponse({"run_id": run_id, "pinned": pinned})
