"""Analyze screen (US-007).

A progress ring + the current stage label, plus a small underlined
"cancel" text link. No header, no sidebar, no persistent chrome.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

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


@router.post("/api/runs")
async def start_run(request: Request) -> JSONResponse:
    """Start a run for the given run_id + horizon. Returns the status URL.

    The actual analysis runs synchronously in this v1 (we re-use
    ``run_analysis`` from the CLI pipeline). A future iteration may
    background this via BackgroundTasks.
    """
    import json
    from pathlib import Path

    from kairon.analysis.contracts import ProvenanceBlock
    from kairon.analysis.engine import build_run_result, run_analysis
    from kairon.analysis.loader import load_csv
    from kairon.store.runs import RunStore

    body = await request.body()
    payload = json.loads(body or b"{}")
    run_id = str(payload.get("run_id", ""))
    horizon = str(payload.get("horizon", "day"))
    csv_path = Path(str(payload.get("csv_path", "")))
    if not run_id or not csv_path.exists():
        return JSONResponse({"error": "missing run_id or csv_path"}, status_code=400)

    # Persist the run via RunStore (idempotent if user retries)
    store: RunStore = request.app.state.run_store
    try:
        result = load_csv(csv_path)
        analysis = run_analysis(
            result.table, symbol=result.symbol, timeframe=result.timeframe, horizon=horizon
        )
        prov = ProvenanceBlock(
            config_hash="web-v1", data_hash=run_id, model_version="kairon-0.1.0", seed=42
        )
        run = build_run_result(
            analysis, horizon=horizon, run_id=run_id, csv_path=csv_path, provenance=prov
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
