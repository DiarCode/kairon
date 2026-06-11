"""Upload screen (US-007).

The only element on this screen is the drop zone + filename/row count + a
disabled [Analyze] button (enabled once a CSV is dropped). No header, no
sidebar, no persistent chrome.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from kairon.ui.web.primitives import PRIMITIVES

router = APIRouter()
_PRIM = PRIMITIVES


@router.get("/upload", response_class=HTMLResponse)
async def upload_screen(request: Request) -> HTMLResponse:
    """Render the Upload screen. Empty form on first visit."""
    templates = request.app.state.templates
    ctx: dict[str, object] = {
        "request": request,
        "filename": None,
        "row_count": None,
    }
    return templates.TemplateResponse(request, "upload.html", ctx)


@router.post("/api/uploads")
async def upload_csv(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a CSV upload, return ``{"run_id": ..., "row_count": ...}``.

    Saves the file under ``runs/<run_id>/input.csv`` on the server.
    """
    if file is None or file.filename is None:
        return JSONResponse({"error": "no file"}, status_code=400)
    run_id = uuid.uuid4().hex[:12]
    runs_dir = Path("runs")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "input.csv"
    content = await file.read()
    out_path.write_bytes(content)
    # count non-empty lines (excluding header)
    raw = content.decode("utf-8", errors="replace")
    rows = [ln for ln in raw.splitlines() if ln.strip()]
    row_count = max(0, len(rows) - 1)
    return JSONResponse(
        {"run_id": run_id, "row_count": row_count, "csv_path": str(out_path), "filename": file.filename}
    )
