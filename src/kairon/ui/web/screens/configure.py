"""Configure screen (US-007).

Three horizon tiles (day / swing / long) with one selected by default
(day), and an [Analyze] button. No header, no sidebar, no persistent
chrome.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/configure", response_class=HTMLResponse)
async def configure_screen(request: Request) -> HTMLResponse:
    """Render the Configure screen with horizon=day selected by default."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "configure.html", {"request": request, "selected": "day"}
    )
