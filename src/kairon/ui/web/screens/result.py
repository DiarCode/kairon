"""Result screen (US-007).

A 4-tile bento (trend / mean_reversion / volatility / ensemble) plus
[Save] and [New Analysis] buttons. Reads ?from= to populate a
``Back-Context`` response header (track or analyze), so the front-end
back button can route correctly.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()


@router.get("/result/{run_id}", response_class=HTMLResponse)
async def result_screen(run_id: str, request: Request) -> HTMLResponse:
    """Render the Result screen for a given run."""
    templates = request.app.state.templates
    store = request.app.state.run_store
    run = store.get(run_id)
    if run is None:
        # Render a minimal "not found" bento rather than a 404 page; the spec
        # is "no header, no sidebar" — we just show a single GlassCard.
        from starlette.responses import Response

        return Response("Run not found", status_code=404)

    # Resolve back target from ?from= (preferred) or Referer (fallback)
    from_param = request.query_params.get("from", "")
    referer = request.headers.get("referer", "")
    if from_param in ("track", "analyze"):
        back_target = from_param
    elif "/track" in referer:
        back_target = "track"
    else:
        back_target = "analyze"

    # Render with the back target available as a header
    response = templates.TemplateResponse(
        request, "result.html", {"request": request, "run": run, "back_target": back_target}
    )
    response.headers["Back-Context"] = back_target
    return response


@router.get("/track", response_class=HTMLResponse)
async def track_screen(request: Request) -> HTMLResponse:
    """Render the Track screen: a 7-column table of past runs."""
    from kairon.ui.web.state import TrackRow

    templates = request.app.state.templates
    store = request.app.state.run_store
    runs = store.list_runs()
    rows: list[TrackRow] = []
    for r in runs:
        v = store.get_verification(r.run_id)
        actual = v[0] if v is not None else None
        delta = v[1] if v is not None else None
        status = v[2] if v is not None else "pending"
        rows.append(
            TrackRow(
                run_id=r.run_id,
                asset=r.asset,
                horizon=r.horizon,
                date=r.created_at_utc,
                predicted_pct=r.models[0].predicted_pct if r.models else 0.0,
                actual_pct=actual,
                delta_pct=delta,
                status=status,
            )
        )
    return templates.TemplateResponse(
        request, "track.html", {"request": request, "rows": rows}
    )
