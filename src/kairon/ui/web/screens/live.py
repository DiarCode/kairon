"""Live trading dashboard screen: in-process session host + SSE stream.

Renders the professional dashboard at ``/live`` and exposes the control +
real-time surface at ``/api/live/*``. Unlike ``/trade`` (polling), the
``/live`` screen subscribes to a Server-Sent Events stream that pushes a
full session snapshot every second, and the Start/Stop buttons drive a
:class:`~kairon.live.host.SessionHost` running inside the server process.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kairon.live.host import SessionError, SessionHost

router = APIRouter()


def _get_host(request: Request) -> SessionHost | None:
    """Return the SessionHost from app state, or None if unavailable."""
    return getattr(request.app.state, "session_host", None)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    """Body for ``POST /api/live/start``."""

    model_config = {"extra": "forbid"}

    symbols: list[str] = Field(default=["BTC-USDT-PERP"], min_length=1)
    timeframe: str = Field(default="1m", pattern=r"^[0-9]+(m|h|d|w)$")
    cadence_seconds: int = Field(default=60, ge=10)
    mode: str = Field(default="testnet", pattern=r"^(paper|testnet|live)$")
    max_daily_loss_pct: float = Field(default=0.03, gt=0, le=1)
    max_open_positions: int = Field(default=5, ge=1)
    warmup_bars: int = Field(default=22, ge=1)
    reconcile_interval_seconds: int = Field(default=30, ge=5)
    reconcile_grace_seconds: int = Field(default=120, ge=10)
    cooldown_seconds: float = Field(default=300.0, gt=0)
    strategy_name: str = Field(default="comprehensive")
    confirm_live: bool = Field(default=False)


# ---------------------------------------------------------------------------
# HTML screen
# ---------------------------------------------------------------------------


@router.get("/live", response_class=HTMLResponse)
async def live_screen(request: Request) -> HTMLResponse:
    """Render the live trading dashboard."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "live.html",
        {
            "request": request,
            "stage": "live",
            "stage_label": "Live Trading",
        },
    )


# ---------------------------------------------------------------------------
# SSE real-time stream
# ---------------------------------------------------------------------------


@router.get("/api/live/stream")
async def live_stream(request: Request) -> StreamingResponse:
    """Push a full session snapshot every ~1s as Server-Sent Events.

    The browser ``EventSource`` auto-reconnects on drop; the generator exits
    cleanly on client disconnect (``request.is_disconnected`` or
    ``CancelledError`` when uvicorn cancels it).
    """
    host = _get_host(request)
    if host is None:
        return JSONResponse({"error": "session_host_unavailable"}, status_code=503)

    async def event_gen() -> AsyncGenerator[str, None]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                snap = host.snapshot()
                yield f"data: {json.dumps(snap, default=str)}\n\n"
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Client closed the stream; uvicorn cancels the generator.
            return

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Control endpoints
# ---------------------------------------------------------------------------


@router.post("/api/live/start", response_class=JSONResponse)
async def live_start(req: StartRequest, request: Request) -> JSONResponse:
    """Start a hosted trading session. Default mode is Bybit testnet."""
    host = _get_host(request)
    if host is None:
        return JSONResponse({"error": "session_host_unavailable"}, status_code=503)

    # Mainnet is gated behind an explicit confirmation (server + browser).
    if req.mode == "live" and not req.confirm_live:
        return JSONResponse(
            {"error": "confirmation_required", "message": "Set confirm_live=true to start mainnet trading."},
            status_code=400,
        )

    # Build the LiveConfig from the request, mapping mode -> broker flags.
    dry_run = req.mode == "paper"
    bybit_testnet = req.mode == "testnet"  # live == mainnet
    # Grace must be >= 2 * cadence (LiveConfig validator); clamp to satisfy it.
    grace = max(req.reconcile_grace_seconds, 2 * req.cadence_seconds)

    from kairon.config import KaironSettings
    from kairon.live.config import LiveConfig

    try:
        settings = KaironSettings()
        config = LiveConfig(
            symbols=tuple(req.symbols),
            timeframe=req.timeframe,
            cadence_seconds=req.cadence_seconds,
            max_daily_loss_pct=req.max_daily_loss_pct,
            max_open_positions=req.max_open_positions,
            warmup_bars=req.warmup_bars,
            reconcile_interval_seconds=req.reconcile_interval_seconds,
            reconcile_grace_seconds=grace,
            dry_run=dry_run,
            bybit_testnet=bybit_testnet,
            bybit_tld=settings.bybit_tld,
            strategy_name=req.strategy_name,
        )
    except Exception as e:  # pydantic validation errors
        return JSONResponse({"error": "validation", "message": str(e)}, status_code=422)

    try:
        result = await host.start_session(config, settings, cooldown_seconds=req.cooldown_seconds)
    except SessionError as e:
        status = 409 if e.code == "already_running" else 500
        return JSONResponse({"error": e.code, "message": e.message}, status_code=status)
    return JSONResponse(result, status_code=200)


@router.post("/api/live/stop", response_class=JSONResponse)
async def live_stop(request: Request) -> JSONResponse:
    """Stop the active session: stop → finalize → flatten → aclose."""
    host = _get_host(request)
    if host is None:
        return JSONResponse({"error": "session_host_unavailable"}, status_code=503)
    result = await host.stop_session()
    return JSONResponse(result)


@router.post("/api/live/halt", response_class=JSONResponse)
async def live_halt(request: Request) -> JSONResponse:
    """Kill switch: keep open positions, stop placing new orders (resumable)."""
    host = _get_host(request)
    if host is None:
        return JSONResponse({"error": "session_host_unavailable"}, status_code=503)
    try:
        return JSONResponse(await host.halt())
    except SessionError as e:
        return JSONResponse({"error": e.code, "message": e.message}, status_code=409)


@router.post("/api/live/unhalt", response_class=JSONResponse)
async def live_unhalt(request: Request) -> JSONResponse:
    """Clear the halt flag and resume trading."""
    host = _get_host(request)
    if host is None:
        return JSONResponse({"error": "session_host_unavailable"}, status_code=503)
    try:
        return JSONResponse(await host.unhalt())
    except SessionError as e:
        return JSONResponse({"error": e.code, "message": e.message}, status_code=409)


@router.get("/api/live/status", response_class=JSONResponse)
async def live_status(request: Request) -> JSONResponse:
    """One-shot snapshot (same shape as the SSE stream)."""
    host = _get_host(request)
    if host is None:
        return JSONResponse({"error": "session_host_unavailable"}, status_code=503)
    return JSONResponse(host.snapshot())


__all__ = [
    "live_halt",
    "live_screen",
    "live_start",
    "live_status",
    "live_stop",
    "live_stream",
    "live_unhalt",
]
