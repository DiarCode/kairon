"""Trade screen: live trading dashboard with status, positions, and controls.

Provides the HTML dashboard at ``/trade`` and JSON API endpoints at
``/api/trade/*`` for the frontend to poll. The dashboard shows:

- Status panel: mode (dry_run / testnet / live), equity, PnL, uptime
- Open positions table: symbol, side, qty, avg entry, unrealized PnL
- Recent orders table: time, symbol, side, qty, status
- Kill switch: halt / unhalt buttons backed by ``LiveStore``
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from kairon.live.store import LiveStore

router = APIRouter()


def _get_store(request: Request) -> LiveStore | None:
    """Get the LiveStore from app state, or None if not initialized."""
    return getattr(request.app.state, "live_store", None)


# ---------------------------------------------------------------------------
# HTML screen
# ---------------------------------------------------------------------------


@router.get("/trade", response_class=HTMLResponse)
async def trade_screen(request: Request) -> HTMLResponse:
    """Render the live trading dashboard."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "trade.html",
        {
            "request": request,
            "stage": "live",
            "stage_label": "Live Trading",
        },
    )


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/trade/status", response_class=JSONResponse)
async def trade_status(request: Request) -> JSONResponse:
    """Return current trading status: mode, equity, PnL, positions count, halted state."""
    store = _get_store(request)
    if store is None:
        return JSONResponse(
            {"status": "unavailable", "message": "LiveStore not initialized"},
            status_code=503,
        )

    halted = store.is_halted()
    positions = store.get_positions()
    halt_reason = store.get_runtime_state("halted") if halted else None

    heartbeat = store.get_recent_heartbeat()
    if heartbeat:
        last_ts = heartbeat["ts"]
        mode = heartbeat["mode"]
        equity = heartbeat["equity"]
        n_positions = heartbeat["n_positions"]
        last_signal_ts = heartbeat["last_signal_ts"]
    else:
        last_ts, mode, equity, n_positions, last_signal_ts = None, "unknown", 0.0, 0, None

    return JSONResponse({
        "status": "halted" if halted else "running",
        "mode": mode or "unknown",
        "equity": equity,
        "n_positions": n_positions or len(positions),
        "halted": halted,
        "halt_reason": halt_reason,
        "last_heartbeat": last_ts,
        "last_signal_ts": last_signal_ts,
    })


@router.get("/api/trade/positions", response_class=JSONResponse)
async def trade_positions(request: Request) -> JSONResponse:
    """Return open positions."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"positions": []}, status_code=503)

    positions = store.get_positions()
    data = []
    for p in positions:
        data.append({
            "symbol": p.symbol,
            "side": p.side.value,
            "qty": p.qty,
            "avg_entry": p.avg_entry,
            "unrealized_pnl": p.unrealized_pnl,
            "ts": p.ts,
        })
    return JSONResponse({"positions": data})


@router.get("/api/trade/orders", response_class=JSONResponse)
async def trade_orders(request: Request) -> JSONResponse:
    """Return recent orders (last 50)."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"orders": []}, status_code=503)

    orders = store.get_recent_orders(limit=50)
    return JSONResponse({"orders": orders})


@router.get("/api/trade/events", response_class=JSONResponse)
async def trade_events(request: Request) -> JSONResponse:
    """Return recent audit events (last 50)."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"events": []}, status_code=503)

    events = store.get_recent_events(limit=50)
    data = []
    for e in events:
        data.append({
            "ts": e["ts"],
            "kind": e["kind"],
            "severity": e["severity"],
            "payload": json.loads(e["payload_json"]) if e["payload_json"] else {},
        })
    return JSONResponse({"events": data})


@router.post("/api/trade/halt", response_class=JSONResponse)
async def trade_halt(request: Request) -> JSONResponse:
    """Halt the trading loop (kill switch)."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "LiveStore not initialized"}, status_code=503)

    reason = "manual_halt_via_dashboard"
    store.halt(reason=reason)
    store.write_event(kind="halt", severity="critical", payload_json=json.dumps({"reason": reason}))
    return JSONResponse({"status": "halted", "reason": reason})


@router.post("/api/trade/unhalt", response_class=JSONResponse)
async def trade_unhalt(request: Request) -> JSONResponse:
    """Clear the halt flag and resume trading."""
    store = _get_store(request)
    if store is None:
        return JSONResponse({"error": "LiveStore not initialized"}, status_code=503)

    store.unhalt()
    store.write_event(kind="unhalt", severity="info", payload_json='{"reason": "manual_resume"}')
    return JSONResponse({"status": "running"})


__all__ = [
    "trade_screen",
    "trade_status",
    "trade_positions",
    "trade_orders",
    "trade_events",
    "trade_halt",
    "trade_unhalt",
]