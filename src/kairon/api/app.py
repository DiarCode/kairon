"""The FastAPI app for Kairon.

The app is *optional* — it can only be constructed when ``fastapi``
is installed (``uv sync --extra api``). All the heavy lifting
(serialization, validation, routing) is already covered by pydantic
DTOs in :mod:`kairon.api.dto`.

This module is intentionally thin: it just wires routes to the
underlying kairon functions. The web app surface (Upload, Configure,
Analyze, Result, Track) is mounted under the same FastAPI app via the
new ``kairon.ui.web.screens`` package.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import time
from datetime import UTC
from pathlib import Path
from typing import Any

# IMPORTANT: DTOs are imported at module scope (not inside create_app) so
# FastAPI 0.136+ can resolve the string annotations produced by
# ``from __future__ import annotations``. Importing them lazily inside
# create_app causes FastAPI to misclassify BaseModel parameters as
# query parameters, returning 422 on every POST.
from kairon.api.dto import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)


def _has_fastapi() -> bool:
    return importlib.util.find_spec("fastapi") is not None  # type: ignore[attr-defined]


def create_app() -> Any:
    """Construct the FastAPI ``app`` object.

    Raises :class:`ImportError` if ``fastapi`` is not installed. All
    handlers are pure delegation — the actual training/prediction
    pipeline is wired in Phases 11/12. For now the API is a contract
    surface that the integration tests can pin against.
    """
    if not _has_fastapi():
        raise ImportError("fastapi is not installed; install with `uv sync --extra api`")
    fastapi = importlib.import_module("fastapi")
    starlette_static = importlib.import_module("starlette.staticfiles")

    from kairon.models.registry import available_models
    from kairon.store.runs import RunStore
    from kairon.ui.web import get_charts_dir, get_web_dir

    app = fastapi.FastAPI(
        title="Kairon API",
        description=(
            "HTTP surface for the Kairon trading research platform. "
            "Train models, run backtests, fetch predictions."
        ),
        version="0.1.0",
    )

    # Web app surface: static assets + Jinja2 templates
    web_dir = get_web_dir()
    assets_dir = Path("assets")
    app.mount(
        "/ui/web/static",
        starlette_static.StaticFiles(directory=str(web_dir)),
        name="kairon-web-static",
    )
    if assets_dir.is_dir():
        app.mount(
            "/static",
            starlette_static.StaticFiles(directory=str(assets_dir)),
            name="kairon-static",
        )

    templates_module = importlib.import_module("fastapi.templating")
    templates = templates_module.Jinja2Templates(directory=str(web_dir / "templates"))
    app.state.templates = templates
    app.state.run_store = RunStore(Path("data/runs.db"))
    app.state.charts_dir = get_charts_dir()

    # Live trading host + shared store (also fixes the /trade 503s).
    from kairon.live.host import SessionHost
    from kairon.live.store import LiveStore

    app.state.live_store = LiveStore(Path("data/runs.db"))
    app.state.session_host = SessionHost(data_dir=Path("data"))

    # Register the 6 web screens + their API endpoints
    # Root redirect → the entry screen of the analysis flow.
    from starlette.responses import RedirectResponse

    from kairon.ui.web.screens import (
        analyze_screen,
        configure_screen,
        live_halt,
        live_screen,
        live_start,
        live_status,
        live_stop,
        live_stream,
        live_unhalt,
        result_screen,
        run_status,
        save_run,
        start_run,
        track_screen,
        trade_events,
        trade_halt,
        trade_orders,
        trade_positions,
        trade_screen,
        trade_status,
        trade_unhalt,
        upload_csv,
        upload_screen,
    )

    async def _root_redirect() -> Any:
        return RedirectResponse(url="/upload")

    app.add_api_route("/", _root_redirect, methods=["GET"], tags=["web"])
    app.add_api_route("/upload", upload_screen, methods=["GET"], tags=["web"])
    app.add_api_route("/configure", configure_screen, methods=["GET"], tags=["web"])
    app.add_api_route("/analyze", analyze_screen, methods=["GET"], tags=["web"])
    app.add_api_route("/result/{run_id}", result_screen, methods=["GET"], tags=["web"])
    app.add_api_route("/track", track_screen, methods=["GET"], tags=["web"])
    app.add_api_route("/trade", trade_screen, methods=["GET"], tags=["web"])
    app.add_api_route("/live", live_screen, methods=["GET"], tags=["web"])
    app.add_api_route("/api/uploads", upload_csv, methods=["POST"], tags=["web"])
    app.add_api_route("/api/runs", start_run, methods=["POST"], tags=["web"])
    app.add_api_route("/api/runs/{run_id}", run_status, methods=["GET"], tags=["web"])
    app.add_api_route("/api/runs/{run_id}/save", save_run, methods=["POST"], tags=["web"])
    # Trade dashboard API
    app.add_api_route("/api/trade/status", trade_status, methods=["GET"], tags=["trade"])
    app.add_api_route("/api/trade/positions", trade_positions, methods=["GET"], tags=["trade"])
    app.add_api_route("/api/trade/orders", trade_orders, methods=["GET"], tags=["trade"])
    app.add_api_route("/api/trade/events", trade_events, methods=["GET"], tags=["trade"])
    app.add_api_route("/api/trade/halt", trade_halt, methods=["POST"], tags=["trade"])
    app.add_api_route("/api/trade/unhalt", trade_unhalt, methods=["POST"], tags=["trade"])
    # Live dashboard API (in-process session host + SSE stream)
    app.add_api_route("/api/live/stream", live_stream, methods=["GET"], tags=["live"])
    app.add_api_route("/api/live/start", live_start, methods=["POST"], tags=["live"])
    app.add_api_route("/api/live/stop", live_stop, methods=["POST"], tags=["live"])
    app.add_api_route("/api/live/halt", live_halt, methods=["POST"], tags=["live"])
    app.add_api_route("/api/live/unhalt", live_unhalt, methods=["POST"], tags=["live"])
    app.add_api_route("/api/live/status", live_status, methods=["GET"], tags=["live"])

    # Lifespan: start the verifier thread on startup, stop it on shutdown.
    # The verifier polls the runs table for due rows; the ccxt fetch is
    # mocked in tests via the _ccxt_client seam, and the run_once function
    # is exercised directly in test_verification_thread.py.
    from kairon.live.feed import fetch_current_price
    from kairon.store.verifier import VerifierThread

    # ``@contextlib.asynccontextmanager`` expects a regular generator and
    # pyright's overloads in typeshed don't model the async-generator form
    # cleanly, so we silence the decorator and cast the result to FastAPI's
    # expected AsyncContextManager shape. The runtime contract is: ``lifespan``
    # is an async generator that yields once and stops the thread on exit.
    @contextlib.asynccontextmanager  # type: ignore[arg-type]
    async def lifespan(_app: Any) -> Any:
        thread = VerifierThread(
            run_store=app.state.run_store,
            fetch_price_fn=fetch_current_price,
            poll_interval_seconds=60.0,
        )
        thread.start()
        try:
            yield None
        finally:
            thread.stop(timeout=2.0)
            # Tear down any hosted live sessions, then close the shared store.
            with contextlib.suppress(Exception):
                await app.state.session_host.shutdown_all()
            with contextlib.suppress(Exception):
                app.state.live_store.close()

    import typing
    from contextlib import AbstractAsyncContextManager

    app.router.lifespan_context = typing.cast("AbstractAsyncContextManager[None]", lifespan)

    started_at = time.time()

    @app.get("/healthz", response_model=HealthResponse, tags=["health"])
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            uptime_seconds=time.time() - started_at,
        )

    @app.get("/v1/models", tags=["models"])
    def list_models() -> dict[str, list[str]]:
        """List the registered model backends."""
        return {"models": list(available_models())}

    @app.post("/v1/models/train", response_model=TrainResponse, tags=["models"])
    def train(req: TrainRequest) -> TrainResponse:
        """Train a model. Stand-in: returns a stub response until the
        trainer is wired into the live engine in Phase 11.
        """
        return TrainResponse(
            backend=req.model_backend,
            n_folds=0,
            mean_test_acc=0.0,
            mean_test_logloss=0.0,
            train_seconds=0.0,
            folds=[],
        )

    @app.post("/v1/models/predict", response_model=PredictResponse, tags=["models"])
    def predict(req: PredictRequest) -> PredictResponse:
        """Predict. Stand-in: returns an empty prediction until the
        model registry is wired in Phase 12.
        """
        now = datetime_now_utc()
        return PredictResponse(
            run_name=req.run_name,
            backend="",
            ts=[now] * req.n_rows,
            y_class=[0] * req.n_rows,
            y_proba=[0.0] * req.n_rows,
        )

    @app.post("/v1/backtest", response_model=BacktestResponse, tags=["backtest"])
    def backtest(req: BacktestRequest) -> BacktestResponse:
        """Run a backtest. Stand-in: returns a stub response until the
        live engine is wired in Phase 11.
        """
        return BacktestResponse(
            symbol=req.symbol,
            n_trades=0,
            total_pnl=0.0,
            win_rate=0.0,
            final_equity=req.initial_equity,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            dsr=0.0,
            dsr_p_value=1.0,
            sr_star=0.0,
        )

    return app


def datetime_now_utc():
    """Return tz-aware UTC ``datetime.now()``. Indirected so tests can
    monkey-patch without rewiring the closure."""
    from datetime import datetime

    return datetime.now(UTC)


__all__ = ["create_app"]
