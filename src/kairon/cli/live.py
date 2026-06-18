"""Live CLI: serve the real-time trading dashboard.

Usage::

    kairon live serve                 # http://127.0.0.1:8000/live
    kairon live serve --host 0.0.0.0 --port 8000
    kairon live serve --reload         # dev hot-reload

Starts the FastAPI app whose lifespan owns the :class:`SessionHost`, so the
``/live`` dashboard can Start/Stop a hosted trading session in-process and
stream its snapshot over SSE. Requires the ``api`` + ``web`` extras::

    uv sync --extra api --extra web
"""

from __future__ import annotations

import typer

live_app = typer.Typer(
    name="live",
    help="Serve the real-time live-trading dashboard.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@live_app.command("serve")
def serve(
    host: str = typer.Option(None, help="Bind host (defaults to settings.api_host)."),
    port: int = typer.Option(None, help="Bind port (defaults to settings.api_port)."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn auto-reload (dev)."),
) -> None:
    """Run the dashboard server (FastAPI + SessionHost + SSE)."""
    from kairon.config import KaironSettings  # noqa: PLC0415

    settings = KaironSettings()
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port

    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as e:
        raise typer.Exit(
            "uvicorn is not installed; run `uv sync --extra api --extra web` first."
        ) from e

    typer.echo(f"Kairon live dashboard → http://{bind_host}:{bind_port}/live")
    typer.echo("Paper / Bybit testnet / Mainnet modes selectable from the UI.")
    uvicorn.run(
        "kairon.api.app:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
    )


__all__ = ["live_app"]
