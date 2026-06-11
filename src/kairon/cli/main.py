"""Kairon CLI entry point."""

from __future__ import annotations

import os

# Ensure UTF-8 encoding on Windows to avoid cp1252 errors
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import typer

app = typer.Typer(
    name="kairon",
    help="Kairon — ML-powered market analysis and prediction.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _register_commands() -> None:
    """Register all CLI subcommands."""
    from kairon.cli.analyze import analyze_app  # noqa: F401
    from kairon.cli.report import report_app  # noqa: F401

    app.add_typer(analyze_app, name="analyze")
    app.add_typer(report_app, name="report")


_register_commands()