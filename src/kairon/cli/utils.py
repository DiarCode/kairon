"""Console helpers for the Kairon CLI."""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

from rich.console import Console

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Use force_terminal=True and safe encoding to avoid Windows cp1252 issues
console = Console(force_terminal=True, legacy_windows=False)


def console_info(msg: str) -> None:
    """Print an info message to the console."""
    console.print(f"[bold blue]INFO[/] {msg}")


def console_warn(msg: str) -> None:
    """Print a warning message to the console."""
    console.print(f"[bold yellow]WARN[/] {msg}")


def console_error(msg: str) -> None:
    """Print an error message to the console."""
    console.print(f"[bold red]ERR[/] {msg}")


def console_success(msg: str) -> None:
    """Print a success message to the console."""
    console.print(f"[bold green]OK[/] {msg}")


def open_in_browser(path: Path) -> None:
    """Open a file in the default web browser."""
    abs_path = path.resolve()
    url = abs_path.as_uri()
    console_info(f"Opening dashboard in browser: {url}")
    webbrowser.open(url)