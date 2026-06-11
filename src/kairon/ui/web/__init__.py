"""Kairon web app — server-rendered 5-screen flow.

This package hosts the new web app (Upload -> Configure -> Analyze -> Result -> Track).
It replaces the legacy single-page dashboard that previously lived in
``kairon/ui/static/``.

The visual language is "DeFi terminal" — dark background, blue-500 accent,
strict borders with no rounding, glass cards, monospace numbers. See
``.omc/specs/deep-interview-kairon-app-flow.md`` for the spec and
``.omc/plans/plan-kairon-web-app.md`` for the consensus plan.
"""

from __future__ import annotations

from pathlib import Path

_WEB_DIR = Path(__file__).resolve().parent
_CHARTS_DIR = _WEB_DIR / "charts"


def get_web_dir() -> Path:
    """Return the absolute path of the web app's static/template directory."""
    return _WEB_DIR


def get_charts_dir() -> Path:
    """Return the absolute path where chart PNGs are written per run."""
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    return _CHARTS_DIR


__all__ = ["get_charts_dir", "get_web_dir"]
