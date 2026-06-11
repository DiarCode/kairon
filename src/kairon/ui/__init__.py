"""Legacy single-page dashboard.

This module's only purpose is to keep a stable import path for any external
caller that still references ``kairon.ui.get_static_dir``. The old static
dashboard at ``kairon/ui/static/`` was replaced in 2026-06 by the new
5-screen web app at ``kairon/ui/web/`` (see ``.omc/specs/deep-interview-kairon-app-flow.md``).

The :func:`get_static_dir` helper now returns the path of the new web app
directory for backward compatibility, but new code should import from
:mod:`kairon.ui.web` instead.
"""

from __future__ import annotations

from pathlib import Path

from kairon.ui.web import get_web_dir as _get_web_dir

_DEPRECATED_STATIC_DIR = _get_web_dir()


def get_static_dir() -> Path:
    """Return the path of the new web app directory (legacy alias)."""
    return _DEPRECATED_STATIC_DIR


__all__ = ["get_static_dir"]
