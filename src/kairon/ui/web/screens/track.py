"""Track screen (US-007).

A 7-column table of past runs (asset, horizon, date, predicted %,
actual %, delta %, status pill). Click a row -> /result/<run_id>.

This is a thin re-export of the ``track_screen`` defined in result.py to
keep file-per-screen symmetry. The import order in app.py matters.
"""

from __future__ import annotations

# track_screen is defined in result.py; re-exported here for symmetry.
from kairon.ui.web.screens.result import track_screen

__all__ = ["track_screen"]
