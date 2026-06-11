"""Session-wide test configuration + hermeticity guards (US-005).

This conftest installs two autouse session-scoped fixtures:

1. **Static-import drift guard.** A session-scoped autouse fixture walks
   every ``.py`` under ``src/kairon/`` and fails the session if it finds
   a top-level ``import ccxt.async_support`` or
   ``from ccxt.async_support import ...`` anywhere other than the single
   allowed path ``src/kairon/data/adapters/_ccxt_client.py``. This catches
   future drift if someone re-imports the ccxt client directly in a new
   module, bypassing the seam.

2. **Web tree escape-hatch guard.** A session-scoped autouse fixture
   greps the new web surface for ``Any``, ``cast``, or
   ``# type: ignore`` and fails the test session on any match (per
   ``AGENTS.md``). The exact pattern is mirrored in
   ``scripts/check_no_any_in_web.py`` for the CI command line.

We deliberately do NOT probe live network sockets here: this dev box
*has* a network, and the right guarantee is "test code never imports
``ccxt.async_support`` outside the seam". Tests that need to exercise
network-touching code paths patch ``make_client`` with a mock; the drift
guard above enforces that the only real caller is the seam module.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Single allowed module to import ccxt.async_support from.
_ALLOWED_CCXT_ASYNC_IMPORT_PATH = "src/kairon/data/adapters/_ccxt_client.py"

# Forbidden escape hatches in the new web surface (per AGENTS.md).
# We match `Any` / `cast` only when used as a type annotation, not when
# they appear inside a string literal (regex, docstring, etc.).
_WEB_TREE = Path("src/kairon/ui/web")
_FORBIDDEN_WEB_PATTERN = re.compile(
    r":\s*Any\b"  # : Any  (annotation)
    r"|->\s*Any\b"  # -> Any  (return)
    r"|\[\s*Any\s*\]"  # list[Any], dict[..., Any], tuple[Any, ...]
    r"|\bcast\s*\("  # cast(...)
    r"|#\s*type:\s*ignore"  # # type: ignore
)


# ---------------------------------------------------------------------------
# 1. Static-import drift guard — autouse session fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _enforce_ccxt_seam() -> None:
    """Fail the session if any module outside the seam imports ccxt.async_support."""
    root = Path("src/kairon")
    if not root.is_dir():
        return
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        rel = path.as_posix()
        if rel == _ALLOWED_CCXT_ASYNC_IMPORT_PATH:
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if "import ccxt.async_support" in src or "from ccxt.async_support" in src:
            offenders.append(rel)
    if offenders:
        pytest.fail(
            "ccxt.async_support must only be imported from "
            f"{_ALLOWED_CCXT_ASYNC_IMPORT_PATH}; offenders: {offenders}"
        )


# ---------------------------------------------------------------------------
# 2. Web tree escape-hatch guard — autouse session fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _enforce_web_tree_escape_hatches() -> None:
    """Fail the session if src/kairon/ui/web/ contains Any / cast / type: ignore."""
    if not _WEB_TREE.is_dir():
        return
    offenders: list[str] = []
    for path in _WEB_TREE.rglob("*.py"):
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if _FORBIDDEN_WEB_PATTERN.search(src):
            offenders.append(path.as_posix())
    if offenders:
        pytest.fail(
            f"src/kairon/ui/web/ contains forbidden escape hatches (Any/cast/"
            f"type:ignore) in: {offenders}. Per AGENTS.md, write proper types "
            f"or add a written justification comment."
        )
