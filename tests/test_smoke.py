"""Smoke test: the package imports cleanly and version is set.

This is a hermetic test that must pass on an empty tree (Phase 0 exit gate).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import kairon


def test_version_is_set() -> None:
    """The package exposes a non-empty version string."""
    assert kairon.__version__
    assert isinstance(kairon.__version__, str)
    parts = kairon.__version__.split(".")
    assert len(parts) >= 2


def test_package_imports() -> None:
    """`import kairon` succeeds without side effects."""
    mod = importlib.import_module("kairon")
    assert mod is kairon


def test_py_typed_marker_present() -> None:
    """The `py.typed` marker file exists so downstream consumers can rely on types."""
    marker = Path(kairon.__file__).parent / "py.typed"
    assert marker.exists(), "py.typed marker missing — PEP 561 conformance broken"
