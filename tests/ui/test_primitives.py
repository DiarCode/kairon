"""Tests for the 9 UI primitives (US-006)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from jinja2 import DictLoader, Environment

from kairon.ui.web.primitives import PRIMITIVES

CSS_PATH = Path("src/kairon/ui/web/primitives.css")

REQUIRED_CLASSES = (
    "kairon-glass-card",
    "kairon-bento-tile",
    "kairon-big-number",
    "kairon-horizon-pill",
    "kairon-chart-frame",
    "kairon-model-strip-chart",
    "kairon-status-pill",
    "kairon-primary-button",
    "kairon-text-link",
)

PRIMITIVE_NAMES = (
    "glass_card",
    "bento_tile",
    "big_number",
    "horizon_pill",
    "chart_frame",
    "model_strip_chart",
    "status_pill",
    "primary_button",
    "text_link",
)


def _env() -> Environment:
    """A fresh Jinja2 env with all 9 primitives registered as globals."""
    env = Environment(loader=DictLoader({}))
    for name, macro in PRIMITIVES.items():
        env.globals[name] = macro
    return env


# ---------- PRIMITIVE EXPORTS -----------------------------------------------


def test_all_9_primitives_are_exported() -> None:
    for name in PRIMITIVE_NAMES:
        assert name in PRIMITIVES
        assert callable(PRIMITIVES[name])


# ---------- RENDER WITHOUT ERROR -------------------------------------------


@pytest.mark.parametrize("name", PRIMITIVE_NAMES)
def test_each_primitive_renders_without_error(name: str) -> None:
    env = _env()
    template = f"{{{{ {name}('') }}}}" if name not in {"big_number", "horizon_pill", "status_pill", "primary_button"} else f"{{{{ {name}('x') }}}}"
    if name in {"primary_button", "text_link"}:
        template = f"{{{{ {name}('go', href='/x') }}}}"
    if name in {"chart_frame", "model_strip_chart"}:
        template = f"{{{{ {name}('/x.png') }}}}"
    if name == "status_pill":
        template = "{{ status_pill('hit') }}"
    if name == "horizon_pill":
        template = "{{ horizon_pill('day') }}"
    if name == "big_number":
        template = "{{ big_number(0.5, '%') }}"
    out = env.from_string(template).render()
    assert isinstance(out, str)
    assert len(out) > 0


def test_glass_card_renders_with_class() -> None:
    env = _env()
    out = env.from_string("{{ glass_card('hi') }}").render()
    assert 'class="kairon-glass-card"' in out
    assert "hi" in out


def test_primary_button_has_glow_in_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    assert "0 0 20px rgba(59, 130, 246, 0.4)" in css
    assert ".kairon-primary-button" in css


def test_text_link_has_no_glow_in_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    # find the .kairon-text-link block
    m = re.search(r"\.kairon-text-link\s*\{[^}]*\}", css, re.DOTALL)
    assert m is not None
    block = m.group(0)
    assert "box-shadow" not in block


def test_monospace_primitives_use_font_mono() -> None:
    """BigNumber, HorizonPill, ChartFrame use JetBrains Mono (var(--font-mono)).

    For ChartFrame the numeric content lives in the caption (axis labels),
    so we check the caption block rather than the frame block.
    """
    css = CSS_PATH.read_text(encoding="utf-8")
    for cls in (".kairon-big-number", ".kairon-horizon-pill"):
        m = re.search(re.escape(cls) + r"\s*\{[^}]*\}", css, re.DOTALL)
        assert m is not None, f"no block for {cls}"
        assert "var(--font-mono)" in m.group(0), f"{cls} does not use var(--font-mono)"
    cap = re.search(r"\.kairon-chart-frame-caption\s*\{[^}]*\}", css, re.DOTALL)
    assert cap is not None, "no .kairon-chart-frame-caption block"
    assert "var(--font-mono)" in cap.group(0)


def test_radius_zero_everywhere() -> None:
    """No primitive sets border-radius to anything but 0."""
    css = CSS_PATH.read_text(encoding="utf-8")
    # the only border-radius values allowed in primitives.css are 0
    radii = re.findall(r"border-radius:\s*([^;]+);", css)
    assert radii, "expected at least one border-radius declaration"
    for r in radii:
        assert r.strip() == "0", f"non-zero border-radius: {r}"


# ---------- CSS CLASSES PRESENT --------------------------------------------


@pytest.mark.parametrize("cls", REQUIRED_CLASSES)
def test_required_css_class_present(cls: str) -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    assert cls in css


# ---------- ESCAPE HATCHES --------------------------------------------------


def test_primitives_py_has_no_any_or_cast() -> None:
    src = Path("src/kairon/ui/web/primitives.py").read_text(encoding="utf-8")
    assert not re.search(r"\b(Any|cast)\b", src)
    assert "# type: ignore" not in src
