"""Kairon web UI primitives (US-006).

9 primitives, each a Jinja2 macro + a CSS class set. All visual language
is driven by the token variables in ``tokens.css``; primitives never
hardcode colors, fonts, or radii.

Primitives:
- GlassCard         — base container (frosted, no rounding)
- BentoTile         — sub-tile inside a GlassCard
- BigNumber         — extra-large monospace number with unit
- HorizonPill       — day / swing / long label
- ChartFrame        — wraps a static PNG (axis labels + 1px border)
- ModelStripChart   — 30-bar mini chart for a model tile
- StatusPill        — verification status (hit / missed / pending)
- PrimaryButton     — blue-500 filled with subtle glow
- TextLink          — underlined blue-500, no glow

Design deviation: the plan called for 18 files (9 .py + 9 .css). The
spec only requires the 9 primitives to be exported from a single
``ui/primitives/`` package. To keep the visual-language surface in one
place, the CSS lives in a single ``primitives.css`` and the macros in a
single ``primitives.py``. The test asserts each macro is exported and
each CSS class is present; this is functionally equivalent to 18 files
with much less drift surface.
"""

from __future__ import annotations

from jinja2 import pass_context
from jinja2.runtime import Context

# All macros are @pass_context so the Jinja2 env (and template context)
# are available if a primitive ever needs to resolve URLs etc.

# ---------------------------------------------------------------------------
# GlassCard — base container
# ---------------------------------------------------------------------------


@pass_context
def glass_card(ctx: Context, body: str = "", *, extra_class: str = "") -> str:
    """Frosted-glass container. Wraps ``body`` in a div with the GlassCard class."""
    cls = f"kairon-glass-card {extra_class}".strip()
    return f'<div class="{cls}">{body}</div>'


# ---------------------------------------------------------------------------
# BentoTile — sub-tile inside a GlassCard
# ---------------------------------------------------------------------------


@pass_context
def bento_tile(ctx: Context, body: str = "", *, extra_class: str = "") -> str:
    """Sub-tile inside a GlassCard. Same visual language, smaller padding."""
    cls = f"kairon-bento-tile {extra_class}".strip()
    return f'<div class="{cls}">{body}</div>'


# ---------------------------------------------------------------------------
# BigNumber — extra-large monospace number with unit
# ---------------------------------------------------------------------------


@pass_context
def big_number(ctx: Context, value: str | float, unit: str = "", *, extra_class: str = "") -> str:
    """Render a large monospace number with a smaller unit label."""
    cls = f"kairon-big-number {extra_class}".strip()
    return (
        f'<span class="{cls}">'
        f'<span class="kairon-big-number-value">{value}</span>'
        f'<span class="kairon-big-number-unit">{unit}</span>'
        f"</span>"
    )


# ---------------------------------------------------------------------------
# HorizonPill — day / swing / long label
# ---------------------------------------------------------------------------


_HORIZON_VARIANT: dict[str, str] = {
    "day": "kairon-horizon-pill--day",
    "swing": "kairon-horizon-pill--swing",
    "long": "kairon-horizon-pill--long",
}


@pass_context
def horizon_pill(ctx: Context, horizon: str, *, extra_class: str = "") -> str:
    """Render a horizon label (day / swing / long) as a monospace pill."""
    variant = _HORIZON_VARIANT.get(horizon, "kairon-horizon-pill--day")
    cls = f"kairon-horizon-pill {variant} {extra_class}".strip()
    return f'<span class="{cls}">{horizon}</span>'


# ---------------------------------------------------------------------------
# ChartFrame — wraps a static PNG with axis labels + 1px border
# ---------------------------------------------------------------------------


@pass_context
def chart_frame(ctx: Context, png_url: str, *, caption: str = "", extra_class: str = "") -> str:
    """Wrap a static PNG in a 1px-bordered frame. No interactivity."""
    cls = f"kairon-chart-frame {extra_class}".strip()
    cap_block = (
        f'<figcaption class="kairon-chart-frame-caption">{caption}</figcaption>' if caption else ""
    )
    return (
        f'<figure class="{cls}">'
        f'<img src="{png_url}" alt="" class="kairon-chart-frame-img" />'
        f"{cap_block}"
        f"</figure>"
    )


# ---------------------------------------------------------------------------
# ModelStripChart — small 30-bar mini chart for a model tile
# ---------------------------------------------------------------------------


@pass_context
def model_strip_chart(ctx: Context, png_url: str, *, extra_class: str = "") -> str:
    """30-bar mini chart; visually a smaller ChartFrame."""
    cls = f"kairon-model-strip-chart {extra_class}".strip()
    return f'<img src="{png_url}" alt="" class="{cls}" />'


# ---------------------------------------------------------------------------
# StatusPill — verification status (hit / missed / pending)
# ---------------------------------------------------------------------------


_STATUS_VARIANT: dict[str, str] = {
    "hit": "kairon-status-pill--hit",
    "missed": "kairon-status-pill--missed",
    "pending": "kairon-status-pill--pending",
}


@pass_context
def status_pill(ctx: Context, status: str, *, extra_class: str = "") -> str:
    """Render a verification status pill. Status ∈ {hit, missed, pending}."""
    variant = _STATUS_VARIANT.get(status, "kairon-status-pill--pending")
    icon = {"hit": "✓", "missed": "✗", "pending": "·"}.get(status, "·")
    cls = f"kairon-status-pill {variant} {extra_class}".strip()
    return (
        f'<span class="{cls}"><span class="kairon-status-pill-icon">{icon}</span> {status}</span>'
    )


# ---------------------------------------------------------------------------
# PrimaryButton — blue-500 filled with subtle glow
# ---------------------------------------------------------------------------


@pass_context
def primary_button(
    ctx: Context, label: str, *, href: str | None = None, extra_class: str = ""
) -> str:
    """Render a primary action. Use ``href`` for a link-styled button."""
    cls = f"kairon-primary-button {extra_class}".strip()
    if href is not None:
        return f'<a href="{href}" class="{cls}">{label}</a>'
    return f'<button type="button" class="{cls}">{label}</button>'


# ---------------------------------------------------------------------------
# TextLink — underlined blue-500, no glow
# ---------------------------------------------------------------------------


@pass_context
def text_link(ctx: Context, label: str, href: str, *, extra_class: str = "") -> str:
    """Underlined link; no box-shadow glow."""
    cls = f"kairon-text-link {extra_class}".strip()
    return f'<a href="{href}" class="{cls}">{label}</a>'


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


PRIMITIVES: dict[str, object] = {
    "glass_card": glass_card,
    "bento_tile": bento_tile,
    "big_number": big_number,
    "horizon_pill": horizon_pill,
    "chart_frame": chart_frame,
    "model_strip_chart": model_strip_chart,
    "status_pill": status_pill,
    "primary_button": primary_button,
    "text_link": text_link,
}


__all__ = [
    "PRIMITIVES",
    "bento_tile",
    "big_number",
    "chart_frame",
    "glass_card",
    "horizon_pill",
    "model_strip_chart",
    "primary_button",
    "status_pill",
    "text_link",
]
