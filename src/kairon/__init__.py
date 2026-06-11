"""Kairon — strictly-typed, cost-aware AI market analysis and prediction platform.

This package is a research-grade ML system for short-interval trading in crypto
and US equities. The design is documented in ``docs/`` and constrained by a
set of architecture decision records in ``docs/adr/``.

Design principles (enforced by code):

1. Strict typing at every boundary (``pyright --strict`` is the CI gate).
2. Walk-forward + purging + embargo is the only valid backtest harness.
3. Architecture diversity in the ensemble (no single-model deployments).
4. Cost-aware by default; pre-cost is a diagnostic line, not a headline.
5. LLM is a reasoning layer, never a numeric oracle.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
