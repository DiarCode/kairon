# ADR-0004: Confidence-thresholded inference is a first-class product primitive

**Status:** Accepted
**Date:** 2026-06-05

## Context
R-002 demonstrated a real, measurable accuracy-coverage trade-off. Users benefit from being able to *see* and *adjust* this trade-off rather than have it hidden.

## Decision
- The inference pipeline outputs: direction, magnitude, vol, **calibrated probability band (5-95%)**, and a **threshold T** knob.
- The product exposes T as a user-tunable slider with a live preview of expected accuracy / coverage / cost-aware Sharpe.
- The system always reports **coverage** alongside accuracy; never a single accuracy number.
- "No signal" is a valid output and is shown honestly.

## Consequences
- More transparent product.
- A new metric (cost-aware Sharpe at threshold T) becomes primary.
- Forces calibrated probabilities (Platt / isotonic) as a hard requirement.

## Alternatives considered
- Fixed threshold: rejected (one size doesn't fit all).
- No threshold: rejected (R-002 evidence is clear that it helps).
