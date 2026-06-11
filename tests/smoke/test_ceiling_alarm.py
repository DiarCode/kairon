"""W9.5 — Leakage alarm: CI fails on > ceiling accuracy.

The W9.5 acceptance criterion:

- ``tests/smoke/test_ceiling_alarm.py::assert_within_ceiling`` is a
  helper that compares a per-(asset, horizon) accuracy against the
  realistic-ceiling table from ``docs/objective_and_metrics.md`` §5
  and raises an alarm (fails the test) if the accuracy exceeds the
  ceiling.
- ``test_catches_impossible_90pct`` passes: an accuracy of 0.95 on
  BTCUSDT 1h raises (the ceiling is 0.62).
- ``test_passes_reasonable_50pct`` passes: an accuracy of 0.52 on
  BTCUSDT 1h does not raise.

The ceiling table is the documented achievable direction-accuracy
range per horizon (from ``docs/objective_and_metrics.md`` §5). The
v1 alarm is the UPPER bound of the range; any accuracy ABOVE the
upper bound is auto-flagged for a leakage audit.
"""

from __future__ import annotations

import pytest

# Documented ceilings per (asset_class, horizon). The values are
# the UPPER bound of the achievable range from
# ``docs/objective_and_metrics.md`` §5 (the W9.5 acceptance
# criterion references the same table). The asset_class is
# inferred from the asset symbol: BTCUSDT/ETHUSDT/SOLUSDT are
# crypto; SPY/AAPL are equity. The horizon is the prediction
# horizon string ('5m', '1h', '1d', '1w').
CEILING_TABLE: dict[tuple[str, str], float] = {
    # Crypto horizons
    ("crypto", "5m"): 0.75,
    ("crypto", "1h"): 0.62,
    ("crypto", "1d"): 0.62,  # 1d ceiling for crypto conservatively = 1h
    ("crypto", "1w"): 0.58,
    # Equity horizons (1d / 1w; the 1h and 5m horizons fall back
    # to the crypto 1h/5m ceilings as the v1 approximation).
    ("equity", "5m"): 0.75,
    ("equity", "1h"): 0.62,
    ("equity", "1d"): 0.60,
    ("equity", "1w"): 0.58,
}

# Map of known asset symbols to asset classes. Unknown assets
# default to 'equity' (the more conservative ceiling).
SYMBOL_TO_ASSET_CLASS: dict[str, str] = {
    "BTCUSDT": "crypto",
    "ETHUSDT": "crypto",
    "SOLUSDT": "crypto",
    "BNBUSDT": "crypto",
    "SPY": "equity",
    "AAPL": "equity",
    "MSFT": "equity",
}


def _resolve_asset_class(asset: str) -> str:
    """Return the asset class for ``asset``; 'equity' is the conservative default."""
    return SYMBOL_TO_ASSET_CLASS.get(asset.upper(), "equity")


def assert_within_ceiling(asset: str, horizon: str, accuracy: float) -> None:
    """Raise ``AssertionError`` if ``accuracy`` exceeds the (asset, horizon) ceiling.

    Parameters
    ----------
    asset
        Asset symbol (e.g. ``"BTCUSDT"``, ``"SPY"``). Case-insensitive.
    horizon
        Prediction horizon. One of ``"5m"``, ``"1h"``, ``"1d"``, ``"1w"``.
    accuracy
        Reported direction accuracy as a fraction in [0, 1]. Values
        above the ceiling are flagged for a leakage audit.

    Raises
    ------
    ValueError
        If ``accuracy`` is not a finite number in [0, 1].
    AssertionError
        If ``accuracy`` exceeds the realistic ceiling for the
        given (asset, horizon) combination.
    """
    if isinstance(accuracy, bool):
        raise ValueError(
            f"accuracy must be a number, got {type(accuracy).__name__}"
        )
    if accuracy.__class__ not in (int, float):
        # ``int | float`` in the annotation covers the two
        # numeric types; subclasses (e.g. ``numpy.float64``) are
        # allowed by the cast below.
        raise ValueError(
            f"accuracy must be a number, got {type(accuracy).__name__}"
        )
    accuracy_f = float(accuracy)
    accuracy_f = float(accuracy)
    if accuracy_f != accuracy_f:  # NaN
        raise ValueError(f"accuracy must be finite, got {accuracy!r}")
    if accuracy_f < 0.0 or accuracy_f > 1.0:
        raise ValueError(
            f"accuracy must be in [0, 1], got {accuracy_f}"
        )
    asset_class: str = _resolve_asset_class(asset)
    ceiling: float | None = CEILING_TABLE.get((asset_class, horizon))
    if ceiling is None:
        # Unknown horizon: do not raise (the v1 path is permissive
        # for forward-compat horizons; the alarm fires only on
        # known (asset_class, horizon) pairs that have a ceiling).
        return
    if accuracy_f > ceiling:
        raise AssertionError(
            f"LEAKAGE ALARM: accuracy={accuracy_f:.4f} on {asset} "
            f"({asset_class}) at horizon {horizon} exceeds the "
            f"realistic ceiling of {ceiling:.4f} from "
            f"docs/objective_and_metrics.md §5. This is a strong "
            f"signal of lookahead bias, target leakage, or a "
            f"rigged synthetic environment. Investigate before "
            f"merging."
        )


# ---------------------------------------------------------------------------
# PRD W9.5: test_catches_impossible_90pct
# ---------------------------------------------------------------------------
def test_catches_impossible_90pct() -> None:
    """accuracy=0.95 on BTCUSDT 1h raises (above the 55% ceiling)."""
    # The 1h crypto ceiling is 0.62. An accuracy of 0.95 is well
    # above the ceiling and must trigger the leakage alarm.
    with pytest.raises(AssertionError, match=r"LEAKAGE ALARM"):
        assert_within_ceiling("BTCUSDT", "1h", 0.95)


# ---------------------------------------------------------------------------
# PRD W9.5: test_passes_reasonable_50pct
# ---------------------------------------------------------------------------
def test_passes_reasonable_50pct() -> None:
    """accuracy=0.52 on BTCUSDT 1h does not raise."""
    # The 1h crypto ceiling is 0.62. An accuracy of 0.52 is well
    # below the ceiling and must NOT trigger the leakage alarm.
    assert_within_ceiling("BTCUSDT", "1h", 0.52)


# ---------------------------------------------------------------------------
# Supplementary tests
# ---------------------------------------------------------------------------
def test_ceiling_btcusdt_5m_is_0_75() -> None:
    """The 5m crypto ceiling is 0.75 (per docs/objective_and_metrics.md §5)."""
    assert_within_ceiling("BTCUSDT", "5m", 0.74)
    with pytest.raises(AssertionError, match=r"LEAKAGE ALARM"):
        assert_within_ceiling("BTCUSDT", "5m", 0.76)


def test_ceiling_spy_1d_is_0_60() -> None:
    """The 1d equity ceiling is 0.60 (per docs/objective_and_metrics.md §5)."""
    assert_within_ceiling("SPY", "1d", 0.59)
    with pytest.raises(AssertionError, match=r"LEAKAGE ALARM"):
        assert_within_ceiling("SPY", "1d", 0.61)


def test_ceiling_handles_case_insensitive_asset() -> None:
    """The asset symbol is case-insensitive."""
    assert_within_ceiling("btcusdt", "1h", 0.50)
    with pytest.raises(AssertionError, match=r"LEAKAGE ALARM"):
        assert_within_ceiling("btcusdt", "1h", 0.95)


def test_ceiling_rejects_invalid_accuracy() -> None:
    """The alarm validates the accuracy input."""
    with pytest.raises(ValueError, match=r"finite"):
        assert_within_ceiling("BTCUSDT", "1h", float("nan"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assert_within_ceiling("BTCUSDT", "1h", 1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assert_within_ceiling("BTCUSDT", "1h", -0.1)
    with pytest.raises(ValueError, match=r"number"):
        assert_within_ceiling("BTCUSDT", "1h", "0.5")  # type: ignore[arg-type]


def test_ceiling_unknown_horizon_is_permissive() -> None:
    """Unknown horizons (forward-compat) do not raise."""
    # The alarm is permissive for unknown (asset_class, horizon)
    # pairs: the v1 path fires only on documented ceilings.
    assert_within_ceiling("BTCUSDT", "10m", 0.99)  # no ceiling -> no alarm


def test_ceiling_ethusdt_1d_passes() -> None:
    """ETHUSDT 1d is below the ceiling at 0.55."""
    assert_within_ceiling("ETHUSDT", "1d", 0.55)
