"""Vol-aware position sizing with a Kelly cap (W6.5).

This package is the *policy* layer's position-sizing surface. The
existing sizing helpers in :mod:`kairon.portfolio` (W3-era
``fixed_fraction_size``, ``kelly_size``, ``vol_target_size``,
``size_position``) remain the call-site for the v1 backtest
engine and the W3 paper trader. The vol-aware sizer in
:mod:`kairon.policy.sizer` is the call-site for the W6.4
multi-head model and the W6.5 vol-aware pipeline.

The two surfaces are independent and additive. W6.5 does NOT
replace or modify the W3 helpers; the sizer is the W6.5 acceptance
criterion surface (``size_position_vol_aware`` is the function the
W6.5 max-drawdown test calls).
"""

from kairon.policy.sizer import (
    DEFAULT_KELLY_CAP,
    DEFAULT_MAX_POSITION_EQUITY_FRACTION,
    size_position_vol_aware,
)

__all__ = [
    "DEFAULT_KELLY_CAP",
    "DEFAULT_MAX_POSITION_EQUITY_FRACTION",
    "size_position_vol_aware",
]
