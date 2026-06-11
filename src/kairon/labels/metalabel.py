"""Meta labels: a binary ``y_meta in {0, 1}`` derived from the existing
triple-barrier label plus the calibrated round-trip cost.

The meta-labeling idea (Lopez de Prado, *Advances in Financial
Machine Learning*, ch. 5) is that a *primary* model produces a
directional prediction; the meta-model decides whether to *trust* that
prediction, i.e. whether the trade has positive expected return *after*
costs. The primary target is typically the triple-barrier direction
(``LabelKind.TRIPLE_BARRIER``); the meta target is a binary
"is the barrier hit AND did the move exceed the round-trip cost?".

Concretely, for each bar at time ``t``:

1. Run :func:`make_triple_barrier_labels` (with ``require_finite=False``
   so vertical bars are kept -- they map to ``y_meta=0``) to get the
   per-bar ``first_hit`` (``upper`` / ``lower`` / ``vertical``) and the
   ``realized_return_bps`` from the entry close to the close at the
   first-hit bar.
2. Set ``y_meta = 1`` iff ``first_hit != 'vertical'`` AND
   ``|realized_return_bps| > cost_model.round_trip_bps``.
3. ``y_meta = 0`` otherwise. The ``0`` class is the *abstain* signal:
   "the primary prediction is not worth executing because the
   barrier hit was either non-existent (vertical) or too small to
   clear round-trip fees".

The bar-level leakage contract is inherited from
:func:`make_triple_barrier_labels`: a bar at time ``t`` can never see
any ``close[k]`` with ``k <= t``. The realised return is computed
inside the same horizon window ``[i+1, end_idx]`` that the primary
label uses. Downstream tests should still call
:func:`tests.fixtures.leakage.assert_no_leakage` on the input table
to verify the table's total span is at least ``horizon_seconds``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from kairon.labels.schema import (
    DirectionClass,
    LabeledBar,
    LabeledFrame,
    LabelKind,
    LabelSpec,
)
from kairon.labels.triple_barrier import make_triple_barrier_labels

if TYPE_CHECKING:
    from kairon.backtest.cost import CostModel


def make_metalabel_labels(
    table: pa.Table,
    *,
    spec: LabelSpec,
    symbol: str,
    cost_model: "CostModel",
    pt_pct: float = 0.01,
    sl_pct: float = 0.01,
) -> LabeledFrame:
    """Build a binary ``y_meta in {0, 1}`` label per bar.

    The implementation re-uses :func:`make_triple_barrier_labels` for
    the underlying barrier scan and realised-return calculation; the
    triple-barrier meta dict is extended (in W3.2) with ``exit_close``
    and ``realized_return_bps`` so this reader does not re-derive any
    barrier logic.

    For each emitted bar in the triple-barrier frame the metalabel is::

        y_meta = 1 iff first_hit != 'vertical' AND
                     |realized_return_bps| > cost_model.round_trip_bps
        else 0

    Parameters
    ----------
    table:
        OHLCV table conforming to ``OHLCV_SCHEMA``. The ``ts`` column
        must be sorted ascending; the recommended pre-condition is a
        call to :func:`tests.fixtures.leakage.assert_no_leakage` with
        ``label_horizon_seconds=spec.horizon_seconds``.
    spec:
        ``LabelSpec(kind=LabelKind.META, horizon=...)``. The horizon
        is forwarded to the inner triple-barrier call.
    symbol:
        Canonical symbol for ``LabeledBar`` provenance.
    cost_model:
        The :class:`CostModel` whose ``round_trip_bps`` is the
        break-even threshold. Do not pass a placeholder -- a
        calibrated cost is required to produce a meaningful meta
        label.
    pt_pct, sl_pct:
        Profit-take and stop-loss in *return* terms, forwarded to the
        inner triple-barrier call. Default 1% / 1% (matches
        :func:`make_labels` defaults).

    Returns
    -------
    LabeledFrame
        A :class:`LabeledFrame` with ``spec.kind = LabelKind.META``
        and one ``LabeledBar`` per bar in the input table whose
        horizon window has at least one future bar. Vertical bars
        (no barrier hit) are kept; their ``y_meta`` is always 0.
        The ``y_class`` field preserves the underlying direction
        (``UP=+1``, ``DOWN=-1``, ``FLAT=0``) for downstream
        inspection.
    """
    if spec.kind is not LabelKind.META:
        raise ValueError(f"spec.kind must be META, got {spec.kind}")
    if pt_pct <= 0 or sl_pct <= 0:
        raise ValueError("pt_pct and sl_pct must be > 0")
    rt_bps = cost_model.round_trip_bps
    if rt_bps < 0:
        raise ValueError(f"cost_model.round_trip_bps must be >= 0, got {rt_bps}")

    # Run the existing triple-barrier with require_finite=False so
    # vertical bars are kept. The returned frame is the per-bar
    # primary label; the metalabel is a derived view of it.
    tb_spec = LabelSpec(
        kind=LabelKind.TRIPLE_BARRIER,
        horizon=spec.horizon,
        params=dict(spec.params),
    )
    tb_frame = make_triple_barrier_labels(
        table,
        spec=tb_spec,
        symbol=symbol,
        pt_pct=pt_pct,
        sl_pct=sl_pct,
        require_finite=False,
    )

    bars: list[LabeledBar] = []
    for tb_bar in tb_frame.bars:
        first_hit_raw = tb_bar.meta.get("first_hit", "vertical")
        first_hit = str(first_hit_raw)
        realized_return_bps = float(tb_bar.meta.get("realized_return_bps", 0.0))
        y_meta = 1 if (first_hit != "vertical" and abs(realized_return_bps) > rt_bps) else 0
        # Preserve the underlying direction class for downstream
        # inspection: +1 (upper hit), -1 (lower hit), 0 (vertical).
        if first_hit == "upper":
            y_class = int(DirectionClass.UP)
        elif first_hit == "lower":
            y_class = int(DirectionClass.DOWN)
        else:
            y_class = int(DirectionClass.FLAT)
        y_meta_int: int = int(y_meta)
        bars.append(
            LabeledBar(
                symbol=symbol,
                ts=tb_bar.ts,
                horizon=spec.horizon,
                kind=LabelKind.META,
                y=y_meta_int,
                y_class=y_class,
                meta={
                    # Traceability back to the primary label.
                    "triple_barrier_y": int(tb_bar.y),
                    "triple_barrier_first_hit": first_hit,
                    "realized_return_bps": realized_return_bps,
                    "round_trip_bps": rt_bps,
                },
            )
        )
    return LabeledFrame(spec=spec, symbol=symbol, bars=tuple(bars))


# ---------------------------------------------------------------------------
# W3.7: cost-ML re-work loop trigger
# ---------------------------------------------------------------------------
def should_redo_metalabels(
    placeholder_eta: float,
    calibrated_eta: float,
    *,
    drift_threshold: float = 2.0,
) -> bool:
    """Decide whether a calibrated ``eta`` has drifted enough to re-run the meta-labels.

    The W1.3 placeholder ``AlmgrenChrissModel.eta = 0.5`` is the
    conservative default; the W2 calibration pass overwrites it with
    a measured value. If the measured value differs from the
    placeholder by more than ``drift_threshold`` in *either*
    direction (i.e. the ratio exceeds the threshold), the
    downstream meta-labels were computed under a wrong cost
    assumption and must be re-generated.

    The check is symmetric and direction-agnostic::

        |placeholder_eta / calibrated_eta| > drift_threshold, OR
        |calibrated_eta / placeholder_eta| > drift_threshold

    Equivalently: ``min(placeholder, calibrated) > 0`` and
    ``max(placeholder, calibrated) / min(placeholder, calibrated) >
    drift_threshold``. The default threshold of 2.0 matches the
    plan's W3.7 spec.

    Parameters
    ----------
    placeholder_eta:
        The W1.3 placeholder impact coefficient (default 0.5).
    calibrated_eta:
        The W2-calibrated impact coefficient recovered from real
        (or synthetic) public-trade prints.
    drift_threshold:
        The maximum allowed ratio between the two etas. Strictly
        positive. Default 2.0 (i.e. re-run if either eta is more
        than 2x the other).

    Returns
    -------
    bool
        ``True`` iff a re-run of :func:`make_metalabel_labels` is
        triggered by the drift.

    Raises
    ------
    ValueError
        On non-finite, non-positive, or threshold-violating input.
    """
    import math

    if not math.isfinite(placeholder_eta) or placeholder_eta <= 0:
        raise ValueError(
            f"placeholder_eta must be a positive real number, got {placeholder_eta!r}"
        )
    if not math.isfinite(calibrated_eta) or calibrated_eta <= 0:
        raise ValueError(
            f"calibrated_eta must be a positive real number, got {calibrated_eta!r}"
        )
    if not math.isfinite(drift_threshold) or drift_threshold <= 0:
        raise ValueError(
            f"drift_threshold must be a positive real number, got {drift_threshold!r}"
        )

    high = max(placeholder_eta, calibrated_eta)
    low = min(placeholder_eta, calibrated_eta)
    ratio = high / low
    return bool(ratio > drift_threshold)


__all__ = ["make_metalabel_labels", "should_redo_metalabels"]
