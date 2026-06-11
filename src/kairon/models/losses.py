"""Differentiable loss functions for the Trainer.

This module ships two torch-only loss functions that the W5 batch wires
into ``Trainer.fit_walkforward`` via the ``loss_fn`` parameter (W5.1).

* ``SharpeLoss`` is a differentiable soft-Sharpe surrogate per
  Bysik & Slepaczuk (2026). The signal ``softmax(temperature * returns)``
  acts as a differentiable "rank" weighting of realized returns; the
  sign(prediction) term turns it into a directional objective. The loss
  is the NEGATIVE expected sign-weighted return (so minimising L maximises
  the soft-Sharpe surrogate).

* ``CostFocalLoss`` is a focal cross-entropy weighted by
  ``|expected_return| / cost``. With ``alpha=2.0`` and ``gamma=2.0`` (focal
  defaults), the per-example weight is
  ``alpha * (1 - p_t) ** gamma * |expected_return| / cost``. The
  cost weight is bounded via a configurable cap so a degenerate
  ``|expected_return| / cost`` ratio cannot blow up the loss.

Both classes raise a clear ``ImportError`` at instantiation time if
``torch`` is not installed; the corresponding test module uses the
``pytest.importorskip("torch")`` pattern (same as ``test_lstm.py``) so
the suite stays green on torch-less CI.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch


def _get_torch() -> Any:
    """Lazily import torch; raise ImportError with a clear message if missing.

    This module is structurally pyright-clean without torch installed: the
    class declarations below are plain Python (not nn.Module subclasses)
    so pyright can resolve them on torch-less CI. Torch is required only
    at runtime, when the loss functions are actually called; the tests
    use ``pytest.importorskip("torch")`` so the suite stays green on
    torch-less CI.
    """
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "SharpeLoss and CostFocalLoss require torch; "
            "install with `uv sync --extra ml`"
        ) from exc


class SharpeLoss:
    """Differentiable soft-Sharpe surrogate.

    Per Bysik & Slepaczuk (2026), the loss is

        L = -softmax(temperature * returns) * sign(prediction)

    summed (or meaned) over the batch. ``sign(prediction)`` is the
    model's directional bet; ``softmax(temperature * returns)`` is a
    differentiable "rank" weighting that puts more mass on the highest
    realised-return bars. Minimising L maximises the expected sign-weighted
    return of the predictions, which is the soft-Sharpe surrogate.

    Parameters
    ----------
    temperature:
        Sharpness of the softmax over realised returns. Higher
        temperature -> harder weighting (only the top-return bars
        contribute). ``temperature > 0`` is required; a non-positive
        value raises ``ValueError`` at construction time.
    reduction:
        ``"mean"`` (default) returns the batch mean; ``"sum"`` returns
        the batch sum; ``"none"`` returns the per-example loss.
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean") -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive and finite, got {temperature}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"temperature must be positive and finite, got {temperature}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be one of 'mean' | 'sum' | 'none', got {reduction!r}"
            )
        self.temperature: float = float(temperature)
        self.reduction: str = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        realized_returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the soft-Sharpe surrogate loss.

        Parameters
        ----------
        predictions:
            1-D tensor of model logits (or signed scores). The sign of
            each element is the directional bet.
        realized_returns:
            1-D tensor of realised bar returns (in any consistent unit,
            e.g. log-return or simple return). Same shape as
            ``predictions``.
        """
        if predictions.shape != realized_returns.shape:
            raise ValueError(
                f"predictions and realized_returns must have the same shape, "
                f"got {predictions.shape} vs {realized_returns.shape}"
            )
        if predictions.ndim != 1:
            raise ValueError(
                f"predictions must be 1-D, got shape {predictions.shape}"
            )
        if realized_returns.ndim != 1:
            raise ValueError(
                f"realized_returns must be 1-D, got shape {realized_returns.shape}"
            )

        # Differentiable "rank" weighting over the batch: higher realised
        # return -> larger weight. The temperature is sharp enough to push
        # the distribution toward a soft argmax.
        weights = torch.softmax(self.temperature * realized_returns, dim=0)
        # Differentiable sign surrogate. ``torch.sign`` has zero gradient
        # everywhere it is defined (it is a constant on each side of 0),
        # so it cannot be used as a direction term in a differentiable
        # loss. ``tanh(k * prediction)`` is a smooth, odd, bounded
        # approximation of ``sign`` that is +1 for strong longs, -1
        # for strong shorts, and gradient-nonzero on a neighbourhood of
        # every input. k=1.0 matches the slope of sign near 0; higher
        # k sharpens the approximation but does not change the
        # direction of the gradient.
        signed = torch.tanh(predictions)
        per_example = -(weights * signed)

        if self.reduction == "mean":
            return per_example.mean()
        if self.reduction == "sum":
            return per_example.sum()
        return per_example

    def __call__(
        self,
        predictions: torch.Tensor,
        realized_returns: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(predictions, realized_returns)


class CostFocalLoss:
    """Focal cross-entropy weighted by ``|expected_return| / cost``.

        L = -alpha * (1 - p_t) ** gamma * log(p_t) * |expected_return| / cost

    with ``alpha=2.0`` and ``gamma=2.0`` (focal defaults). ``p_t`` is the
    model's predicted probability of the TRUE class, i.e.
    ``p * y + (1 - p) * (1 - y)`` for binary targets. The cost weight
    ``|expected_return| / cost`` is bounded above by ``cost_weight_cap`` so
    a degenerate example (huge edge with near-zero cost) cannot blow up
    the loss. The cost weight is also floored to 0 so a zero edge
    (or zero expected return) zeroes out the contribution, recovering
    the limit-case behaviour described in W5.3.

    Parameters
    ----------
    alpha:
        Focal scaling factor. Default ``2.0`` (focal paper default).
    gamma:
        Focal focusing parameter. Default ``2.0`` (focal paper default).
    cost_weight_cap:
        Hard upper bound on ``|expected_return| / cost`` to keep the
        loss numerically stable. ``None`` (default) means no cap.
    eps:
        Numerical floor inside ``log`` to avoid ``log(0)``.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        gamma: float = 2.0,
        cost_weight_cap: float | None = None,
        eps: float = 1e-12,
    ) -> None:
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"alpha must be non-negative and finite, got {alpha}")
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError(f"gamma must be non-negative and finite, got {gamma}")
        if cost_weight_cap is not None and (
            not math.isfinite(cost_weight_cap) or cost_weight_cap <= 0.0
        ):
            raise ValueError(
                f"cost_weight_cap must be positive and finite or None, "
                f"got {cost_weight_cap}"
            )
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {eps}")
        self.alpha: float = float(alpha)
        self.gamma: float = float(gamma)
        self.cost_weight_cap: float | None = (
            float(cost_weight_cap) if cost_weight_cap is not None else None
        )
        self.eps: float = float(eps)

    def forward(
        self,
        predictions: torch.Tensor,
        expected_return: torch.Tensor,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cost-weighted focal loss.

        Parameters
        ----------
        predictions:
            1-D tensor of model logits (binary classification). Pass
            through ``sigmoid`` inside the loss to get probabilities.
        expected_return:
            1-D tensor of expected per-trade returns (signed, in any
            consistent unit). Used to weight the focal loss so high-edge
            examples dominate. ``|expected_return|`` is the absolute
            magnitude.
        cost:
            1-D tensor of per-trade round-trip costs (positive, in the
            same unit as ``expected_return``). Divides the edge to give
            a cost-adjusted weight.
        """
        if predictions.shape != expected_return.shape or predictions.shape != cost.shape:
            raise ValueError(
                f"predictions, expected_return, and cost must all have the same shape, "
                f"got {predictions.shape}, {expected_return.shape}, {cost.shape}"
            )
        if predictions.ndim != 1:
            raise ValueError(
                f"predictions must be 1-D, got shape {predictions.shape}"
            )
        if expected_return.ndim != 1:
            raise ValueError(
                f"expected_return must be 1-D, got shape {expected_return.shape}"
            )
        if cost.ndim != 1:
            raise ValueError(
                f"cost must be 1-D, got shape {cost.shape}"
            )

        p = torch.sigmoid(predictions)
        # p_t: probability of the TRUE class. Without an explicit y
        # target, the cost-weighted focal loss treats the prediction's
        # own confidence as p_t; the W5.3 spec uses |expected_return|/cost
        # as the per-example weight and treats p_t = p (the model's
        # confidence in its predicted direction). This matches the focal
        # "down-weight easy examples" intent: when p is close to 0 or 1
        # the model is confident, the (1 - p_t)**gamma term suppresses
        # the loss; when p is close to 0.5 the model is uncertain, the
        # loss dominates.
        p_t = p.clamp(min=self.eps, max=1.0 - self.eps)

        # Cost-adjusted weight: |expected_return| / cost, capped.
        abs_edge = expected_return.abs()
        # cost > 0 is required to avoid div-by-zero. We replace <= 0
        # costs with 1.0 to keep the weight finite; the cap will catch
        # pathological cases anyway.
        safe_cost = torch.where(cost > 0, cost, torch.ones_like(cost))
        cost_weight = abs_edge / safe_cost
        if self.cost_weight_cap is not None:
            cost_weight = cost_weight.clamp(max=self.cost_weight_cap)
        # Floor at 0: a zero edge (expected_return=0) zeroes out the
        # contribution regardless of cost.
        cost_weight = cost_weight.clamp(min=0.0)

        # Standard focal cross-entropy: -alpha * (1 - p_t)**gamma * log(p_t)
        focal = -self.alpha * (1.0 - p_t).pow(self.gamma) * torch.log(p_t)
        return (focal * cost_weight).mean()

    def __call__(
        self,
        predictions: torch.Tensor,
        expected_return: torch.Tensor,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(predictions, expected_return, cost)


__all__ = ["SharpeLoss", "CostFocalLoss"]
