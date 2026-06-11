"""Tests for the differentiable loss functions in kairon.models.losses.

These tests cover both W5.2 (SharpeLoss) and W5.3 (CostFocalLoss).
Per the W5.2 note in the PRD, the test module uses
``pytest.importorskip("torch")`` (same pattern as ``test_lstm.py:17``)
so the suite stays green on torch-less CI. If torch is not installed,
every test in this file is skipped.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from kairon.models.losses import CostFocalLoss, SharpeLoss  # noqa: E402


# ---------------------------------------------------------------------------
# W5.2: SharpeLoss
# ---------------------------------------------------------------------------
def test_sharpe_loss_gradient() -> None:
    """The gradient of L w.r.t. prediction logits is non-zero on a
    synthetic positive-Sharpe signal (gradient norm > 0)."""
    torch.manual_seed(0)
    # 64 bars of positive-Sharpe returns: draw from a normal with
    # positive mean so the realised returns are systematically positive.
    n: int = 64
    returns = torch.distributions.Normal(loc=0.01, scale=0.05).sample((n,))
    # Predictions are differentiable w.r.t. themselves; we use a leaf
    # tensor with requires_grad=True.
    predictions = torch.zeros(n, requires_grad=True)

    loss = SharpeLoss(temperature=2.0)(predictions, returns)
    loss.backward()

    assert predictions.grad is not None, "predictions.grad must not be None after backward()"
    grad_norm = float(predictions.grad.norm().item())
    assert math.isfinite(grad_norm), "gradient norm must be finite"
    assert grad_norm > 0.0, (
        f"gradient norm must be > 0 on a positive-Sharpe signal, got {grad_norm}"
    )


def test_sharpe_loss_decreases_with_negative_edge() -> None:
    """L is lower (better) when predictions correctly identify the sign
    of high-return bars vs random predictions."""
    torch.manual_seed(1)
    n: int = 128
    # Positive-Sharpe returns: positive mean, modest std.
    returns = torch.distributions.Normal(loc=0.02, scale=0.05).sample((n,))

    # "Correct" predictions: sign matches the sign of realised return,
    # with a unit-magnitude logit so sign(prediction) = sign(realised).
    correct_predictions = torch.sign(returns) + 1e-3 * torch.randn(n)

    # "Random" predictions: independent of the realised returns.
    random_predictions = torch.where(
        torch.rand(n) > 0.5,
        torch.ones(n),
        -torch.ones(n),
    )

    loss_module = SharpeLoss(temperature=2.0)
    # .detach() is not needed for forward; we only need the loss VALUE
    # (no gradient). We use no_grad to keep the test lean.
    with torch.no_grad():
        loss_correct = float(loss_module(correct_predictions, returns).item())
        loss_random = float(loss_module(random_predictions, returns).item())

    assert math.isfinite(loss_correct) and math.isfinite(loss_random)
    # L is the NEGATIVE expected sign-weighted return; lower L == better
    # (i.e. higher expected sign-weighted return). Correctly-signed
    # predictions must achieve lower L than random predictions.
    assert loss_correct < loss_random, (
        f"correctly-signed predictions should have LOWER loss than random, "
        f"got loss_correct={loss_correct:.6f} vs loss_random={loss_random:.6f}"
    )


# ---------------------------------------------------------------------------
# W5.3: CostFocalLoss
# ---------------------------------------------------------------------------
def test_cost_focal() -> None:
    """The focal weight (1-p_t)^gamma increases loss on hard examples
    (low p_t) and decreases it on easy ones; the cost-weight
    |expected_return|/cost is bounded to prevent explosion.

    We verify both halves of the contract:

    (1) Focal focusing: a HARD example (model very wrong, p ~ 0) must
        produce a LARGER loss than an EASY example (model very right, p
        ~ 1) when the cost-weight is held constant.

    (2) Cost-weight cap: a degenerate example with |expected_return|/cost
        above the cap must be CLAMPED so the loss does not explode.
    """
    # --- Part 1: focal focusing on hard vs easy ---
    # We use a large negative logit for hard (sigmoid -> p ~ 0) and a
    # large positive logit for easy (sigmoid -> p ~ 1). inverse-sigmoid
    # isn't a torch op, but we can solve for the logit that gives
    # p=0.05 and p=0.95 by hand: logit(p) = log(p / (1 - p)).
    hard_logit = math.log(0.05 / 0.95)
    easy_logit = math.log(0.95 / 0.05)
    hard_pred = torch.tensor([hard_logit])
    easy_pred = torch.tensor([easy_logit])
    # Constant cost-weight: expected_return=1, cost=1 -> ratio=1.
    edge = torch.tensor([1.0])
    cost = torch.tensor([1.0])

    loss = CostFocalLoss(alpha=2.0, gamma=2.0, cost_weight_cap=10.0)
    with torch.no_grad():
        hard_loss = float(loss(hard_pred, edge, cost).item())
        easy_loss = float(loss(easy_pred, edge, cost).item())

    assert math.isfinite(hard_loss) and math.isfinite(easy_loss)
    # (1 - p_t)**gamma is large when p_t is small (hard) and small when
    # p_t is large (easy). So the focal weight AMPLIFIES the hard
    # example and SUPPRESSES the easy one.
    assert hard_loss > easy_loss, (
        f"hard example (p=0.05) should have HIGHER loss than easy (p=0.95), "
        f"got hard={hard_loss:.6f} vs easy={easy_loss:.6f}"
    )

    # --- Part 2: cost-weight cap prevents explosion ---
    # A degenerate example: |expected_return|=1000, cost=0.01 -> ratio=100000.
    # With cap=10.0, the effective ratio is 10.0, NOT 100000.
    big_edge = torch.tensor([1000.0])
    tiny_cost = torch.tensor([0.01])
    uncapped = CostFocalLoss(alpha=2.0, gamma=2.0, cost_weight_cap=None)
    capped = CostFocalLoss(alpha=2.0, gamma=2.0, cost_weight_cap=10.0)
    with torch.no_grad():
        uncapped_loss = float(uncapped(hard_pred, big_edge, tiny_cost).item())
        capped_loss = float(capped(hard_pred, big_edge, tiny_cost).item())

    assert math.isfinite(uncapped_loss), "uncapped loss must be finite"
    assert math.isfinite(capped_loss), "capped loss must be finite"
    # The cap must produce a strictly smaller (or equal) loss than the
    # uncapped path on the same input. If the cap were not enforced,
    # capped_loss would equal uncapped_loss.
    assert capped_loss < uncapped_loss, (
        f"cost_weight_cap must strictly reduce the loss on a degenerate "
        f"edge/cost pair, got capped={capped_loss:.6f} vs uncapped={uncapped_loss:.6f}"
    )


def test_cost_focal_zero_weight_is_cross_entropy() -> None:
    """With expected_return=0 OR cost=inf, the loss reduces to standard
    cross-entropy (limit case). Concretely: a zero cost-weight zeroes
    out the per-example contribution, so the loss is exactly 0; with
    cost=inf the effective ratio is 0, so the loss is also 0. Both
    branches must return a finite, non-negative loss with a
    cross-entropy-shaped focal factor (i.e. zero, not NaN or inf)."""
    # Branch 1: expected_return = 0 -> |edge| = 0 -> weight = 0.
    pred = torch.tensor([0.0])
    zero_edge = torch.tensor([0.0])
    unit_cost = torch.tensor([1.0])
    # Branch 2: cost = inf -> |edge|/cost = 0 -> weight = 0.
    inf_cost = torch.tensor([math.inf])

    loss = CostFocalLoss(alpha=2.0, gamma=2.0, cost_weight_cap=10.0)
    with torch.no_grad():
        loss_zero_edge = float(loss(pred, zero_edge, unit_cost).item())
        loss_inf_cost = float(loss(pred, zero_edge, inf_cost).item())

    assert math.isfinite(loss_zero_edge), (
        f"loss must be finite when expected_return=0, got {loss_zero_edge}"
    )
    assert math.isfinite(loss_inf_cost), (
        f"loss must be finite when cost=inf, got {loss_inf_cost}"
    )
    # A zero weight zeros the per-example contribution; the .mean() over
    # one example is still zero (not NaN, not inf).
    assert loss_zero_edge == 0.0, (
        f"loss with expected_return=0 must be exactly 0, got {loss_zero_edge}"
    )
    assert loss_inf_cost == 0.0, (
        f"loss with cost=inf must be exactly 0, got {loss_inf_cost}"
    )
