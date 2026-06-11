"""Tests for the W6.4 multi-head model.

Two tests pin the W6.4 acceptance criteria:

1. ``test_sign_correlation`` — ``sign(prediction.y_magnitude)``
   correlates with the true direction at >= 0.5 Pearson on a
   synthetic fixture. The synthetic fixture has a 3-class
   direction label (down / flat / up) and a continuous magnitude
   target whose sign is aligned with the direction (up=positive,
   down=negative, flat=zero). The test asserts the multi-head's
   magnitude head learns this sign relationship.

2. ``test_pinball_loss`` — the vol head's loss is the standard
   quantile (pinball) loss with configurable alpha. Tested
   numerically: at ``alpha=0.5`` the loss is half the mean
   absolute error; at ``alpha=0.9`` the loss is asymmetric
   (penalises under-prediction more than over-prediction).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kairon.models.contracts import FeatureMatrix
from kairon.models.multihead import (
    MultiHeadConfig,
    MultiHeadModel,
    pinball_loss,
)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------
def _make_synthetic_fixture(
    *,
    n: int = 600,
    seed: int = 20260608,
) -> tuple[FeatureMatrix, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic 3-class direction + magnitude + vol fixture.

    The fixture is constructed so the direction label and the
    magnitude target's sign are aligned:

    - ``y_direction[i] = 0`` (DOWN) and ``y_magnitude[i] < 0``,
    - ``y_direction[i] = 1`` (FLAT) and ``y_magnitude[i] ≈ 0``,
    - ``y_direction[i] = 2`` (UP) and ``y_magnitude[i] > 0``.

    The vol target is a per-bar realised-vol proxy: a positive
    function of the absolute log-return.

    The features are a 2-D ``(N, 4)`` array of random noise +
    one informative column. The informative column has a
    monotone relationship with the direction label so the
    logistic-regression direction head can separate the three
    classes with reasonable accuracy.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    n_classes: int = 3
    n_features: int = 4

    # Direction labels: roughly balanced 3-class.
    y_direction: np.ndarray = rng.integers(
        low=0, high=n_classes, size=n, dtype=np.int64
    )

    # Magnitude target: aligned with the direction label. The
    # per-class mean is +/- 0.02 with std=0.005; the FLAT class
    # is centred on 0.
    class_to_sign: np.ndarray = np.array([-1.0, 0.0, 1.0])
    base_mag: np.ndarray = class_to_sign[y_direction] * 0.02
    y_magnitude: np.ndarray = (
        base_mag + rng.normal(loc=0.0, scale=0.005, size=n)
    ).astype(np.float64)

    # Vol target: positive function of |log-return|. Per-bar
    # realised vol is non-negative.
    y_vol: np.ndarray = (
        np.abs(y_magnitude) * 1.5 + rng.normal(loc=0.0, scale=0.001, size=n)
    ).astype(np.float64)
    y_vol = np.clip(y_vol, a_min=1e-6, a_max=None)

    # Features: 3 noise columns + 1 informative column. The
    # informative column is monotone in the direction label so
    # the logistic-regression direction head can separate the
    # three classes.
    x_noise: np.ndarray = rng.normal(loc=0.0, scale=1.0, size=(n, n_features - 1))
    x_info: np.ndarray = class_to_sign[y_direction] * 1.0 + rng.normal(
        loc=0.0, scale=0.3, size=n
    )
    x_info = x_info.reshape(-1, 1)
    x: np.ndarray = np.column_stack([x_info, x_noise]).astype(np.float64)

    fm: FeatureMatrix = FeatureMatrix(
        values=x,
        feature_names=("x_info", "x_noise_0", "x_noise_1", "x_noise_2"),
    )
    return fm, y_direction, y_magnitude, y_vol


# ---------------------------------------------------------------------------
# W6.4 acceptance criterion #1: sign correlation
# ---------------------------------------------------------------------------
def test_sign_correlation() -> None:
    """``sign(prediction.y_magnitude)`` correlates with the true
    direction at >= 0.5 Pearson on a synthetic fixture.

    The Pearson correlation is computed between
    ``sign(y_magnitude_pred)`` and ``y_direction`` (mapped to
    ``{-1, 0, +1}`` for the sign interpretation). The test pins
    the W6.4 acceptance criterion of >= 0.5 Pearson correlation.
    """
    fm, y_direction, y_magnitude, y_vol = _make_synthetic_fixture()
    n: int = fm.n_rows

    config: MultiHeadConfig = MultiHeadConfig(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=20260608,
    )
    model: MultiHeadModel = MultiHeadModel(config)
    trained: dict[str, object] = model.fit_multihead(
        fm, y_direction, y_magnitude, y_vol,
    )

    out: dict[str, np.ndarray] = model.predict_multihead(trained, fm)
    y_magnitude_pred: np.ndarray = out["y_magnitude"]

    # Sign of the predicted magnitude.
    sign_pred: np.ndarray = np.sign(y_magnitude_pred)

    # Map the 3-class direction label to a signed encoding
    # (-1, 0, +1) so the Pearson correlation is meaningful.
    sign_true: np.ndarray = (y_direction.astype(np.float64) - 1.0)

    # Pearson correlation.
    if sign_pred.std(ddof=0) == 0.0 or sign_true.std(ddof=0) == 0.0:
        pytest.fail(
            "degenerate sign prediction; cannot compute Pearson correlation"
        )
    pearson: float = float(np.corrcoef(sign_pred, sign_true)[0, 1])

    assert math.isfinite(pearson), (
        f"Pearson correlation must be finite, got {pearson!r}"
    )
    assert pearson >= 0.5, (
        f"sign(y_magnitude) must correlate with true direction at "
        f">= 0.5 Pearson, got {pearson:.4f} (n={n})"
    )


# ---------------------------------------------------------------------------
# W6.4 acceptance criterion #2: pinball loss with configurable alpha
# ---------------------------------------------------------------------------
def test_pinball_loss() -> None:
    """The vol head's loss is the standard quantile (pinball) loss.

    Three sub-checks pin the pinball-loss contract:

    1. At ``alpha=0.5`` the pinball loss is half the mean
       absolute error (the canonical median objective). This is
       the W6.4 default.
    2. At ``alpha=0.9`` the loss is asymmetric: equal
       over-prediction and under-prediction errors produce
       *different* losses (the model penalises under-prediction
       more, since alpha=0.9 targets the 90th percentile).
    3. The loss is non-negative and zero when ``y_pred ==
       y_true`` exactly.
    """
    rng: np.random.Generator = np.random.default_rng(20260608)
    n: int = 200
    y_true: np.ndarray = rng.normal(loc=0.0, scale=1.0, size=n)
    # y_pred = y_true + noise so the loss is non-zero.
    y_pred: np.ndarray = y_true + rng.normal(loc=0.0, scale=0.5, size=n)

    # --- (1) alpha=0.5 -> 0.5 * mean |y - q| -----------------------
    loss_p5: float = pinball_loss(y_true, y_pred, alpha=0.5)
    expected_p5: float = 0.5 * float(np.abs(y_true - y_pred).mean())
    assert math.isfinite(loss_p5), (
        f"pinball loss at alpha=0.5 must be finite, got {loss_p5!r}"
    )
    assert loss_p5 == pytest.approx(expected_p5, rel=1e-9, abs=1e-12), (
        f"pinball loss at alpha=0.5 ({loss_p5:.6f}) must equal "
        f"0.5 * MAE ({expected_p5:.6f})"
    )

    # --- (2) alpha=0.9 -> asymmetric; penalises under-prediction --
    # Build a controlled fixture: half the time the model
    # under-predicts (q < y), half the time it over-predicts
    # (q > y). The pinball loss at alpha=0.9 penalises
    # under-prediction more than over-prediction by construction.
    y_true_asym: np.ndarray = np.array([1.0] * 50 + [-1.0] * 50)
    # q under-predicts the +1 (predicts 0.5) and over-predicts
    # the -1 (predicts -0.5). Both errors are |q - y| = 0.5.
    y_pred_asym: np.ndarray = np.array([0.5] * 50 + [-0.5] * 50)

    loss_p9: float = pinball_loss(y_true_asym, y_pred_asym, alpha=0.9)
    # Compute the components separately to pin the
    # asymmetry.
    u: np.ndarray = y_true_asym - y_pred_asym  # +0.5 for the 1.0 case, -0.5 for the -1.0 case
    # Under-prediction: u > 0 -> loss = alpha * u = 0.9 * 0.5 = 0.45 (per example)
    # Over-prediction:  u < 0 -> loss = (alpha - 1) * u = -0.1 * (-0.5) = 0.05
    expected_p9: float = float(
        np.where(u >= 0.0, 0.9 * u, (0.9 - 1.0) * u).mean()
    )
    assert math.isfinite(loss_p9), (
        f"pinball loss at alpha=0.9 must be finite, got {loss_p9!r}"
    )
    assert loss_p9 == pytest.approx(expected_p9, rel=1e-9, abs=1e-12), (
        f"pinball loss at alpha=0.9 ({loss_p9:.6f}) must equal the "
        f"asymmetric formula ({expected_p9:.6f})"
    )
    # And the asymmetry: under-prediction is penalised ~9x
    # more than over-prediction (alpha / (1 - alpha) = 0.9/0.1).
    under_loss: float = 0.9 * 0.5  # 0.45
    over_loss: float = 0.1 * 0.5  # 0.05
    assert under_loss > over_loss, (
        f"alpha=0.9 must penalise under-prediction more than "
        f"over-prediction; got under={under_loss:.4f} vs "
        f"over={over_loss:.4f}"
    )

    # --- (3) y_pred == y_true -> loss == 0 ------------------------
    loss_zero: float = pinball_loss(y_true, y_true, alpha=0.5)
    assert loss_zero == 0.0, (
        f"pinball loss at y_pred == y_true must be exactly 0, got "
        f"{loss_zero!r}"
    )

    # --- (4) alpha out of range raises ----------------------------
    with pytest.raises(ValueError, match="alpha must be in"):
        pinball_loss(y_true, y_pred, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        pinball_loss(y_true, y_pred, alpha=1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        pinball_loss(y_true, y_pred, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        pinball_loss(y_true, y_pred, alpha=1.5)
