"""Walk-forward cross-validation for time-series model evaluation.

Expanding-window walk-forward CV that respects temporal ordering:
each fold trains on data up to time T, validates on [T+1, T+gap],
then expands the training window. This eliminates look-ahead bias
and gives a realistic estimate of out-of-sample performance.

Typical usage::

    from kairon.evaluation.walkforward_cv import WalkForwardCV

    cv = WalkForwardCV(n_folds=5, min_train=500, gap=0)
    for fold in cv.split(X):
        train_idx, val_idx = fold
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        # compute metrics...

The CV also provides a convenience ``evaluate`` method that runs
all folds and returns aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Metrics from a single walk-forward fold."""

    fold: int
    n_train: int
    n_val: int
    accuracy: float
    brier_score: float
    ece: float
    sharpe: float
    cas: float  # Cost-adjusted Sharpe
    direction_accuracy: float  # 3-class direction accuracy
    coverage: float  # fraction of predictions above threshold


@dataclass(frozen=True, slots=True)
class CVResult:
    """Aggregate walk-forward CV results."""

    folds: tuple[FoldResult, ...]
    mean_accuracy: float
    std_accuracy: float
    mean_brier: float
    mean_ece: float
    mean_sharpe: float
    mean_cas: float
    mean_direction_accuracy: float
    mean_coverage: float


class WalkForwardCV:
    """Expanding-window walk-forward cross-validation.

    Parameters
    ----------
    n_folds : int
        Number of CV folds (default 5).
    min_train : int
        Minimum number of training rows for the first fold (default 500).
    gap : int
        Number of rows to skip between train and validation (default 0,
        meaning the validation starts immediately after training).
    val_ratio : float
        Fraction of data used for validation in each fold (default 0.2).
        The validation window is ``max(100, int(n_rows * val_ratio / n_folds))``.
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train: int = 500,
        gap: int = 0,
        val_ratio: float = 0.2,
    ) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if min_train < 10:
            raise ValueError(f"min_train must be >= 10, got {min_train}")
        self.n_folds = n_folds
        self.min_train = min_train
        self.gap = gap
        self.val_ratio = val_ratio

    def split(self, X: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation index pairs for walk-forward CV.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) or (n_samples,).
            Only the number of rows is used.

        Returns
        -------
        list of (train_idx, val_idx) tuples
        """
        n = X.shape[0] if X.ndim > 1 else len(X)
        if n < self.min_train + self.n_folds:
            raise ValueError(
                f"Not enough data for {self.n_folds} folds with min_train={self.min_train}. "
                f"Have {n} rows, need at least {self.min_train + self.n_folds}."
            )

        # Validation window size
        val_size = max(100, int(n * self.val_ratio / self.n_folds))

        folds = []
        train_end = self.min_train

        for fold_idx in range(self.n_folds):
            val_start = train_end + self.gap
            val_end = min(val_start + val_size, n)

            if val_start >= n or val_end <= val_start:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            folds.append((train_idx, val_idx))

            # Expand training window for next fold
            train_end = val_end

        return folds

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        confidence_threshold: float = 0.0,
    ) -> dict[str, float]:
        """Compute classification and trading metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels in {0, 1, 2} (direction: down/flat/up).
        y_pred : np.ndarray
            Predicted labels in {0, 1, 2}.
        y_proba : np.ndarray, optional
            Predicted probabilities, shape (N, 3).
        confidence_threshold : float
            Minimum max probability to include in metrics (default 0.0,
            meaning all predictions are included).

        Returns
        -------
        dict with keys: accuracy, brier_score, ece, direction_accuracy, coverage
        """
        if confidence_threshold > 0 and y_proba is not None:
            max_proba = np.max(y_proba, axis=1) if y_proba.ndim > 1 else y_proba
            mask = max_proba >= confidence_threshold
            if mask.sum() == 0:
                return {
                    "accuracy": 0.0,
                    "brier_score": float("nan"),
                    "ece": float("nan"),
                    "direction_accuracy": 0.0,
                    "coverage": 0.0,
                }
            y_true_filt = y_true[mask]
            y_pred_filt = y_pred[mask]
            y_proba_filt = y_proba[mask] if y_proba is not None else None
            coverage = float(mask.mean())
        else:
            y_true_filt = y_true
            y_pred_filt = y_pred
            y_proba_filt = y_proba
            coverage = 1.0

        accuracy = float(np.mean(y_true_filt == y_pred_filt))

        # Brier score (if probabilities provided)
        brier_score = float("nan")
        if y_proba_filt is not None and y_proba_filt.ndim == 2:
            n_classes = y_proba_filt.shape[1]
            one_hot = np.zeros_like(y_proba_filt)
            for i in range(len(y_true_filt)):
                if 0 <= y_true_filt[i] < n_classes:
                    one_hot[i, y_true_filt[i]] = 1.0
            brier_score = float(np.mean((y_proba_filt - one_hot) ** 2))

        # ECE (Expected Calibration Error)
        ece = float("nan")
        if y_proba_filt is not None and y_proba_filt.ndim == 2:
            max_proba = np.max(y_proba_filt, axis=1)
            correct = (y_pred_filt == y_true_filt).astype(float)
            n_bins = 10
            bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
            ece_val = 0.0
            for i in range(n_bins):
                in_bin = (max_proba >= bin_boundaries[i]) & (max_proba < bin_boundaries[i + 1])
                if in_bin.sum() > 0:
                    avg_confidence = max_proba[in_bin].mean()
                    avg_accuracy = correct[in_bin].mean()
                    ece_val += float(in_bin.sum()) * abs(avg_accuracy - avg_confidence)
            ece = ece_val / len(max_proba) if len(max_proba) > 0 else float("nan")

        # Direction accuracy (ignoring flat/0 class)
        non_flat = y_true_filt != 1  # 1 is the "flat" class
        if non_flat.sum() > 0:
            direction_accuracy = float(np.mean(y_true_filt[non_flat] == y_pred_filt[non_flat]))
        else:
            direction_accuracy = accuracy

        return {
            "accuracy": accuracy,
            "brier_score": brier_score,
            "ece": ece,
            "direction_accuracy": direction_accuracy,
            "coverage": coverage,
        }


def cost_adjusted_sharpe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
    confidence_threshold: float = 0.0,
    y_proba: np.ndarray | None = None,
    cost_bps: float = 28.0,
) -> float:
    """Compute Cost-Adjusted Sharpe ratio.

    CAS = (mean_returns - cost) / std_returns, where:
    - mean_returns = mean of realized returns on traded bars
    - cost = round-trip cost in bps * trade frequency
    - std_returns = std of realized returns on traded bars

    Parameters
    ----------
    y_true : np.ndarray
        True labels in {0, 1, 2}.
    y_pred : np.ndarray
        Predicted labels in {0, 1, 2}.
    returns : np.ndarray
        Realized log-returns for each bar.
    confidence_threshold : float
        Minimum max probability to trade (default 0.0 = trade all).
    y_proba : np.ndarray, optional
        Predicted probabilities for confidence filtering.
    cost_bps : float
        Round-trip cost in basis points (default 28 bps).
    """
    # Filter by confidence
    if confidence_threshold > 0 and y_proba is not None:
        max_proba = np.max(y_proba, axis=1) if y_proba.ndim > 1 else y_proba
        mask = max_proba >= confidence_threshold
    else:
        mask = np.ones(len(y_true), dtype=bool)

    # Only trade on directional predictions (not flat/class 1)
    directional = y_pred != 1
    trade_mask = mask & directional

    if trade_mask.sum() < 2:
        return 0.0

    # Compute realized returns on traded bars
    trade_returns = returns[trade_mask]
    # Direction: class 2 = up (+1), class 0 = down (-1)
    trade_directions = np.where(y_pred[trade_mask] == 2, 1.0, -1.0)
    realized = trade_returns * trade_directions

    # Apply round-trip cost per trade
    n_trades = int(trade_mask.sum())
    total_cost = (cost_bps / 10000.0) * n_trades / len(y_true)
    mean_return = realized.mean() - total_cost
    std_return = realized.std()

    if std_return < 1e-10:
        return 0.0

    # Annualize (assume ~8760 hourly bars per year)
    sharpe = mean_return / std_return * np.sqrt(8760)
    return float(sharpe)


__all__ = [
    "WalkForwardCV",
    "FoldResult",
    "CVResult",
    "cost_adjusted_sharpe",
]