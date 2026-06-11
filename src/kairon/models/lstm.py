"""LSTM baseline (PyTorch).

This is a small sequence-to-one classifier: the input is a sliding
window of length ``sequence_length`` over the feature matrix, the output
is a class probability at the final timestep. We deliberately keep the
architecture simple and only enable this backend when ``torch`` is
installed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix


@dataclass(frozen=True, slots=True)
class LSTMConfig:
    sequence_length: int = 32
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = False
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 3
    device: str = "cpu"
    random_state: int = 42
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {self.sequence_length}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None  # type: ignore[attr-defined]


def _make_sequences(
    values: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    if n <= seq_len:
        # Edge case: not enough data for a single sequence — return the
        # entire matrix as one (padded-by-truncation) sequence.
        x = np.zeros((1, seq_len, values.shape[1]), dtype=np.float32)
        x[0, :n] = values
        return x, y[-1:] if y.size else np.zeros(0, dtype=np.int64)
    n_seq = n - seq_len
    x = np.zeros((n_seq, seq_len, values.shape[1]), dtype=np.float32)
    for i in range(n_seq):
        x[i] = values[i : i + seq_len]
    return x, y[seq_len:]


def _make_predict_sequences(
    values: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    n = values.shape[0]
    if n <= seq_len:
        x = np.zeros((1, seq_len, values.shape[1]), dtype=np.float32)
        x[0, :n] = values
        return x
    n_seq = n - seq_len
    x = np.zeros((n_seq, seq_len, values.shape[1]), dtype=np.float32)
    for i in range(n_seq):
        x[i] = values[i : i + seq_len]
    return x


class LSTMModel(Model[LSTMConfig]):
    """A small LSTM sequence classifier."""

    name = "lstm"
    kind = "deep"

    def __init__(self, config: LSTMConfig | None = None) -> None:
        super().__init__(config or LSTMConfig())
        if not _has_torch():
            raise ModelError(
                "torch is not installed; install with `uv sync --extra ml`"
            )

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        optim = importlib.import_module("torch.optim")

        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)

        x_np, y_np = _make_sequences(features.values, y, self.config.sequence_length)
        classes = np.unique(y_np)
        n_classes = int(classes.size)
        if n_classes < 2:
            raise ModelError(f"LSTM needs >= 2 classes, got {n_classes}")

        # Map arbitrary class labels to contiguous 0..n_classes-1.
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[int(v)] for v in y_np], dtype=np.int64)

        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_idx)

        device = torch.device(self.config.device)
        model = _build_lstm(
            n_features=features.n_features,
            n_classes=n_classes,
            hidden=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
        ).to(device)
        # The W5.1 contract: loss_fn is the *name* of the loss
        # family (string). For W5.1 the v1 backends continue to
        # use cross-entropy; the W5.2 / W5.3 release will branch
        # on loss_fn here. Renamed to criterion to avoid
        # shadowing the str-typed kwarg.
        _ = loss_fn  # advisory: the W5.1 backend treats it as metadata
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        n = x_t.shape[0]
        best_loss = float("inf")
        best_state: dict[str, Any] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        patience_left = self.config.patience
        history: list[float] = []
        for _epoch in range(self.config.epochs):
            model.train()
            perm = torch.randperm(n)
            epoch_losses: list[float] = []
            for start in range(0, n, self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                xb = x_t[idx].to(device)
                yb = y_t[idx].to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.item()))
            mean_loss = float(np.mean(epoch_losses))
            history.append(mean_loss)
            if mean_loss < best_loss - 1e-4:
                best_loss = mean_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        model.load_state_dict(best_state)
        state = {
            "model": model.eval().cpu(),
            "classes": tuple(int(c) for c in classes),
            "n_features": features.n_features,
        }
        return state, {"train_loss": best_loss, "epochs_run": float(len(history))}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        torch = importlib.import_module("torch")
        state = trained
        model = state["model"]
        classes: tuple[int, ...] = state["classes"]

        x_np = _make_predict_sequences(features.values, self.config.sequence_length)
        x_t = torch.from_numpy(x_np)
        with torch.no_grad():
            logits = model(x_t)
            proba_all = torch.softmax(logits, dim=-1).numpy()
        y_idx = proba_all.argmax(axis=1).astype(np.int64)
        y_class = np.array([classes[int(i)] for i in y_idx], dtype=np.int64)
        if proba_all.shape[1] == 2:
            y_proba = proba_all[:, 1]
        else:
            y_proba = proba_all
        return y_class, y_proba, None


def _build_lstm(
    *,
    n_features: int,
    n_classes: int,
    hidden: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
) -> Any:
    nn = importlib.import_module("torch.nn")

    class _LSTMNet(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_dim = hidden * (2 if bidirectional else 1)
            self.head = nn.Sequential(
                nn.Linear(out_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last)

    return _LSTMNet()
