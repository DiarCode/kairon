"""Deep ensemble: architecture-diverse stack of deep nets.

The ensemble combines up to three deep backends (MLP, LSTM, N-BEATS)
plus the always-available tabular backends (LogReg, RandomForest) into
a single :class:`Model` whose predictions go through the Top-K
confidence combinator.

If ``torch`` is not installed, only the non-torch constituents are
included. The ensemble degrades gracefully.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairon.models.base import Model, ModelError, TrainedModel
from kairon.models.contracts import FeatureMatrix
from kairon.models.ensemble import EnsembleSpec, TopKConfidenceEnsemble
from kairon.models.linear import LinearConfig, LogisticRegressionModel
from kairon.models.lstm import LSTMConfig, LSTMModel
from kairon.models.nbeats import NBEATSConfig, NBEATSModel
from kairon.models.tree import RandomForestConfig, RandomForestModel


@dataclass(frozen=True, slots=True)
class DeepEnsembleConfig:
    """Configuration for the deep-architecture-diverse ensemble."""

    lookback: int = 32
    include_mlp: bool = True
    include_lstm: bool = True
    include_nbeats: bool = True
    include_rf: bool = True
    include_lr: bool = True
    spec: EnsembleSpec = field(default_factory=EnsembleSpec)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {self.lookback}")


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None  # type: ignore[attr-defined]


def _MLPModel() -> Model[Any]:
    """Build a tiny MLP backend (only when torch is installed)."""
    if not _has_torch():
        raise RuntimeError("torch not installed")
    from dataclasses import dataclass as _dc

    import torch
    from torch import nn, optim

    @_dc(frozen=True, slots=True)
    class MLPConfig:
        hidden_size: int = 64
        n_layers: int = 2
        dropout: float = 0.1
        epochs: int = 20
        batch_size: int = 128
        learning_rate: float = 1e-3
        weight_decay: float = 1e-5
        patience: int = 3
        device: str = "cpu"
        random_state: int = 42

    class _MLPNet(nn.Module):  # type: ignore[misc]
        def __init__(self, n_features: int, n_classes: int, hidden: int, n_layers: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            d_in = n_features
            for _ in range(n_layers):
                layers.append(nn.Linear(d_in, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                d_in = hidden
            layers.append(nn.Linear(d_in, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class _MLP(Model[MLPConfig]):  # type: ignore[misc, type-arg]
        name = "mlp"
        kind = "deep"

        def __init__(self, config: MLPConfig | None = None) -> None:
            super().__init__(config or MLPConfig())

        def _fit_core(
            self,
            features: FeatureMatrix,
            y: np.ndarray,
            *,
            sample_weight: np.ndarray | None,
            loss_fn: str,
        ) -> tuple[Any, dict[str, float]]:
            np.random.seed(self.config.random_state)
            torch.manual_seed(self.config.random_state)
            x_t = torch.from_numpy(features.values.astype(np.float32))
            classes = np.unique(y)
            n_classes = int(classes.size)
            if n_classes < 2:
                raise ModelError("MLP needs >= 2 classes")
            class_to_idx = {int(c): i for i, c in enumerate(classes)}
            y_idx = torch.from_numpy(
                np.array([class_to_idx[int(v)] for v in y], dtype=np.int64)
            )
            device = torch.device(self.config.device)
            model = _MLPNet(
                n_features=features.n_features,
                n_classes=n_classes,
                hidden=self.config.hidden_size,
                n_layers=self.config.n_layers,
                dropout=self.config.dropout,
            ).to(device)
            opt = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            # The W5.1 contract: loss_fn is the *name* of the loss
            # family (string). For W5.1 the v1 backends continue to
            # use cross-entropy; the W5.2 / W5.3 release will branch
            # on loss_fn here. Renamed to criterion to avoid
            # shadowing the str-typed kwarg.
            _ = loss_fn  # advisory: the W5.1 backend treats it as metadata
            criterion = nn.CrossEntropyLoss()
            n = x_t.shape[0]
            best_loss = float("inf")
            best_state: dict[str, Any] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = self.config.patience
            for _ in range(self.config.epochs):
                model.train()
                perm = torch.randperm(n)
                losses: list[float] = []
                for start in range(0, n, self.config.batch_size):
                    idx = perm[start : start + self.config.batch_size]
                    xb = x_t[idx].to(device)
                    yb = y_idx[idx].to(device)
                    opt.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
                mean_loss = float(np.mean(losses))
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
            return {
                "model": model.eval().cpu(),
                "classes": tuple(int(c) for c in classes),
                "n_features": features.n_features,
            }, {"train_loss": best_loss}

        def _predict_core(
            self,
            trained: Any,
            features: FeatureMatrix,
        ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
            x_t = torch.from_numpy(features.values.astype(np.float32))
            model = trained["model"]
            classes: tuple[int, ...] = trained["classes"]
            with torch.no_grad():
                proba = torch.softmax(model(x_t), dim=-1).numpy()
            y_idx = proba.argmax(axis=1).astype(np.int64)
            y_class = np.array([classes[int(i)] for i in y_idx], dtype=np.int64)
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]
            else:
                y_proba = proba
            return y_class, y_proba, None

    return _MLP(MLPConfig())


class DeepEnsemble(Model[DeepEnsembleConfig]):
    """A bag of architecture-diverse :class:`Model` constituents combined
    via the Top-K confidence ensemble combinator.
    """

    name = "deep_ensemble"
    kind = "ensemble"

    def __init__(self, config: DeepEnsembleConfig | None = None) -> None:
        super().__init__(config or DeepEnsembleConfig())
        self._constituents: list[Model[Any]] = []
        torch_ok = _has_torch()
        if self.config.include_lr:
            self._constituents.append(LogisticRegressionModel(LinearConfig()))
        if self.config.include_rf:
            self._constituents.append(RandomForestModel(RandomForestConfig()))
        if self.config.include_mlp and torch_ok:
            self._constituents.append(_MLPModel())
        if self.config.include_lstm and torch_ok:
            self._constituents.append(LSTMModel(LSTMConfig(sequence_length=self.config.lookback)))
        if self.config.include_nbeats and torch_ok:
            self._constituents.append(
                NBEATSModel(NBEATSConfig(lookback=self.config.lookback))
            )
        if not self._constituents:
            raise ValueError("DeepEnsemble: at least one constituent must be enabled")
        self._combiner = TopKConfidenceEnsemble(
            models=self._constituents, config=self.config.spec
        )

    @property
    def n_constituents(self) -> int:
        return len(self._constituents)

    @property
    def constituent_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self._constituents)

    def _fit_core(
        self,
        features: FeatureMatrix,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        loss_fn: str,
    ) -> tuple[Any, dict[str, float]]:
        state, metrics = self._combiner._fit_core(
            features, y, sample_weight=sample_weight, loss_fn=loss_fn
        )
        return state, {
            **metrics,
            "n_constituents": float(len(self._constituents)),
        }

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        return self._combiner._predict_core(trained, features)


__all__ = ["DeepEnsemble", "DeepEnsembleConfig"]
