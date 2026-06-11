"""N-BEATS — Neural Basis Expansion Analysis for Time Series.

A *minimal* but architecturally-faithful N-BEATS implementation:

- Each *block* consists of a fully-connected stack with ReLU, two
  residual branches (backcast and forecast), and a basis-expansion
  layer. The basis is the generic ``theta ⋅ b(t)`` form where ``b(t)``
  is a small set of polynomial features of the time index — this is
  the *generic* stack in the original paper (Oreshkin et al., 2020).
- The forecast horizon is fixed at construction time. We use a
  *sequence-to-one* output (last bar) plus the *block stack's* own
  forecast head, producing a single horizon-1 prediction; the wider
  ``forecast_length`` produces ``H`` bars of forecast, useful for
  indirect classification targets.
- The model is a :class:`kairon.models.base.Model` so it integrates
  with the trainer/registry like every other backend.

Reference
---------
Oreshkin, B. N., Carpov, D., Chapados, N., Bengio, Y. (2020).
N-BEATS: Neural basis expansion analysis for interpretable time
series forecasting. ICLR 2020.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairon.models.base import Model, ModelError
from kairon.models.contracts import FeatureMatrix


@dataclass(frozen=True, slots=True)
class NBEATSConfig:
    lookback: int = 32
    horizon: int = 1
    hidden_size: int = 64
    n_blocks: int = 2
    n_layers: int = 2
    theta_dim: int = 4
    basis_degree: int = 3  # polynomial basis degree (>= 1)
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 3
    device: str = "cpu"
    random_state: int = 42
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {self.lookback}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.theta_dim < 1:
            raise ValueError(f"theta_dim must be >= 1, got {self.theta_dim}")
        if self.basis_degree < 1:
            raise ValueError(f"basis_degree must be >= 1, got {self.basis_degree}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None  # type: ignore[attr-defined]


def _polynomial_basis(length: int, degree: int) -> np.ndarray:
    """Return shape ``(length, degree+1)`` with t**0..t**degree/degree!"""
    t = np.arange(length, dtype=np.float32) / max(1, length - 1)
    out = np.stack([t**k / max(1, float(math.factorial(k))) for k in range(degree + 1)], axis=-1)
    return out.astype(np.float32)


def _make_sequences_n(
    values: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, Y_target, Y_window) where Y_target is the last ``y`` per
    sequence and Y_window is the trailing ``horizon`` close-equivalent
    proxy (``values[:, -1]`` i.e. last feature) for each sequence.

    For this generic stack we use ``values[:, -1]`` (the most recent
    signal) as a proxy for "level"; the model learns the mapping.
    """
    n = values.shape[0]
    if n <= lookback:
        x = np.zeros((1, lookback, values.shape[1]), dtype=np.float32)
        x[0, :n] = values
        return x, y[-1:] if y.size else np.zeros(0, dtype=np.int64), x[0:1, -horizon:, -1]
    n_seq = n - lookback
    x = np.zeros((n_seq, lookback, values.shape[1]), dtype=np.float32)
    y_window = np.zeros((n_seq, horizon), dtype=np.float32)
    for i in range(n_seq):
        x[i] = values[i : i + lookback]
        end = min(i + lookback + horizon, n)
        y_window[i, : end - (i + lookback)] = values[i + lookback : end, -1]
    y_target = y[lookback:]
    return x, y_target, y_window


def _make_predict_sequences_n(
    values: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    if n <= lookback:
        x = np.zeros((1, lookback, values.shape[1]), dtype=np.float32)
        x[0, :n] = values
        return x, x[0:1, -horizon:, -1]
    n_seq = n - lookback
    x = np.zeros((n_seq, lookback, values.shape[1]), dtype=np.float32)
    y_window = np.zeros((n_seq, horizon), dtype=np.float32)
    for i in range(n_seq):
        x[i] = values[i : i + lookback]
        end = min(i + lookback + horizon, n)
        y_window[i, : end - (i + lookback)] = values[i + lookback : end, -1]
    return x, y_window


class NBEATSModel(Model[NBEATSConfig]):
    """A small, generic N-BEATS stack used as a deep-TS backend.

    The output is a *forecast* of the next ``horizon`` bars of the
    last feature (treated as a level proxy). For classification we
    reduce to a binary signal: ``forecast[0] > last_value`` → class 1.
    """

    name = "nbeats"
    kind = "deep"

    def __init__(self, config: NBEATSConfig | None = None) -> None:
        super().__init__(config or NBEATSConfig())
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

        x_np, y_np, y_win = _make_sequences_n(
            features.values, y, self.config.lookback, self.config.horizon
        )
        classes = np.unique(y_np)
        n_classes = int(classes.size)
        if n_classes < 2:
            raise ModelError(f"NBEATS needs >= 2 classes, got {n_classes}")
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[int(v)] for v in y_np], dtype=np.int64)

        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_idx)
        yw_t = torch.from_numpy(y_win)

        device = torch.device(self.config.device)
        model = _NBEATSNet(
            n_features=features.n_features,
            lookback=self.config.lookback,
            horizon=self.config.horizon,
            hidden=self.config.hidden_size,
            n_blocks=self.config.n_blocks,
            n_layers=self.config.n_layers,
            theta_dim=self.config.theta_dim,
            basis_degree=self.config.basis_degree,
        ).to(device)
        loss_cls = nn.CrossEntropyLoss()
        loss_reg = nn.MSELoss()
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
        for _epoch in range(self.config.epochs):
            model.train()
            perm = torch.randperm(n)
            epoch_losses: list[float] = []
            for start in range(0, n, self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                xb = x_t[idx].to(device)
                yb = y_t[idx].to(device)
                ywb = yw_t[idx].to(device)
                opt.zero_grad()
                logits, forecast = model(xb)
                l_cls = loss_cls(logits, yb)
                l_reg = loss_reg(forecast, ywb)
                loss = l_cls + l_reg
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.item()))
            mean_loss = float(np.mean(epoch_losses))
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
        return state, {"train_loss": best_loss}

    def _predict_core(
        self,
        trained: Any,
        features: FeatureMatrix,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        torch = importlib.import_module("torch")
        state = trained
        model = state["model"]
        classes: tuple[int, ...] = state["classes"]
        x_np, _ = _make_predict_sequences_n(
            features.values, self.config.lookback, self.config.horizon
        )
        x_t = torch.from_numpy(x_np)
        with torch.no_grad():
            logits, forecast = model(x_t)
            proba_all = torch.softmax(logits, dim=-1).numpy()
        y_idx = proba_all.argmax(axis=1).astype(np.int64)
        y_class = np.array([classes[int(i)] for i in y_idx], dtype=np.int64)
        if proba_all.shape[1] == 2:
            y_proba = proba_all[:, 1]
        else:
            y_proba = proba_all
        return y_class, y_proba, forecast[:, 0].numpy()


# ---------------------------------------------------------------------------
# Internal network
# ---------------------------------------------------------------------------
def _NBEATSNet(
    *,
    n_features: int,
    lookback: int,
    horizon: int,
    hidden: int,
    n_blocks: int,
    n_layers: int,
    theta_dim: int,
    basis_degree: int,
) -> Any:
    nn = importlib.import_module("torch.nn")
    torch = importlib.import_module("torch")

    class _FCStack(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            d_in = n_features
            for _ in range(n_layers):
                layers.append(nn.Linear(d_in, hidden))
                layers.append(nn.ReLU())
                d_in = hidden
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            # x: (B, L, F) -> (B, L*hidden) after flattening per timestep
            b, length, _ = x.shape
            h = self.net(x.reshape(b * length, -1))
            return h.reshape(b, length, -1).reshape(b, -1)

    class _Block(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.fc = _FCStack()
            self.theta_fc = nn.Linear(lookback * hidden, theta_dim * lookback, bias=False)
            self.theta_fb = nn.Linear(lookback * hidden, theta_dim * horizon, bias=False)
            self.basis_degree = basis_degree
            self.lookback = lookback
            self.horizon = horizon
            self.theta_dim = theta_dim
            self.backcast_basis = nn.Parameter(
                torch.from_numpy(_polynomial_basis(lookback, basis_degree)),
                requires_grad=False,
            )
            self.forecast_basis = nn.Parameter(
                torch.from_numpy(_polynomial_basis(horizon, basis_degree)),
                requires_grad=False,
            )

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = self.fc(x)
            tb = self.theta_fc(h).reshape(-1, self.theta_dim, self.lookback)
            tf = self.theta_fb(h).reshape(-1, self.theta_dim, self.horizon)
            backcast = (tb * self.backcast_basis.T).sum(dim=1)
            forecast = (tf * self.forecast_basis.T).sum(dim=1)
            return backcast, forecast

    class _NBEATSNetImpl(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
            d_hid = hidden
            self.classifier = nn.Sequential(
                nn.Linear(horizon, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, max(2, 2)),  # classification outputs ≥ 2 logits
            )
            self.horizon = horizon

        def forward(self, x: Any) -> tuple[Any, Any]:
            residual = x[:, :, -1]  # last feature as level proxy (B, L)
            forecast = torch.zeros(x.shape[0], self.horizon, device=x.device)
            for blk in self.blocks:
                backcast, fore = blk(x)
                residual = residual - backcast
                forecast = forecast + fore
            logits = self.classifier(forecast)
            return logits, forecast

    return _NBEATSNetImpl()


__all__ = ["NBEATSConfig", "NBEATSModel"]
