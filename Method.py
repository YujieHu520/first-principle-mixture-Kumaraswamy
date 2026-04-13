"""
Method.py

Semi-supervised temporal modeling with a truncated mixture of Kumaraswamy components.

"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ArrayLike = np.ndarray


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _standardize_intervals(intervals: Any, n: int) -> np.ndarray:
    if isinstance(intervals, (tuple, list)) and len(intervals) == 2:
        L = _to_numpy(intervals[0]).reshape(-1)
        U = _to_numpy(intervals[1]).reshape(-1)
        arr = np.column_stack([L, U])
    else:
        arr = _to_numpy(intervals)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("intervals must be (L, U) or an array of shape [N, 2].")
    if arr.shape[0] != n:
        raise ValueError(f"interval count {arr.shape[0]} does not match n={n}.")
    return arr.astype(np.float32)


def ensure_valid_intervals(intervals: np.ndarray, margin: float = 1e-4) -> np.ndarray:
    arr = intervals.astype(np.float32).copy()
    L = np.clip(arr[:, 0], 0.0, 1.0)
    U = np.clip(arr[:, 1], 0.0, 1.0)
    U = np.maximum(U, L + margin)
    U = np.clip(U, 0.0, 1.0)
    L = np.minimum(L, U - margin)
    L = np.clip(L, 0.0, 1.0)
    return np.column_stack([L, U]).astype(np.float32)


def kumaraswamy_cdf(z: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    z = torch.clamp(z, eps, 1.0 - eps)
    a = torch.clamp(a, eps, None)
    b = torch.clamp(b, eps, None)
    return 1.0 - torch.pow(1.0 - torch.pow(z, a), b)


def kumaraswamy_log_pdf(z: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    z = torch.clamp(z, eps, 1.0 - eps)
    a = torch.clamp(a, eps, None)
    b = torch.clamp(b, eps, None)
    return (
        torch.log(a + eps)
        + torch.log(b + eps)
        + (a - 1.0) * torch.log(z)
        + (b - 1.0) * torch.log(torch.clamp(1.0 - torch.pow(z, a), eps, None))
    )


def interval_probability(L: torch.Tensor, U: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.clamp(kumaraswamy_cdf(U, a, b, eps) - kumaraswamy_cdf(L, a, b, eps), min=eps)


class SSRKumaraswamyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int = 2,
        weight_lstm_hidden_dim: int = 32,
        weight_lstm_layers: int = 1,
        component_lstm_hidden_dim: int = 32,
        component_lstm_layers: int = 1,
        conv_kernel_size: Tuple[int, int] = (3, 1),
        conv_stride: Tuple[int, int] = (1, 1),
        conv_padding: Tuple[int, int] = (1, 0),
        dropout: float = 0.0,
        shape_floor: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        self.shape_floor = shape_floor

        self.weight_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=weight_lstm_hidden_dim,
            num_layers=weight_lstm_layers,
            batch_first=True,
            dropout=dropout if weight_lstm_layers > 1 else 0.0,
        )
        self.weight_head = nn.Linear(weight_lstm_hidden_dim, n_components)

        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=n_components,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            bias=True,
        )

        self.component_lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=input_dim,
                    hidden_size=component_lstm_hidden_dim,
                    num_layers=component_lstm_layers,
                    batch_first=True,
                    dropout=dropout if component_lstm_layers > 1 else 0.0,
                )
                for _ in range(n_components)
            ]
        )
        self.component_heads = nn.ModuleList(
            [nn.Linear(component_lstm_hidden_dim, 2) for _ in range(n_components)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, S]
        h_v, _ = self.weight_lstm(x)
        weights = torch.softmax(self.weight_head(h_v[:, -1, :]), dim=-1)

        conv_in = x.unsqueeze(1)  # [B, 1, T, S]
        conv_out = torch.tanh(self.conv2d(conv_in))  # [B, K, T, S]

        a_list = []
        b_list = []
        for k in range(self.n_components):
            seq_k = conv_out[:, k, :, :]  # [B, T, S]
            h_k, _ = self.component_lstms[k](seq_k)
            raw = self.component_heads[k](h_k[:, -1, :])
            ab = F.softplus(raw) + self.shape_floor
            a_list.append(ab[:, 0])
            b_list.append(ab[:, 1])

        a = torch.stack(a_list, dim=1)
        b = torch.stack(b_list, dim=1)
        return weights, a, b


@dataclass
class MethodConfig:
    n_components: int = 2

    weight_lstm_hidden_dim: int = 32
    weight_lstm_layers: int = 1
    component_lstm_hidden_dim: int = 32
    component_lstm_layers: int = 1

    conv_kernel_size: Tuple[int, int] = (3, 1)
    conv_stride: Tuple[int, int] = (1, 1)
    conv_padding: Tuple[int, int] = (1, 0)
    dropout: float = 0.0
    shape_floor: float = 0.2

    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    labeled_batch_size: int = 32
    unlabeled_batch_size: int = 64
    lambda_u: float = 0.3
    q_s: float = 0.9
    alpha: float = 0.1
    grad_clip: float = 5.0
    early_stopping_patience: int = 40
    optimizer_name: str = "Adam"
    device: str = "cpu"
    random_seed: int = 42
    verbose: bool = True


def _make_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _compute_labeled_loss(weights: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, intervals: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = y.view(-1, 1)
    L = intervals[:, 0:1]
    U = intervals[:, 1:2]
    y = torch.clamp(y, L + 1e-6, U - 1e-6)

    log_pdf = kumaraswamy_log_pdf(y, a, b, eps=eps)
    prob_interval = interval_probability(L, U, a, b, eps=eps)
    comp = weights * torch.exp(log_pdf) / prob_interval
    comp_sum = torch.clamp(comp.sum(dim=1, keepdim=True), min=eps)
    r = (comp / comp_sum).detach()

    return -torch.mean(
        torch.sum(r * (torch.log(weights + eps) + log_pdf - torch.log(prob_interval + eps)), dim=1)
    )


def _compute_unlabeled_loss(weights: torch.Tensor, a: torch.Tensor, b: torch.Tensor, intervals: torch.Tensor, q_s: float, eps: float = 1e-8) -> torch.Tensor:
    L = intervals[:, 0:1]
    U = intervals[:, 1:2]
    prob_interval = interval_probability(L, U, a, b, eps=eps)
    comp = weights * prob_interval
    comp_sum = torch.clamp(comp.sum(dim=1, keepdim=True), min=eps)
    R = (comp / comp_sum).detach()

    core = -torch.mean(torch.sum(R * (torch.log(weights + eps) + torch.log(prob_interval + eps)), dim=1))
    mix_prob = torch.sum(weights * prob_interval, dim=1)
    penalty = torch.mean(torch.relu(torch.as_tensor(q_s, device=weights.device) - mix_prob))
    return core + penalty


@torch.no_grad()
def _predict_quantiles(model: nn.Module, x: torch.Tensor, intervals: torch.Tensor, alpha: float, n_bisect_steps: int = 50):
    model.eval()
    weights, a, b = model(x)
    L = intervals[:, 0:1]
    U = intervals[:, 1:2]

    def mix_cdf(z: torch.Tensor) -> torch.Tensor:
        num = torch.sum(weights * (kumaraswamy_cdf(z, a, b) - kumaraswamy_cdf(L, a, b)), dim=1)
        den = torch.sum(weights * (kumaraswamy_cdf(U, a, b) - kumaraswamy_cdf(L, a, b)), dim=1)
        return num / torch.clamp(den, min=1e-8)

    def invert(q: float) -> torch.Tensor:
        lo = L.clone()
        hi = U.clone()
        q_tensor = torch.full((x.shape[0],), float(q), device=x.device)
        for _ in range(n_bisect_steps):
            mid = 0.5 * (lo + hi)
            cdf_mid = mix_cdf(mid)
            go_right = cdf_mid < q_tensor
            lo = torch.where(go_right.unsqueeze(1), mid, lo)
            hi = torch.where(go_right.unsqueeze(1), hi, mid)
        return 0.5 * (lo + hi)

    point = invert(0.5).squeeze(1).cpu().numpy()
    lower = invert(alpha / 2.0).squeeze(1).cpu().numpy()
    upper = invert(1.0 - alpha / 2.0).squeeze(1).cpu().numpy()
    aux = {
        "weights": weights.cpu().numpy(),
        "a": a.cpu().numpy(),
        "b": b.cpu().numpy(),
    }
    return point, lower, upper, aux


def fit_predict(
    X_train: ArrayLike,
    y_train: ArrayLike,
    intervals_train: Any,
    X_val: ArrayLike,
    y_val: ArrayLike,
    intervals_val: Any,
    X_unlabeled: ArrayLike,
    intervals_unlabeled: Any,
    X_test: ArrayLike,
    intervals_test: Optional[Any] = None,
    config: Optional[MethodConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = MethodConfig()

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    X_train = _to_numpy(X_train).astype(np.float32)
    y_train = _to_numpy(y_train).reshape(-1).astype(np.float32)
    X_val = _to_numpy(X_val).astype(np.float32)
    y_val = _to_numpy(y_val).reshape(-1).astype(np.float32)
    X_unlabeled = _to_numpy(X_unlabeled).astype(np.float32)
    X_test = _to_numpy(X_test).astype(np.float32)

    intervals_train = ensure_valid_intervals(_standardize_intervals(intervals_train, X_train.shape[0]))
    intervals_val = ensure_valid_intervals(_standardize_intervals(intervals_val, X_val.shape[0]))
    intervals_unlabeled = ensure_valid_intervals(_standardize_intervals(intervals_unlabeled, X_unlabeled.shape[0]))
    if intervals_test is None:
        intervals_test = np.column_stack([np.zeros(X_test.shape[0]), np.ones(X_test.shape[0])]).astype(np.float32)
    else:
        intervals_test = ensure_valid_intervals(_standardize_intervals(intervals_test, X_test.shape[0]))

    _, _, input_dim = X_train.shape

    requested = config.device.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested)

    model = SSRKumaraswamyNet(
        input_dim=input_dim,
        n_components=config.n_components,
        weight_lstm_hidden_dim=config.weight_lstm_hidden_dim,
        weight_lstm_layers=config.weight_lstm_layers,
        component_lstm_hidden_dim=config.component_lstm_hidden_dim,
        component_lstm_layers=config.component_lstm_layers,
        conv_kernel_size=config.conv_kernel_size,
        conv_stride=config.conv_stride,
        conv_padding=config.conv_padding,
        dropout=config.dropout,
        shape_floor=config.shape_floor,
    ).to(device)

    if config.optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(intervals_train, dtype=torch.float32),
    )
    unlabeled_ds = TensorDataset(
        torch.tensor(X_unlabeled, dtype=torch.float32),
        torch.tensor(intervals_unlabeled, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=config.labeled_batch_size, shuffle=True, drop_last=False)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=config.unlabeled_batch_size, shuffle=True, drop_last=False)

    x_val_t = _make_tensor(X_val, device)
    y_val_t = _make_tensor(y_val, device)
    int_val_t = _make_tensor(intervals_val, device)

    history = {"train_total": [], "train_labeled": [], "train_unlabeled": [], "val_rmse": []}
    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_l = 0.0
        epoch_u = 0.0
        n_steps = max(len(train_loader), len(unlabeled_loader))

        labeled_iter = cycle(train_loader)
        unlabeled_iter = cycle(unlabeled_loader)

        for _ in range(n_steps):
            xb_l, yb_l, ib_l = next(labeled_iter)
            xb_u, ib_u = next(unlabeled_iter)

            xb_l = xb_l.to(device)
            yb_l = yb_l.to(device)
            ib_l = ib_l.to(device)
            xb_u = xb_u.to(device)
            ib_u = ib_u.to(device)

            optimizer.zero_grad()
            w_l, a_l, b_l = model(xb_l)
            loss_l = _compute_labeled_loss(w_l, a_l, b_l, yb_l, ib_l)

            w_u, a_u, b_u = model(xb_u)
            loss_u = _compute_unlabeled_loss(w_u, a_u, b_u, ib_u, q_s=config.q_s)

            loss = loss_l + config.lambda_u * loss_u
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_l += float(loss_l.item())
            epoch_u += float(loss_u.item())

        val_point, _, _, _ = _predict_quantiles(model, x_val_t, int_val_t, alpha=config.alpha)
        val_rmse = float(np.sqrt(np.mean((val_point - y_val) ** 2)))

        history["train_total"].append(epoch_total / n_steps)
        history["train_labeled"].append(epoch_l / n_steps)
        history["train_unlabeled"].append(epoch_u / n_steps)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if config.verbose and (epoch == 1 or epoch % 25 == 0 or epoch == config.epochs):
            print(
                f"[Epoch {epoch:03d}] total={history['train_total'][-1]:.4f} "
                f"labeled={history['train_labeled'][-1]:.4f} "
                f"unlabeled={history['train_unlabeled'][-1]:.4f} "
                f"val_rmse={val_rmse:.4f}"
            )

        if patience >= config.early_stopping_patience:
            if config.verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    x_test_t = _make_tensor(X_test, device)
    int_test_t = _make_tensor(intervals_test, device)
    y_point, y_lower, y_upper, aux = _predict_quantiles(model, x_test_t, int_test_t, alpha=config.alpha)

    return {
        "y_point": y_point,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "history": history,
        "aux": aux,
        "model": model,
        "config": config,
    }
