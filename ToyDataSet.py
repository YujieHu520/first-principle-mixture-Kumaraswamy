"""
ToyDataSet.py

Synthetic dataset generation and toy first-principles interval construction for
visual demonstration.

Two interval qualities are provided:
- WFPI: wider feature interval -> wider label interval
- SFPI: narrower feature interval -> narrower label interval
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


ArrayDict = Dict[str, np.ndarray]


def _compute_y_from_x_last(X: np.ndarray) -> np.ndarray:
    """Deterministic toy label-generation formula based on the last time step.
    """
    tmpd_last = X[:, -1, 0]
    tppd_last = X[:, -1, 1]
    D_last = X[:, -1, 2]
    B_last = X[:, -1, 3]
    F_last = np.clip(X[:, -1, 4], 1e-6, None)
    extra_last = X[:, -1, 5]

    y = (
        0.52 * np.power(1.0 - tppd_last, 1.2)
        + 0.18 * tmpd_last
        + 0.10 * (D_last / F_last)
        + 0.06 * B_last
        + 0.06 * extra_last
    )
    return np.clip(y, 0.03, 0.97).astype(np.float32)


def make_synthetic_dataset(
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 80,
    n_unlabeled: int = 512,
    T: int = 20,
    S: int = 6,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    def make_split(n: int):
        regime = rng.integers(0, 2, size=n).astype(np.float32)
        X = np.zeros((n, T, S), dtype=np.float32)
        for i in range(n):
            r = regime[i]
            t = np.linspace(0, 1, T, dtype=np.float32)
            tmpd = 0.10 + 0.06 * np.sin(2 * np.pi * t + 0.7 * r) + 0.03 * r + 0.01 * rng.normal(size=T)
            tppd = 0.70 + 0.08 * np.cos(2 * np.pi * t + 0.4 * r) - 0.06 * r + 0.015 * rng.normal(size=T)
            D = 0.55 + 0.08 * np.sin(1.5 * np.pi * t + 0.2) + 0.02 * rng.normal(size=T)
            B = 0.45 + 0.06 * np.cos(1.1 * np.pi * t + 0.3) + 0.02 * rng.normal(size=T)
            F = 1.00 + 0.05 * np.sin(0.7 * np.pi * t) + 0.03 * rng.normal(size=T)
            extra = 0.30 + 0.20 * r + 0.05 * np.sin(3 * np.pi * t) + 0.01 * rng.normal(size=T)
            X[i, :, 0] = np.clip(tmpd, 0.02, 0.40)
            X[i, :, 1] = np.clip(tppd, 0.20, 0.95)
            X[i, :, 2] = np.clip(D, 0.20, 0.95)
            X[i, :, 3] = np.clip(B, 0.15, 0.95)
            X[i, :, 4] = np.clip(F, 0.60, 1.40)
            X[i, :, 5] = np.clip(extra, 0.00, 1.00)

        y = _compute_y_from_x_last(X)
        return X, y

    X_train, y_train = make_split(n_train)
    X_val, y_val = make_split(n_val)
    X_test, y_test = make_split(n_test)
    X_unlabeled, y_unlabeled_hidden = make_split(n_unlabeled)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_unlabeled, y_unlabeled_hidden


def _interval_from_feature_uncertainty(
    X: np.ndarray,
    kind: str,
    tppd_idx: int = 1,
) -> np.ndarray:
    """Construct label intervals by imposing an interval on x (TPPD-like variable).

    The label formula contains the nonlinear term 0.52 * (1 - x)^1.2. Since this
    term is monotone decreasing in x over [0, 1], an interval on x induces a
    closed-form interval on y without directly modifying y.

    kind='weak'  -> wider x-interval (WFPI)
    kind='strong' -> narrower x-interval (SFPI)
    """
    X = np.asarray(X, dtype=np.float32)
    tppd_last = np.clip(X[:, -1, tppd_idx], 0.20, 0.95)
    tmpd_last = X[:, -1, 0]
    D_last = X[:, -1, 2]
    B_last = X[:, -1, 3]
    F_last = np.clip(X[:, -1, 4], 1e-6, None)
    extra_last = X[:, -1, 5]

    # Width is imposed on x (not on y). WFPI is intentionally wider than SFPI.
    if kind == "weak":
        dx = 0.20 + 0.020 * np.abs(np.sin(4.0 * tppd_last))
    elif kind == "strong":
        dx = 0.15 + 0.010 * np.abs(np.sin(4.0 * tppd_last))
    else:
        raise ValueError("kind must be 'weak' or 'strong'.")

    x_low = np.clip(tppd_last - dx, 0.20, 0.95)
    x_high = np.clip(tppd_last + dx, 0.20, 0.95)

    base = 0.18 * tmpd_last + 0.10 * (D_last / F_last) + 0.06 * B_last + 0.06 * extra_last

    # Because g(x) = 0.52 * (1 - x)^1.2 is decreasing in x,
    # y_high corresponds to x_low and y_low corresponds to x_high.
    y_low = base + 0.52 * np.power(1.0 - x_high, 1.2)
    y_high = base + 0.52 * np.power(1.0 - x_low, 1.2)

    y_low = np.clip(y_low, 0.0, 1.0)
    y_high = np.clip(y_high, 0.0, 1.0)
    y_high = np.maximum(y_high, y_low + 1e-4)

    return np.stack([y_low, y_high], axis=1).astype(np.float32)


def build_toy_intervals(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    X_unlabeled: np.ndarray,
) -> Tuple[ArrayDict, ArrayDict, ArrayDict, ArrayDict]:
    """Return WFPI/SFPI intervals for labeled and unlabeled splits."""
    wfpi_labeled = {
        "train": _interval_from_feature_uncertainty(X_train, kind="weak"),
        "val": _interval_from_feature_uncertainty(X_val, kind="weak"),
        "test": _interval_from_feature_uncertainty(X_test, kind="weak"),
    }
    sfpi_labeled = {
        "train": _interval_from_feature_uncertainty(X_train, kind="strong"),
        "val": _interval_from_feature_uncertainty(X_val, kind="strong"),
        "test": _interval_from_feature_uncertainty(X_test, kind="strong"),
    }
    wfpi_unlabeled = {
        "test": _interval_from_feature_uncertainty(X_unlabeled, kind="weak"),
    }
    sfpi_unlabeled = {
        "test": _interval_from_feature_uncertainty(X_unlabeled, kind="strong"),
    }
    return wfpi_labeled, sfpi_labeled, wfpi_unlabeled, sfpi_unlabeled
