"""
ToyDataSet.py

Synthetic dataset generation and toy first-principles interval construction for
visual demonstration.

Key idea
--------
The label interval is NOT obtained by directly expanding/shrinking y. Instead,
an interval is imposed on one feature appearing in the label-generation formula.
Unlike the previous demo, the label-generation formula uses temporal summaries
computed from the whole input window X[:, :T, :], rather than only the last time
step. The corresponding y-interval is then obtained by propagating this feature
interval through the same formula.

Two interval qualities are provided:
- WFPI: wider feature interval -> wider label interval
- SFPI: narrower feature interval -> narrower label interval
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


ArrayDict = Dict[str, np.ndarray]


def _temporal_weights(T: int) -> np.ndarray:
    """Mildly emphasize recent samples while using the full time window."""
    w = np.linspace(0.6, 1.4, T, dtype=np.float32)
    w = w / np.sum(w)
    return w



def _temporal_features(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute temporal summary features from the whole input window.

    All features are obtained from the previous T time steps, so the synthetic
    label depends on the whole sequence rather than only the last sample.
    """
    X = np.asarray(X, dtype=np.float32)
    T = X.shape[1]
    w = _temporal_weights(T)[None, :, None]

    weighted = np.sum(X * w, axis=1)

    feats = {
        "tmpd": weighted[:, 0],
        "tppd": weighted[:, 1],
        "D": weighted[:, 2],
        "B": weighted[:, 3],
        "F": np.clip(weighted[:, 4], 1e-6, None),
        "extra": weighted[:, 5],
    }
    return feats



def _compute_y_from_temporal_x(X: np.ndarray) -> np.ndarray:
    """Deterministic toy label-generation formula using the full time window.

    The label is generated from temporal summary features extracted from the
    previous T time steps. This keeps the demo consistent with dynamic modeling.
    """
    feats = _temporal_features(X)

    y = (
        0.52 * np.power(1.0 - feats["tppd"], 1.2)
        + 0.18 * feats["tmpd"]
        + 0.10 * (feats["D"] / feats["F"])
        + 0.06 * feats["B"]
        + 0.06 * feats["extra"]
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

        y = _compute_y_from_temporal_x(X)
        return X, y

    X_train, y_train = make_split(n_train)
    X_val, y_val = make_split(n_val)
    X_test, y_test = make_split(n_test)
    X_unlabeled, y_unlabeled_hidden = make_split(n_unlabeled)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_unlabeled, y_unlabeled_hidden



def _interval_from_feature_uncertainty(X: np.ndarray, kind: str) -> np.ndarray:
    """Construct label intervals by imposing an interval on a temporal feature.

    The selected feature is the temporal summary of the TPPD-like variable.
    Because the label formula contains the monotone-decreasing term
    0.52 * (1 - x_tppd)^1.2, an interval on this temporal feature induces a
    closed-form interval on y without directly modifying y.

    kind='weak'  -> wider feature interval (WFPI)
    kind='strong' -> narrower feature interval (SFPI)
    """
    feats = _temporal_features(X)
    x_tppd = np.clip(feats["tppd"], 0.20, 0.95)

    if kind == "weak":
        dx = 0.30 + 0.020 * np.abs(np.sin(4.0 * x_tppd))
    elif kind == "strong":
        dx = 0.20 + 0.010 * np.abs(np.sin(4.0 * x_tppd))
    else:
        raise ValueError("kind must be 'weak' or 'strong'.")

    x_low = np.clip(x_tppd - dx, 0.20, 0.95)
    x_high = np.clip(x_tppd + dx, 0.20, 0.95)

    base = (
        0.18 * feats["tmpd"]
        + 0.10 * (feats["D"] / feats["F"])
        + 0.06 * feats["B"]
        + 0.06 * feats["extra"]
    )

    # Since g(x) = 0.52 * (1 - x)^1.2 is decreasing in x,
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
