"""
Run.py

Demonstration script:
1) generate synthetic 2D time-series data [N, T, S];
2) construct weak and strong first-principles intervals (WFPI, SFPI);
3) train the semi-supervised Kumaraswamy-mixture model;
4) obtain point and interval predictions on the test set.
"""

from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

from WFPI import build_wfpi, WFPIConfig
from SFPI import build_sfpi, SFPIConfig
from Method import fit_predict, MethodConfig


def make_synthetic_dataset(
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 64,
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
            + 0.03 * regime
            + 0.012 * rng.normal(size=n)
        )
        y = np.clip(y, 0.03, 0.97).astype(np.float32)
        return X, y

    X_train, y_train = make_split(n_train)
    X_val, y_val = make_split(n_val)
    X_test, y_test = make_split(n_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def coverage(y_true, low, high):
    return float(np.mean((y_true >= low) & (y_true <= high)))


def avg_width(low, high):
    return float(np.mean(high - low))


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = make_synthetic_dataset()

    wfpi = build_wfpi(
        X_train, X_val, X_test,
        y_train=y_train, y_val=y_val,
        config=WFPIConfig(tmpd_idx=0, tppd_idx=1, d_idx=2, b_idx=3, f_idx=4, min_ppd=0.0, max_ppd=1.0),
    )

    sfpi = build_sfpi(
        X_train, X_val, X_test,
        y_train=y_train, y_val=y_val,
        config=SFPIConfig(top_idx=1, min_ppd=0.0, max_ppd=1.0, stage_candidates=tuple(range(8, 61, 2)), calibration_tolerance=0.02, stage_scale=0.04, nonlinearity_power=1.25),
    )

    print("WFPI metadata:", wfpi["metadata"])
    print("SFPI metadata:", sfpi["metadata"])

    cfg = MethodConfig(
        n_components=2,
        lambda_u=0.3,
        q_s=0.9,
        alpha=0.1,
        epochs=120,
        learning_rate=1e-3,
        weight_decay=1e-5,
        weight_hidden_size=32,
        component_hidden_size=32,
        conv_kernel_size=5,
        device="cpu",
        verbose=True,
    )

    print("\n=== Training with WFPI ===")
    result_w = fit_predict(
        X_train, y_train, wfpi["train"],
        X_val, y_val, wfpi["val"],
        X_test, intervals_test=wfpi["test"],
        config=cfg,
    )

    print("\n=== Training with SFPI ===")
    result_s = fit_predict(
        X_train, y_train, sfpi["train"],
        X_val, y_val, sfpi["val"],
        X_test, intervals_test=sfpi["test"],
        config=cfg,
    )

    print("\n=== Test metrics ===")
    for name, res in [("WFPI", result_w), ("SFPI", result_s)]:
        y_hat = res["y_point"]
        y_lo = res["y_lower"]
        y_hi = res["y_upper"]
        print(
            f"{name}: RMSE={rmse(y_test, y_hat):.4f}, "
            f"Coverage={coverage(y_test, y_lo, y_hi):.4f}, "
            f"AvgWidth={avg_width(y_lo, y_hi):.4f}"
        )


if __name__ == "__main__":
    main()
