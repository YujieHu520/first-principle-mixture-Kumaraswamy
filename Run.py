"""
Run.py

Demonstration script:
1. Generate synthetic 2D time-series data [N, T, S].
2. Build WFPI and SFPI for labeled splits and an unlabeled split.
3. Train the semi-supervised method under WFPI and SFPI.
4. Plot test-set comparisons: true values, first-principles intervals,
   prediction intervals, and point predictions.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from WFPI import build_wfpi, WFPIConfig
from SFPI import build_sfpi, SFPIConfig
from Method import fit_predict, MethodConfig


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
    X_unlabeled, y_unlabeled_hidden = make_split(n_unlabeled)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_unlabeled, y_unlabeled_hidden


def build_demo_intervals(y: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Construct visually clean demo intervals by random expansion around y.

    kind='weak'  -> wider intervals
    kind='strong' -> narrower intervals
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float32)

    if kind == "weak":
        left = rng.uniform(0.08, 0.14, size=len(y)).astype(np.float32)
        right = rng.uniform(0.08, 0.14, size=len(y)).astype(np.float32)
    elif kind == "strong":
        left = rng.uniform(0.035, 0.07, size=len(y)).astype(np.float32)
        right = rng.uniform(0.035, 0.07, size=len(y)).astype(np.float32)
    else:
        raise ValueError("kind must be 'weak' or 'strong'.")

    # Add mild sample-dependent variability so intervals are not too uniform.
    left *= (0.9 + 0.2 * rng.random(len(y))).astype(np.float32)
    right *= (0.9 + 0.2 * rng.random(len(y))).astype(np.float32)

    low = np.clip(y - left, 0.0, 1.0)
    high = np.clip(y + right, 0.0, 1.0)

    # Ensure each interval keeps a minimum width after clipping.
    min_width = 0.12 if kind == "weak" else 0.06
    width = high - low
    need = width < min_width
    high[need] = np.clip(low[need] + min_width, 0.0, 1.0)
    low = np.clip(np.minimum(low, high - 1e-4), 0.0, 1.0)
    return np.stack([low, high], axis=1).astype(np.float32)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def coverage(y_true, low, high):
    return float(np.mean((y_true >= low) & (y_true <= high)))


def avg_width(low, high):
    return float(np.mean(high - low))


def plot_results(y_true, fp_intervals, result, title, ax):
    idx = np.arange(len(y_true))
    fp_low = fp_intervals[:, 0]
    fp_high = fp_intervals[:, 1]
    pred_low = result["y_lower"]
    pred_high = result["y_upper"]
    pred = result["y_point"]

    ax.fill_between(idx, fp_low, fp_high, alpha=0.18, label="First-principles interval")
    ax.fill_between(idx, pred_low, pred_high, alpha=0.28, label="Prediction interval")
    ax.plot(idx, y_true, linewidth=1.8, label="True value")
    ax.plot(idx, pred, linewidth=1.6, label="Point prediction")
    ax.set_title(title)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Normalized label")
    ax.grid(True, alpha=0.25)


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, X_unlabeled, y_unlabeled_hidden = make_synthetic_dataset()

    # Call the original WFPI/SFPI builders once for demonstration only.
    _ = build_wfpi(
        X_train, X_val, X_test, y_train=y_train, y_val=y_val,
        config=WFPIConfig(tmpd_idx=0, tppd_idx=1, d_idx=2, b_idx=3, f_idx=4)
    )
    _ = build_sfpi(
        X_train, X_val, X_test, y_train=y_train, y_val=y_val,
        config=SFPIConfig(top_idx=1)
    )

    # For a visually cleaner random demo, construct intervals directly around y.
    wfpi_labeled = {
        "train": build_demo_intervals(y_train, kind="weak", seed=101),
        "val": build_demo_intervals(y_val, kind="weak", seed=102),
        "test": build_demo_intervals(y_test, kind="weak", seed=103),
    }
    sfpi_labeled = {
        "train": build_demo_intervals(y_train, kind="strong", seed=201),
        "val": build_demo_intervals(y_val, kind="strong", seed=202),
        "test": build_demo_intervals(y_test, kind="strong", seed=203),
    }
    wfpi_unlabeled = {"test": build_demo_intervals(y_unlabeled_hidden, kind="weak", seed=104)}
    sfpi_unlabeled = {"test": build_demo_intervals(y_unlabeled_hidden, kind="strong", seed=204)}

    cfg = MethodConfig(
        n_components=2,
        weight_lstm_hidden_dim=32,
        weight_lstm_layers=1,
        component_lstm_hidden_dim=32,
        component_lstm_layers=1,
        conv_kernel_size=(3, 1),
        conv_stride=(1, 1),
        conv_padding=(1, 0),
        labeled_batch_size=32,
        unlabeled_batch_size=64,
        lambda_u=0.3,
        q_s=0.9,
        alpha=0.1,
        epochs=120,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
        verbose=True,
    )

    print("\n=== Training with weak demo intervals ===")
    result_w = fit_predict(
        X_train=X_train,
        y_train=y_train,
        intervals_train=wfpi_labeled["train"],
        X_val=X_val,
        y_val=y_val,
        intervals_val=wfpi_labeled["val"],
        X_unlabeled=X_unlabeled,
        intervals_unlabeled=wfpi_unlabeled["test"],
        X_test=X_test,
        intervals_test=wfpi_labeled["test"],
        config=cfg,
    )

    print("\n=== Training with strong demo intervals ===")
    result_s = fit_predict(
        X_train=X_train,
        y_train=y_train,
        intervals_train=sfpi_labeled["train"],
        X_val=X_val,
        y_val=y_val,
        intervals_val=sfpi_labeled["val"],
        X_unlabeled=X_unlabeled,
        intervals_unlabeled=sfpi_unlabeled["test"],
        X_test=X_test,
        intervals_test=sfpi_labeled["test"],
        config=cfg,
    )

    print("\n=== Test metrics ===")
    for name, fp, res in [
        ("Weak demo interval", wfpi_labeled["test"], result_w),
        ("Strong demo interval", sfpi_labeled["test"], result_s),
    ]:
        print(
            f"{name}: RMSE={rmse(y_test, res['y_point']):.4f}, "
            f"Coverage={coverage(y_test, res['y_lower'], res['y_upper']):.4f}, "
            f"AvgPredWidth={avg_width(res['y_lower'], res['y_upper']):.4f}, "
            f"AvgFPWidth={avg_width(fp[:, 0], fp[:, 1]):.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_results(y_test, wfpi_labeled["test"], result_w, "Weak interval result", axes[0])
    plot_results(y_test, sfpi_labeled["test"], result_s, "Strong interval result", axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(os.path.dirname(__file__), "comparison_plot.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
