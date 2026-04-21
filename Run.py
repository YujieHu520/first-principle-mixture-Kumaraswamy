"""
Run.py

Demonstration script:
1. Generate synthetic 2D time-series data [N, T, S].
2. Build toy WFPI and SFPI for labeled splits and an unlabeled split.
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

from Method import fit_predict, MethodConfig
from ToyDataSet import make_synthetic_dataset, build_toy_intervals


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
    ax.plot(idx, pred, linewidth=1.6, label="Predicted Value")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Normalized label")
    ax.grid(True, alpha=0.25)


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, X_unlabeled, y_unlabeled_hidden = make_synthetic_dataset()
    wfpi_labeled, sfpi_labeled, wfpi_unlabeled, sfpi_unlabeled = build_toy_intervals(
        X_train, X_val, X_test, X_unlabeled
    )

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

    print("\n=== Training with weak toy intervals ===")
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

    print("\n=== Training with strong toy intervals ===")
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
        ("Weak toy interval", wfpi_labeled["test"], result_w),
        ("Strong toy interval", sfpi_labeled["test"], result_s),
    ]:
        print(
            f"{name}: RMSE={rmse(y_test, res['y_point']):.4f}, "
            f"Coverage={coverage(y_test, res['y_lower'], res['y_upper']):.4f}, "
            f"AvgPredWidth={avg_width(res['y_lower'], res['y_upper']):.4f}, "
            f"AvgFPWidth={avg_width(fp[:, 0], fp[:, 1]):.4f}"
        )

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_results(y_test, wfpi_labeled["test"], result_w, "WFPI result", axes[0])
    plot_results(y_test, sfpi_labeled["test"], result_s, "SFPI result", axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(os.path.dirname(__file__), "comparison_plot.svg")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
