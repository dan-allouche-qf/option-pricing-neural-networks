"""Utility functions: data splitting, DataLoader, plotting.

Paths are resolved relative to the project root (parent of src/). All
generated artifacts live under outputs/: outputs/data, outputs/figures,
outputs/models, outputs/results.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
DATA_DIR = os.path.join(OUTPUTS_DIR, "data")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

for _d in (FIGURES_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_device():
    """CPU is faster than MPS for 4x400 MLPs on M-series Macs."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed=42):
    """Make a single training run bit-reproducible.

    Seeds Python's random, numpy global RNG, and torch CPU/CUDA RNG.
    Without this, xavier_uniform_ init and DataLoader shuffle are
    non-deterministic, so two consecutive runs produce different weights
    even when LHS data generation is seeded.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data(inputs, outputs, train_ratio=0.9, seed=42):
    rng = np.random.default_rng(seed)
    n = len(inputs)
    idx = rng.permutation(n)
    split = int(n * train_ratio)
    return (
        inputs[idx[:split]], outputs[idx[:split]],
        inputs[idx[split:]], outputs[idx[split:]],
    )


def split_data_3way(inputs, outputs, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split into train / val / test (Heston: 80/10/10, Section 4.4.1)."""
    rng = np.random.default_rng(seed)
    n = len(inputs)
    idx = rng.permutation(n)
    s1 = int(n * ratios[0])
    s2 = int(n * (ratios[0] + ratios[1]))
    return (
        inputs[idx[:s1]], outputs[idx[:s1]],
        inputs[idx[s1:s2]], outputs[idx[s1:s2]],
        inputs[idx[s2:]], outputs[idx[s2:]],
    )


def make_dataloader(X, y, batch_size=1024, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def save_fig(filename):
    """Save the current figure into FIGURES_DIR."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {path}")


def plot_error_histogram(y_true, y_pred, title="Error distribution", filename=None):
    """Histogram of prediction errors with CDF overlay (Figure 7)."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    errors = (y_true - y_pred).flatten()

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.hist(errors, bins=80, density=True, alpha=0.7, color="#2980B9", edgecolor="white")
    ax1.set_xlabel("Prediction error", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(title, fontsize=13)

    ax2 = ax1.twinx()
    sorted_err = np.sort(errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax2.plot(sorted_err, cdf, color="#E67E22", linewidth=2)
    ax2.set_ylabel("Distribution (CDF)", fontsize=12)

    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()


def plot_scatter(y_true, y_pred, title="Predicted vs Actual", filename=None):
    """Scatter plot (Figures 8a, 9a)."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().flatten()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=0.5, alpha=0.3, color="#2980B9")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Actual Value", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title(f"{title}\n$R^2 = {r2:.7f}$", fontsize=13)
    ax.set_aspect("equal")

    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()


def plot_training_history(train_losses, val_losses=None, title="Training loss", filename=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, np.log10(np.array(train_losses) + 1e-20), color="red", alpha=0.7, label="Training")
    if val_losses is not None:
        ax.plot(epochs, np.log10(np.array(val_losses) + 1e-20), color="black", alpha=0.7, label="Validation")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("log(MSE)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()


def plot_training_history_sidebyside(history1, history2, titles, filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, history, title in zip([ax1, ax2], [history1, history2], titles):
        epochs = range(1, len(history["train_losses"]) + 1)
        ax.plot(epochs, np.log10(np.array(history["train_losses"]) + 1e-20),
                color="red", alpha=0.7, label=f"{title}-training")
        if history.get("val_losses"):
            ax.plot(epochs, np.log10(np.array(history["val_losses"]) + 1e-20),
                    color="black", alpha=0.7, label=f"{title}-validation")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("log(MSE)", fontsize=12)
        ax.set_title("model loss", fontsize=13)
        ax.legend()
    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()


def plot_lr_schedule_comparison(histories_dict, filename=None):
    """Figure 4 of the article."""
    colors = {"Constant LR": "black", "Decay LR": "red", "Cyclical LR": "blue"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, losses in histories_dict.items():
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, np.log10(np.array(losses) + 1e-20),
                color=colors.get(label, "gray"), alpha=0.7, label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("log(MSE)", fontsize=12)
    ax.set_title("The history of training loss", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()


def plot_lr_finder(lrs, losses, filename=None):
    """Figure 3 of the article."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, color="#2980B9", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate", fontsize=12)
    ax.set_ylabel("Avg loss", fontsize=12)
    ax.set_title("Average training loss against varying learning rates", fontsize=13)
    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()


def plot_dataset_size_study(cases, train_mse, test_mse, r2_test,
                            train_std=None, test_std=None, r2_std=None,
                            filename=None):
    """Figure 6 of the article."""
    fig, ax1 = plt.subplots(figsize=(8, 6))
    log_train = np.log10(np.array(train_mse) + 1e-20)
    log_test = np.log10(np.array(test_mse) + 1e-20)

    if train_std is not None:
        log_train_std = np.array(train_std) / (np.array(train_mse) * np.log(10) + 1e-20)
        log_test_std = np.array(test_std) / (np.array(test_mse) * np.log(10) + 1e-20)
        ax1.errorbar(cases, log_train, yerr=log_train_std, color="blue", marker="o",
                     linestyle="-", label="Training", capsize=3)
        ax1.errorbar(cases, log_test, yerr=log_test_std, color="red", marker="s",
                     linestyle="--", label="Testing", capsize=3)
    else:
        ax1.plot(cases, log_train, "bo-", label="Training")
        ax1.plot(cases, log_test, "rs--", label="Testing")

    ax1.set_xlabel("case", fontsize=12)
    ax1.set_ylabel("log(MSE)", fontsize=12, color="blue")
    ax1.legend(loc="center left")

    ax2 = ax1.twinx()
    r2_pct = np.array(r2_test) * 100
    if r2_std is not None:
        r2_pct_std = np.array(r2_std) * 100
        ax2.errorbar(cases, r2_pct, yerr=r2_pct_std, color="gray", marker="^",
                     linestyle=":", label="$R^2$ on test", capsize=3)
    else:
        ax2.plot(cases, r2_pct, "g^:", label="$R^2$ on test")
    ax2.set_ylabel("$R^2$(%) ", fontsize=12)
    ax2.legend(loc="center right")

    plt.title("$R^2$ and MSE vs. size of the training set", fontsize=13)
    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()


def plot_pipeline_diagram(filename=None):
    """Figure 10 of the article."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.text(0.1, 0.5, "COS-Brent Approach:", fontsize=11, fontweight="bold",
            va="center", ha="left")
    boxes = [
        (1.8, r"$r,\rho,\kappa,v_0,\bar{v},\gamma$" + "\n" + r"$\tau, m$"),
        (3.6, "Heston model"),
        (5.2, "COS method"),
        (6.8, "Option price"),
        (8.1, "Brent"),
        (9.2, "Implied\nvolatility"),
    ]
    for x, label in boxes:
        bbox = dict(boxstyle="round,pad=0.3", facecolor="#D5E8D4", edgecolor="black")
        ax.text(x, 0.5, label, fontsize=9, va="center", ha="center", bbox=bbox)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.45
        x2 = boxes[i + 1][0] - 0.45
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5),
                     arrowprops=dict(arrowstyle="->", color="black"))

    ax = axes[1]
    ax.text(0.1, 0.5, "Two-ANNs Approach:", fontsize=11, fontweight="bold",
            va="center", ha="left")
    boxes2 = [
        (1.8, r"$r,\rho,\kappa,v_0,\bar{v},\gamma$" + "\n" + r"$\tau, m$"),
        (3.6, "Heston model"),
        (5.2, "ANN-Heston"),
        (6.8, "Option price"),
        (8.1, "ANN-IV"),
        (9.2, "Implied\nvolatility"),
    ]
    for x, label in boxes2:
        color = "#DAE8FC" if "ANN" in label else "#D5E8D4"
        bbox = dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="black")
        ax.text(x, 0.5, label, fontsize=9, va="center", ha="center", bbox=bbox)
    for i in range(len(boxes2) - 1):
        x1 = boxes2[i][0] + 0.45
        x2 = boxes2[i + 1][0] - 0.45
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5),
                     arrowprops=dict(arrowstyle="->", color="black"))

    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.close()
