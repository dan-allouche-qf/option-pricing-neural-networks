"""Metrics for evaluating ANN pricing models (Eq. 20-21 of the article)."""

import torch
import numpy as np


def compute_metrics(y_true, y_pred):
    """Compute MSE, RMSE, MAE, MAPE, R2 between true and predicted values."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    diff = y_true - y_pred

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    mask = np.abs(y_true) > 1e-10
    mape = float(np.mean(np.abs(diff[mask]) / np.abs(y_true[mask]))) if mask.sum() > 0 else float("nan")

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def print_metrics(metrics, label=""):
    """Pretty-print metrics dict."""
    if label:
        print(f"\n--- {label} ---")
    for k, v in metrics.items():
        if k == "R2":
            print(f"  {k:>6s} = {v:.7f}")
        else:
            print(f"  {k:>6s} = {v:.6e}")
