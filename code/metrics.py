"""Metrics for evaluating ANN pricing models (Eq. 20-21 of the article)."""

import torch
import numpy as np


def compute_metrics(y_true, y_pred):
    """Compute MSE, RMSE, MAE, MAPE, R2 between true and predicted values.

    Args:
        y_true: numpy array or torch tensor of true values
        y_pred: numpy array or torch tensor of predicted values

    Returns:
        dict with keys: MSE, RMSE, MAE, MAPE, R2
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    diff = y_true - y_pred

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    # MAPE: avoid division by zero
    mask = np.abs(y_true) > 1e-10
    mape = np.mean(np.abs(diff[mask]) / np.abs(y_true[mask])) if mask.sum() > 0 else np.nan

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    }


def print_metrics(metrics, label=""):
    """Pretty-print metrics dict."""
    if label:
        print(f"\n--- {label} ---")
    for k, v in metrics.items():
        if k == "R2":
            print(f"  {k:>6s} = {v:.7f}")
        else:
            print(f"  {k:>6s} = {v:.6e}")
