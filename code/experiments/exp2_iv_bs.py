"""Experiment 2: Reproduce IV-ANN results (Table 7, Figure 8 of the article).

Train an ANN to learn Black-Scholes implied volatility.
Compare performance with and without gradient-squash transformation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from black_scholes import generate_iv_data
from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics, print_metrics
from utils import (
    get_device, split_data, make_dataloader,
    plot_error_histogram, plot_scatter, plot_training_history
)


def train_iv_model(X, y, label, device, epochs=3000):
    """Train an IV model and return predictions + metrics."""
    X_train_all, y_train_all, X_test, y_test = split_data(X, y, train_ratio=0.9, seed=42)

    # Validation split from training set
    n_val = len(X_train_all) // 10
    X_val, y_val = X_train_all[-n_val:], y_train_all[-n_val:]
    X_train, y_train = X_train_all[:-n_val], y_train_all[:-n_val]

    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True, device=device)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False, device=device)

    model = PricingMLP(input_dim=X.shape[1], hidden_dim=400, n_hidden=4, output_dim=1)
    model = model.to(device)

    print(f"\n  Training IV-ANN ({label}) for {epochs} epochs...")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=epochs, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=500,
    )

    # Re-train on train+val combined (article p.9)
    print(f"  Re-training on train+val ({label})...")
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    trainval_loader = make_dataloader(X_trainval, y_trainval, batch_size=1024, shuffle=True, device=device)
    train_model(
        model, trainval_loader,
        epochs=200, lr_start=1e-5, lr_end=1e-5,
        device=device, print_every=999,
    )

    y_pred = predict(model, X_test, device=device)
    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, f"IV-ANN ({label})")

    return model, y_test, y_pred, metrics, history


def main():
    print("=" * 60)
    print("EXPERIMENT 2: IV-ANN with Gradient-Squash (Table 7, Figure 8)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- 1. Generate data ---
    print("\n[1] Generating IV data (1M samples, LHS)...")
    inputs_raw, inputs_scaled, outputs = generate_iv_data(n_samples=1_000_000, seed=42)
    print(f"    Raw input shape: {inputs_raw.shape}")
    print(f"    Scaled input shape: {inputs_scaled.shape}")
    print(f"    Output shape: {outputs.shape}")

    # --- 2. Train WITHOUT gradient-squash ---
    print("\n[2] WITHOUT gradient-squash (V/K -> sigma*):")
    _, y_test_raw, y_pred_raw, m_raw, _ = train_iv_model(
        inputs_raw, outputs, "without scaling", device, epochs=1000  # baseline, fewer epochs sufficient
    )

    # --- 3. Train WITH gradient-squash ---
    print("\n[3] WITH gradient-squash (log(V_hat/K) -> sigma*):")
    model_scaled, y_test_sc, y_pred_sc, m_scaled, history = train_iv_model(
        inputs_scaled, outputs, "with scaling", device, epochs=3000
    )

    # --- 4. Print comparison (Table 7) ---
    print("\n" + "=" * 60)
    print("TABLE 7 COMPARISON:")
    print(f"{'':30s} {'MSE':>12s} {'MAE':>12s} {'MAPE':>12s} {'R2':>12s}")
    print("-" * 78)
    print(f"{'Without scaling':30s} {m_raw['MSE']:12.2e} {m_raw['MAE']:12.2e} {m_raw['MAPE']:12.2e} {m_raw['R2']:12.5f}")
    print(f"{'With gradient-squash':30s} {m_scaled['MSE']:12.2e} {m_scaled['MAE']:12.2e} {m_scaled['MAPE']:12.2e} {m_scaled['R2']:12.7f}")
    print(f"\nMSE improvement factor: {m_raw['MSE'] / m_scaled['MSE']:.0f}x")

    # --- 5. Figures ---
    print("\n[4] Generating figures...")
    plot_scatter(y_test_sc, y_pred_sc,
                 title="IV-ANN: Predicted vs Actual (with scaling)",
                 filename="fig8a_iv_scatter.png")
    plot_error_histogram(y_test_sc, y_pred_sc,
                         title="IV-ANN Error Distribution (with scaling)",
                         filename="fig8b_iv_error.png")
    plot_training_history(history["train_losses"],
                          title="IV-ANN Training Loss (with scaling)",
                          filename="fig_iv_training_loss.png")

    # --- 6. Save model ---
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iv_ann_model.pt")
    torch.save(model_scaled.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
