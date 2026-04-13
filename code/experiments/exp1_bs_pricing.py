"""Experiment 1: Reproduce BS-ANN results (Table 5, Figure 7 of the article).

Train an ANN to learn the Black-Scholes pricing formula for European calls.
Input: {S0/K, tau, r, sigma}, Output: V/K
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from black_scholes import generate_bs_data
from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics, print_metrics
from utils import (
    get_device, split_data, make_dataloader,
    plot_error_histogram, plot_training_history
)


def main():
    print("=" * 60)
    print("EXPERIMENT 1: BS-ANN Pricing (Table 5, Figure 7)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- 1. Generate data ---
    print("\n[1] Generating 1,000,000 BS samples (wide range, LHS)...")
    X_all, y_all = generate_bs_data(n_samples=1_000_000, param_range="wide", seed=42)
    print(f"    Input shape: {X_all.shape}, Output shape: {y_all.shape}")

    # --- 2. Split 90/10, then further split train into train/val ---
    X_train_all, y_train_all, X_test_wide, y_test_wide = split_data(X_all, y_all, train_ratio=0.9, seed=42)
    # Validation split from training set (10% of train = 9% of total)
    n_val = len(X_train_all) // 10
    X_val, y_val = X_train_all[-n_val:], y_train_all[-n_val:]
    X_train, y_train = X_train_all[:-n_val], y_train_all[:-n_val]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test (wide): {len(X_test_wide)}")

    # Generate narrow test set
    X_narrow, y_narrow = generate_bs_data(n_samples=100_000, param_range="narrow", seed=123)
    print(f"    Test (narrow): {len(X_narrow)}")

    # --- 3. Create data loaders (pre-load on device to avoid per-batch transfers) ---
    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True, device=device)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False, device=device)

    # --- 4. Build model ---
    model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1)
    model = model.to(device)
    print(f"\n[2] Model: {model.count_parameters()} parameters")

    # --- 5. Train ---
    print(f"\n[3] Training for 3000 epochs (DecayLR 1e-3 -> 1e-5)...")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=3000, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=500,
    )

    # --- 6. Re-train on train+val (article p.9: "train on the whole data set") ---
    print("\n[4] Re-training on train+val combined (as per article p.9)...")
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    trainval_loader = make_dataloader(X_trainval, y_trainval, batch_size=1024, shuffle=True, device=device)
    # Short fine-tuning with low LR (model is already well-trained)
    train_model(
        model, trainval_loader,
        epochs=200, lr_start=1e-5, lr_end=1e-5,
        device=device, print_every=100,
    )

    # --- 7. Evaluate ---
    print("\n[5] Evaluating...")

    # Training set metrics
    y_pred_train = predict(model, X_train, device=device)
    m_train = compute_metrics(y_train, y_pred_train)
    print_metrics(m_train, "Training (wide)")

    # Test wide
    y_pred_wide = predict(model, X_test_wide, device=device)
    m_wide = compute_metrics(y_test_wide, y_pred_wide)
    print_metrics(m_wide, "Testing (wide)")

    # Test narrow
    y_pred_narrow = predict(model, X_narrow, device=device)
    m_narrow = compute_metrics(y_narrow, y_pred_narrow)
    print_metrics(m_narrow, "Testing (narrow)")

    # --- 8. Figures ---
    print("\n[6] Generating figures...")
    plot_error_histogram(y_test_wide, y_pred_wide,
                         title="BS-ANN Error Distribution (wide set)",
                         filename="fig7a_bs_error_wide.png")
    plot_error_histogram(y_narrow, y_pred_narrow,
                         title="BS-ANN Error Distribution (narrow set)",
                         filename="fig7b_bs_error_narrow.png")
    plot_training_history(history["train_losses"], history["val_losses"],
                          title="BS-ANN Training & Validation Loss",
                          filename="fig5_bs_training_loss.png")

    # --- 9. Save model ---
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bs_ann_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
