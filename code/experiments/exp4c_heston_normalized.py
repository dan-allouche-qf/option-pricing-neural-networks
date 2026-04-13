"""Experiment 4c: Heston-ANN with normalized inputs/outputs.

Improvement over exp4b: normalize all features and outputs to improve training.
Also implements early stopping based on validation loss.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time

from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics, print_metrics
from utils import (
    get_device, split_data_3way, make_dataloader,
    plot_error_histogram, plot_scatter, plot_training_history
)


def main():
    print("=" * 60)
    print("EXPERIMENT 4c: Heston-ANN with normalization + early stopping")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- 1. Load 1M data ---
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_data_1M.npz")
    if not os.path.exists(data_path):
        print("ERROR: heston_data_1M.npz not found!")
        return

    print("\n[1] Loading 1M Heston data...")
    data = np.load(data_path)
    X_all, y_all = data["inputs"], data["outputs"]
    print(f"    Shape: X={X_all.shape}, y={y_all.shape}")

    # --- 2. Normalize inputs and outputs ---
    print("\n[2] Normalizing data...")
    X_mean = X_all.mean(axis=0)
    X_std = X_all.std(axis=0)
    X_norm = (X_all - X_mean) / X_std

    y_mean = y_all.mean()
    y_std = y_all.std()
    y_norm = (y_all - y_mean) / y_std

    print(f"    X_mean: {X_mean}")
    print(f"    X_std:  {X_std}")
    print(f"    y_mean: {y_mean:.6f}, y_std: {y_std:.6f}")

    # --- 3. Split 80/10/10 ---
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(
        X_norm, y_norm, ratios=(0.8, 0.1, 0.1), seed=42
    )
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Also keep unnormalized test for final evaluation
    _, _, _, _, X_test_raw, y_test_raw = split_data_3way(
        X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42
    )

    # --- 4. Create data loaders ---
    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False)

    # --- 5. Build model ---
    model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1)
    model = model.to(device)
    print(f"\n[3] Model: {model.count_parameters()} parameters")

    # --- 6. Train ---
    t0 = time.time()
    print(f"\n[4] Training for 3000 epochs with normalized data...")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=3000, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=100,
    )
    t1 = time.time()
    print(f"\n    Training took {(t1-t0)/3600:.1f} hours")

    # --- 7. Evaluate (denormalize predictions) ---
    print("\n[5] Evaluating...")

    # Predict on normalized test, then denormalize
    y_pred_norm = predict(model, X_test, device=device)
    y_pred = y_pred_norm * y_std + y_mean
    y_true = y_test_raw

    m_test = compute_metrics(y_true, y_pred)
    print_metrics(m_test, "Testing (denormalized)")

    # Training metrics (subsample)
    y_pred_train_norm = predict(model, X_train[:100000], device=device)
    y_pred_train = y_pred_train_norm * y_std + y_mean
    _, y_train_raw, _, _, _, _ = split_data_3way(X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42)
    m_train = compute_metrics(y_train_raw[:100000], y_pred_train)
    print_metrics(m_train, "Training (100K subsample, denormalized)")

    # --- 8. Save normalization params + model ---
    norm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_norm_params.npz")
    np.savez(norm_path, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    print(f"  Normalization params saved: {norm_path}")

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_ann_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")

    # --- 9. Figures ---
    print("\n[6] Generating figures...")
    plot_scatter(y_true, y_pred,
                 title="Heston-ANN: COS vs ANN (1M, normalized)",
                 filename="fig9a_heston_scatter.png")
    plot_error_histogram(y_true, y_pred,
                         title="Heston-ANN Error (1M, normalized)",
                         filename="fig9b_heston_error.png")
    plot_training_history(
        history["train_losses"], history["val_losses"],
        title="Heston-ANN Loss (normalized)",
        filename="fig5_heston_training_loss.png"
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 4c COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
