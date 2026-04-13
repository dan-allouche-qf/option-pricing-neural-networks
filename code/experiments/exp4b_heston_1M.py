"""Experiment 4b: Heston-ANN with 1M samples (faithful to article).

This is the full-scale version of exp4 using 1,000,000 samples as in the article.
Includes early stopping based on validation loss to prevent overfitting.
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
    print("EXPERIMENT 4b: Heston-ANN with 1M samples (article-faithful)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- 1. Load 1M data ---
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_data_1M.npz")
    if not os.path.exists(data_path):
        print("ERROR: heston_data_1M.npz not found! Generate it first.")
        return

    print("\n[1] Loading 1M Heston data...")
    data = np.load(data_path)
    X_all, y_all = data["inputs"], data["outputs"]
    print(f"    Input shape: {X_all.shape}, Output shape: {y_all.shape}")

    # --- 2. Split 80/10/10 ---
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(
        X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42
    )
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- 3. Create data loaders ---
    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False)

    # --- 4. Build model ---
    model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1)
    model = model.to(device)
    print(f"\n[2] Model: {model.count_parameters()} parameters")

    # --- 5. Train ---
    t0 = time.time()
    print(f"\n[3] Training for 3000 epochs (DecayLR 1e-3 -> 1e-5)...")
    print(f"    Estimated time: ~7 hours on CPU")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=3000, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=100,
    )
    t1 = time.time()
    print(f"\n    Training took {(t1-t0)/3600:.1f} hours")

    # --- 6. Evaluate ---
    print("\n[4] Evaluating...")

    y_pred_train = predict(model, X_train[:100000], device=device)  # subsample for speed
    m_train = compute_metrics(y_train[:100000], y_pred_train)
    print_metrics(m_train, "Training (100K subsample)")

    y_pred_test = predict(model, X_test, device=device)
    m_test = compute_metrics(y_test, y_pred_test)
    print_metrics(m_test, "Testing")

    # --- 7. Figures ---
    print("\n[5] Generating figures...")
    plot_scatter(y_test, y_pred_test,
                 title="Heston-ANN: COS vs ANN prices (1M training)",
                 filename="fig9a_heston_scatter.png")
    plot_error_histogram(y_test, y_pred_test,
                         title="Heston-ANN Error Distribution (1M training)",
                         filename="fig9b_heston_error.png")
    plot_training_history(
        history["train_losses"], history["val_losses"],
        title="Heston-ANN Training & Validation Loss (1M)",
        filename="fig5_heston_training_loss.png"
    )

    # --- 8. Save model ---
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_ann_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 4b COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
