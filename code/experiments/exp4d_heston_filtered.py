"""Experiment 4d: Heston-ANN with 1M filtered samples.

Filters prices to V in (1e-6, 0.67) per Table 9, then retrains on train+val.
Final attempt after exp4 (200K) -> exp4b (1M unfiltered) -> exp4c (1M normalized).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time

from heston_cos import generate_heston_data
from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics, print_metrics
from utils import (
    get_device, split_data_3way, make_dataloader,
    plot_error_histogram, plot_scatter, plot_training_history
)


def main():
    print("=" * 60)
    print("EXPERIMENT 4d: Heston-ANN 1M filtered (best attempt)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- 1. Load or generate 1M data ---
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(code_dir, "heston_data_1M.npz")

    if os.path.exists(data_path):
        print("\n[1] Loading 1M Heston data...")
        data = np.load(data_path)
        X_all, y_all = data["inputs"], data["outputs"]
    else:
        print("\n[1] Generating 1M Heston samples...")
        X_all, y_all = generate_heston_data(n_samples=1_000_000, seed=42)
        np.savez(data_path, inputs=X_all, outputs=y_all)

    print(f"    Shape: X={X_all.shape}, y={y_all.shape}")

    # --- 2. Filter per Table 9: output in (0, 0.67) ---
    print("\n[2] Filtering data (Table 9: V in (1e-6, 0.67))...")
    mask = (y_all.flatten() > 1e-6) & (y_all.flatten() < 0.67)
    X_all, y_all = X_all[mask], y_all[mask]
    print(f"    After filtering: {len(X_all)} samples")

    # --- 3. Split 80/10/10 ---
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(
        X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42
    )
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- 4. DataLoaders ---
    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False)

    # --- 5. Model ---
    model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1)
    model = model.to(device)
    print(f"\n[3] Model: {model.count_parameters()} parameters")

    # --- 6. Training ---
    t0 = time.time()
    print(f"\n[4] Training 3000 epochs (DecayLR 1e-3 -> 1e-5)...")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=3000, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=100,
    )
    t1 = time.time()
    print(f"\n    Duration: {(t1-t0)/3600:.1f} hours")

    # --- 7. Re-training on train+val (article p.9) ---
    print("\n[5] Re-training on train+val...")
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    trainval_loader = make_dataloader(X_trainval, y_trainval, batch_size=1024, shuffle=True)
    train_model(
        model, trainval_loader,
        epochs=200, lr_start=1e-5, lr_end=1e-5,
        device=device, print_every=100,
    )

    # --- 8. Evaluation ---
    print("\n[6] Evaluation...")
    y_pred_train = predict(model, X_train[:100000], device=device)
    m_train = compute_metrics(y_train[:100000], y_pred_train)
    print_metrics(m_train, "Training (100K subsample)")

    y_pred_test = predict(model, X_test, device=device)
    m_test = compute_metrics(y_test, y_pred_test)
    print_metrics(m_test, "Test (filtered)")

    # --- 8. Figures ---
    print("\n[6] Generating figures...")
    plot_scatter(y_test, y_pred_test,
                 title="Heston-ANN (1M filtered)",
                 filename="fig9a_heston_scatter.png")
    plot_error_histogram(y_test, y_pred_test,
                         title="Heston-ANN Error (1M filtered)",
                         filename="fig9b_heston_error.png")
    plot_training_history(
        history["train_losses"], history["val_losses"],
        title="Heston-ANN Loss (1M filtered)",
        filename="fig5_heston_training_loss.png"
    )

    # --- 9. Save model ---
    model_path = os.path.join(code_dir, "heston_ann_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 4d COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
