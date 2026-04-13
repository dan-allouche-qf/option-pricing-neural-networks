"""Experiment 4: Reproduce Heston-ANN results (Table 10, Figure 9 of the article).

Train an ANN to learn Heston option pricing via COS method.
Input: {m, tau, r, rho, kappa, vbar, gamma, v0}, Output: V (call price)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

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
    print("EXPERIMENT 4: Heston-ANN Pricing (Table 10, Figure 9)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- 1. Generate or load data ---
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_data.npz")

    if os.path.exists(data_path):
        print("\n[1] Loading cached Heston data...")
        data = np.load(data_path)
        X_all, y_all = data["inputs"], data["outputs"]
    else:
        print("\n[1] Generating Heston data (this takes a while)...")
        # First attempt (200K) -- see exp4d for final version
        n = 200_000
        X_all, y_all = generate_heston_data(n_samples=n, seed=42)
        np.savez(data_path, inputs=X_all, outputs=y_all)
        print(f"    Data saved to {data_path}")

    print(f"    Input shape: {X_all.shape}, Output shape: {y_all.shape}")

    # --- 2. Split 80/10/10 (article convention for Heston) ---
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(
        X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42
    )
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- 3. Create data loaders ---
    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True, device=device)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False, device=device)

    # --- 4. Build model (8 inputs for Heston) ---
    model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1)
    model = model.to(device)
    print(f"\n[2] Model: {model.count_parameters()} parameters")

    # --- 5. Train ---
    print(f"\n[3] Training for 3000 epochs...")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=3000, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=500,
    )

    # --- 6. Evaluate ---
    print("\n[4] Evaluating...")

    y_pred_train = predict(model, X_train, device=device)
    m_train = compute_metrics(y_train, y_pred_train)
    print_metrics(m_train, "Training")

    y_pred_test = predict(model, X_test, device=device)
    m_test = compute_metrics(y_test, y_pred_test)
    print_metrics(m_test, "Testing")

    # --- 7. Figures ---
    print("\n[5] Generating figures...")
    plot_scatter(y_test, y_pred_test,
                 title="Heston-ANN: COS vs ANN prices",
                 filename="fig9a_heston_scatter.png")
    plot_error_histogram(y_test, y_pred_test,
                         title="Heston-ANN Error Distribution",
                         filename="fig9b_heston_error.png")
    plot_training_history(
        history["train_losses"], history["val_losses"],
        title="Heston-ANN Training & Validation Loss",
        filename="fig5_heston_training_loss.png"
    )

    # --- 8. Save model ---
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_ann_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
