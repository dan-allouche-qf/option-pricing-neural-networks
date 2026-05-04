"""Experiment 1: BS-ANN pricing (Table 5, Figure 7 of the article).

Input: [S0/K, tau, r, sigma], Output: V/K. Wide range (Table 4).
3000 epochs with step-decay LR + 200 epochs re-training on train+val (p.9).
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from black_scholes import generate_bs_data
from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics, print_metrics
from utils import (
    get_device, set_seed, split_data, make_dataloader,
    plot_error_histogram, plot_training_history,
    MODELS_DIR, RESULTS_DIR, DATA_DIR,
)


def main():
    print("=" * 60)
    print("EXPERIMENT 1: BS-ANN Pricing (Table 5, Figure 7)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 1. Data
    cache = os.path.join(DATA_DIR, "bs_data_1M.npz")
    if os.path.exists(cache):
        print(f"\n[1] Loading cached BS data from {cache}")
        d = np.load(cache)
        X_all, y_all = d["inputs"], d["outputs"]
    else:
        print("\n[1] Generating 1,000,000 BS samples (wide range, LHS)...")
        X_all, y_all = generate_bs_data(n_samples=1_000_000, param_range="wide", seed=42)
        np.savez(cache, inputs=X_all, outputs=y_all)
    print(f"    Input shape: {X_all.shape}, Output shape: {y_all.shape}")

    # 2. 90/10 split, then val from train
    X_train_all, y_train_all, X_test_wide, y_test_wide = split_data(X_all, y_all, train_ratio=0.9, seed=42)
    n_val = len(X_train_all) // 10
    X_val, y_val = X_train_all[-n_val:], y_train_all[-n_val:]
    X_train, y_train = X_train_all[:-n_val], y_train_all[:-n_val]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test (wide): {len(X_test_wide)}")

    X_narrow, y_narrow = generate_bs_data(n_samples=100_000, param_range="narrow", seed=123)
    print(f"    Test (narrow): {len(X_narrow)}")

    # 3. Loaders
    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False)

    # 4. Model
    model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1).to(device)
    print(f"\n[2] Model: {model.count_parameters():,} parameters")

    # 5. Train (3000 epochs, step-decay LR)
    print("\n[3] Training for 3000 epochs (MultiStepLR 1e-3 -> 1e-4 -> 1e-5 at 1000, 2000)...")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=3000, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=50, log_prefix="[BS] ",
    )

    # 6. Re-train on train+val (article p.9)
    print("\n[4] Re-training on train+val combined (200 epochs @ 1e-5, p.9)...")
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    tv_loader = make_dataloader(X_tv, y_tv, batch_size=1024, shuffle=True)
    ft_history = train_model(
        model, tv_loader,
        epochs=200, lr_start=1e-5, lr_end=1e-5,
        schedule="constant",
        device=device, print_every=50, log_prefix="[BS-ft] ",
    )

    # 7. Evaluate
    print("\n[5] Evaluating...")
    y_pred_train = predict(model, X_train, device=device)
    m_train = compute_metrics(y_train, y_pred_train)
    print_metrics(m_train, "Training (wide)")

    y_pred_wide = predict(model, X_test_wide, device=device)
    m_wide = compute_metrics(y_test_wide, y_pred_wide)
    print_metrics(m_wide, "Testing (wide)")

    y_pred_narrow = predict(model, X_narrow, device=device)
    m_narrow = compute_metrics(y_narrow, y_pred_narrow)
    print_metrics(m_narrow, "Testing (narrow)")

    # 8. Figures
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

    # 9. Persist model + metrics + history
    model_path = os.path.join(MODELS_DIR, "bs_ann.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    metrics_path = os.path.join(RESULTS_DIR, "bs_ann_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"train_wide": m_train, "test_wide": m_wide, "test_narrow": m_narrow}, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    hist_path = os.path.join(RESULTS_DIR, "bs_ann_history.json")
    with open(hist_path, "w") as f:
        json.dump({
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "lrs": history["lrs"],
            "finetune_train_losses": ft_history["train_losses"],
        }, f)
    print(f"  History saved: {hist_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
