"""Experiment 7: Reproduce Figures 4 and 5 of the article.

Compare three learning rate schedules: Constant, Decay, Cyclical.
Figure 4: overlay of training losses for all three.
Figure 5: side-by-side DecayLR vs CLR (training + validation).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from heston_cos import generate_heston_data
from model import PricingMLP
from train import train_model
from utils import (
    get_device, split_data_3way, make_dataloader,
    plot_lr_schedule_comparison, plot_training_history_sidebyside,
)


def main():
    print("=" * 60)
    print("EXPERIMENT 7: LR Schedule Comparison (Figures 4-5)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- 1. Load 1M Heston dataset (as in the article) ---
    data_path_1M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_data_1M.npz")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heston_data.npz")
    if os.path.exists(data_path_1M):
        print("\n[1] Loading 1M Heston data...")
        data = np.load(data_path_1M)
        X_all, y_all = data["inputs"], data["outputs"]
    elif os.path.exists(data_path):
        print("\n[1] Loading cached Heston data...")
        data = np.load(data_path)
        X_all, y_all = data["inputs"], data["outputs"]
    else:
        print("\n[1] Generating Heston data for LR comparison...")
        X_all, y_all = generate_heston_data(n_samples=1_000_000, seed=42)
    # Filter as per Table 9 output range
    mask = (y_all.flatten() > 1e-6) & (y_all.flatten() < 0.67)
    X_all, y_all = X_all[mask], y_all[mask]

    print(f"    Using {len(X_all)} samples")

    # Split 80/10/10
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(
        X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42
    )

    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True, device=device)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False, device=device)

    epochs = 3000
    histories = {}
    schedules = ["constant", "decay", "cyclical"]
    labels = {"constant": "Constant LR", "decay": "Decay LR", "cyclical": "Cyclical LR"}

    for sched in schedules:
        print(f"\n[2] Training with {labels[sched]} for {epochs} epochs...")

        model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1)
        model = model.to(device)

        history = train_model(
            model, train_loader, val_loader=val_loader,
            epochs=epochs, lr_start=1e-3, lr_end=1e-5,
            schedule=sched,
            device=device, print_every=1000,
        )
        histories[sched] = history

    # --- Figure 4: Comparison of 3 LR schedules ---
    print("\n[3] Generating Figure 4...")
    losses_dict = {labels[s]: histories[s]["train_losses"] for s in schedules}
    plot_lr_schedule_comparison(losses_dict, filename="fig4_lr_schedules.png")

    # --- Figure 5: Side-by-side DecayLR vs CLR ---
    print("\n[4] Generating Figure 5...")
    plot_training_history_sidebyside(
        histories["decay"], histories["cyclical"],
        titles=["DecayLR", "CLR"],
        filename="fig5_decay_vs_clr.png",
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
