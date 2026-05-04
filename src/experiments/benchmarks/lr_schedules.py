"""Experiment 7: Compare LR schedules on Heston (Figures 4-5).

Three runs: Constant, Decay (step, MultiStepLR), Cyclical.
Uses the same 1M Heston dataset from exp4.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from heston import generate_heston_data
from model import PricingMLP
from train import train_model
from utils import (
    get_device, set_seed, split_data_3way, make_dataloader,
    plot_lr_schedule_comparison, plot_training_history_sidebyside,
    DATA_DIR, RESULTS_DIR,
)


def main():
    print("=" * 60)
    print("EXPERIMENT 7: LR Schedule Comparison (Figures 4-5)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    cache = os.path.join(DATA_DIR, "heston_data_1M.npz")
    if os.path.exists(cache):
        print(f"\n[1] Loading cached Heston data from {cache}")
        d = np.load(cache)
        X_all, y_all = d["inputs"], d["outputs"]
    else:
        print("\n[1] Generating 1M Heston samples...")
        X_all, y_all = generate_heston_data(n_samples=1_000_000, seed=42)
        np.savez(cache, inputs=X_all, outputs=y_all)

    print(f"    Using {len(X_all):,} samples")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(
        X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42
    )
    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False)

    epochs = 3000
    schedules = ["constant", "decay", "cyclical"]
    labels = {"constant": "Constant LR", "decay": "Decay LR", "cyclical": "Cyclical LR"}
    histories = {}

    for sched in schedules:
        print(f"\n[2] Training with {labels[sched]} for {epochs} epochs...")
        # Re-seed so all three schedules start from identical Glorot init
        # (otherwise the comparison conflates schedule effect with init noise).
        set_seed(42)
        model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1).to(device)
        history = train_model(
            model, train_loader, val_loader=val_loader,
            epochs=epochs, lr_start=1e-3, lr_end=1e-5,
            schedule=sched,
            device=device, print_every=100, log_prefix=f"[{labels[sched]}] ",
        )
        histories[sched] = history

    print("\n[3] Generating Figure 4...")
    losses_dict = {labels[s]: histories[s]["train_losses"] for s in schedules}
    plot_lr_schedule_comparison(losses_dict, filename="fig4_lr_schedules.png")

    print("\n[4] Generating Figure 5...")
    plot_training_history_sidebyside(
        histories["decay"], histories["cyclical"],
        titles=["DecayLR", "CLR"],
        filename="fig5_decay_vs_clr.png",
    )

    with open(os.path.join(RESULTS_DIR, "lr_schedules_history.json"), "w") as f:
        json.dump({s: {
            "train_losses": histories[s]["train_losses"],
            "val_losses": histories[s]["val_losses"],
            "lrs": histories[s]["lrs"],
        } for s in schedules}, f)

    print("\n" + "=" * 60)
    print("EXPERIMENT 7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
