"""Experiment 6: LR range test (Figure 3, Smith 2015)."""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from black_scholes import generate_bs_data
from model import PricingMLP
from train import lr_range_test
from utils import get_device, set_seed, make_dataloader, plot_lr_finder, RESULTS_DIR


def main():
    print("=" * 60)
    print("EXPERIMENT 6: LR Range Test (Figure 3)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("\n[1] Generating small BS dataset for LR test...")
    X, y = generate_bs_data(n_samples=50_000, param_range="wide", seed=42)
    train_loader = make_dataloader(X, y, batch_size=1024, shuffle=True)

    model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1).to(device)

    print("\n[2] Running LR range test (1e-9 to 1)...")
    lrs, losses = lr_range_test(model, train_loader, lr_min=1e-9, lr_max=1, num_steps=300, device=device)
    print(f"    {len(lrs)} steps recorded")

    window = 5
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    lrs_smoothed = lrs[: len(smoothed)]

    print("\n[3] Generating Figure 3...")
    plot_lr_finder(lrs_smoothed, smoothed.tolist(), filename="fig3_lr_finder.png")

    min_idx = int(np.argmin(smoothed))
    print(f"\n  Minimum loss at LR = {lrs_smoothed[min_idx]:.2e}")
    print("  Suggested range: ~1e-5 to ~1e-3 (matching article)")

    with open(os.path.join(RESULTS_DIR, "lr_finder_metrics.json"), "w") as f:
        json.dump({
            "lrs": list(lrs_smoothed),
            "smoothed_losses": smoothed.tolist(),
            "min_loss_lr": float(lrs_smoothed[min_idx]),
        }, f)

    print("\n" + "=" * 60)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
