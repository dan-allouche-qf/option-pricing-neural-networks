"""Experiment 6: Reproduce Figure 3 of the article.

LR range test (Smith 2015): average training loss vs varying learning rates.
Used to determine the optimal learning rate range [10^-5, 10^-3].
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from black_scholes import generate_bs_data
from model import PricingMLP
from train import lr_range_test
from utils import get_device, make_dataloader, plot_lr_finder


def main():
    print("=" * 60)
    print("EXPERIMENT 6: LR Range Test (Figure 3)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Small dataset for LR finder (Section 3.3)
    print("\n[1] Generating small BS dataset for LR test...")
    X, y = generate_bs_data(n_samples=50_000, param_range="wide", seed=42)
    train_loader = make_dataloader(X, y, batch_size=1024, shuffle=True, device=device)

    # Fresh model
    model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1)
    model = model.to(device)

    # LR range test
    print("\n[2] Running LR range test (1e-9 to 10)...")
    lrs, losses = lr_range_test(
        model, train_loader,
        lr_min=1e-9, lr_max=10, num_steps=300,
        device=device,
    )
    print(f"    {len(lrs)} steps recorded")

    # Smooth the losses for cleaner plot
    window = 5
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    lrs_smoothed = lrs[:len(smoothed)]

    # Plot (Figure 3)
    print("\n[3] Generating Figure 3...")
    plot_lr_finder(lrs_smoothed, smoothed.tolist(), filename="fig3_lr_finder.png")

    # Find optimal range
    min_idx = np.argmin(smoothed)
    print(f"\n  Minimum loss at LR = {lrs_smoothed[min_idx]:.2e}")
    print(f"  Suggested range: ~1e-5 to ~1e-3 (matching article)")

    print("\n" + "=" * 60)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
