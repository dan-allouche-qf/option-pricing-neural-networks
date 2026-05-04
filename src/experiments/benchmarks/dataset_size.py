"""Experiment 8: Dataset size study (Table 3, Figure 6).

7 cases x 5 seeds x 200 epochs on IV data.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from black_scholes import generate_iv_data
from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics
from utils import (
    get_device, make_dataloader, plot_dataset_size_study,
    DATA_DIR, RESULTS_DIR,
)


def main():
    print("=" * 60)
    print("EXPERIMENT 8: Dataset Size Study (Table 3, Figure 6)")
    print("=" * 60)

    device = get_device()

    baseline = 24_300
    multipliers = [1/8, 1/4, 1/2, 1, 2, 4, 8]
    n_seeds = 5
    epochs = 200

    cache = os.path.join(DATA_DIR, "iv_data_300k.npz")
    if os.path.exists(cache):
        print(f"\n[1] Loading cached IV data from {cache}")
        d = np.load(cache)
        inputs_scaled, outputs = d["scaled"], d["outputs"]
    else:
        print("\n[1] Generating large IV dataset (300K)...")
        _, inputs_scaled, outputs = generate_iv_data(n_samples=300_000, seed=42)
        np.savez(cache, scaled=inputs_scaled, outputs=outputs)
    print(f"    Total samples: {len(inputs_scaled):,}")

    n_test = int(baseline * 8 * 0.1)
    X_test = inputs_scaled[-n_test:]
    y_test = outputs[-n_test:]
    X_pool = inputs_scaled[:-n_test]
    y_pool = outputs[:-n_test]

    cases = list(range(len(multipliers)))
    train_means, train_stds = [], []
    test_means, test_stds = [], []
    r2_means, r2_stds = [], []

    for idx, mult in enumerate(multipliers):
        n_train = int(baseline * mult)
        print(f"\n[Case {idx}] Training size = {n_train} (x{mult})")

        c_train, c_test, c_r2 = [], [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 100)
            torch.manual_seed(seed + 100)
            sel = rng.choice(len(X_pool), size=min(n_train, len(X_pool)), replace=False)
            X_tr, y_tr = X_pool[sel], y_pool[sel]

            train_loader = make_dataloader(X_tr, y_tr, batch_size=1024, shuffle=True)
            model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1).to(device)
            history = train_model(
                model, train_loader,
                epochs=epochs, lr_start=1e-3, lr_end=1e-5,
                device=device, print_every=9999, progress=False,
            )
            c_train.append(history["train_losses"][-1])
            y_pred = predict(model, X_test, device=device)
            m = compute_metrics(y_test, y_pred)
            c_test.append(m["MSE"])
            c_r2.append(m["R2"])
            print(f"    Seed {seed}: train MSE={history['train_losses'][-1]:.2e}, "
                  f"test MSE={m['MSE']:.2e}, R2={m['R2']:.6f}")

        train_means.append(float(np.mean(c_train)))
        train_stds.append(float(np.std(c_train)))
        test_means.append(float(np.mean(c_test)))
        test_stds.append(float(np.std(c_test)))
        r2_means.append(float(np.mean(c_r2)))
        r2_stds.append(float(np.std(c_r2)))

    print("\n" + "=" * 60)
    print("TABLE 3: Dataset size study results")
    print(f"{'Case':>6s} {'Train Size':>12s} {'Train MSE':>12s} {'Test MSE':>12s} {'R2 (%)':>10s}")
    print("-" * 55)
    for i, mult in enumerate(multipliers):
        n_train = int(baseline * mult)
        print(f"{i:6d} {n_train:12d} {train_means[i]:12.2e} {test_means[i]:12.2e} "
              f"{r2_means[i] * 100:10.4f}")

    print("\n[2] Generating Figure 6...")
    plot_dataset_size_study(
        cases, train_means, test_means, r2_means,
        train_std=train_stds, test_std=test_stds, r2_std=r2_stds,
        filename="fig6_dataset_size.png",
    )

    with open(os.path.join(RESULTS_DIR, "dataset_size_metrics.json"), "w") as f:
        json.dump({
            "cases": cases, "multipliers": multipliers,
            "train_mse_mean": train_means, "train_mse_std": train_stds,
            "test_mse_mean": test_means, "test_mse_std": test_stds,
            "r2_mean": r2_means, "r2_std": r2_stds,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT 8 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
