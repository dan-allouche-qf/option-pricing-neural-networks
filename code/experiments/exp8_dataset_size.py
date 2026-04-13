"""Experiment 8: Reproduce Figure 6 and Table 3 of the article.

Dataset size study: train IV-ANN with varying data sizes (Cases 0-6).
Baseline = 24,300 samples. Multipliers: [1/8, 1/4, 1/2, 1, 2, 4, 8].
5 random seeds per case, 200 epochs each.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from black_scholes import generate_iv_data
from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics
from utils import get_device, split_data, make_dataloader, plot_dataset_size_study


def main():
    print("=" * 60)
    print("EXPERIMENT 8: Dataset Size Study (Table 3, Figure 6)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Table 3: baseline = 24300, cases multiply by [1/8, 1/4, 1/2, 1, 2, 4, 8]
    baseline = 24_300
    multipliers = [1/8, 1/4, 1/2, 1, 2, 4, 8]
    n_seeds = 5
    epochs = 200

    # Generate a large IV dataset to subsample from
    print("\n[1] Generating large IV dataset...")
    _, inputs_scaled, outputs = generate_iv_data(n_samples=300_000, seed=42)
    print(f"    Total samples: {len(inputs_scaled)}")

    # Fixed test set (10% of baseline*8 = last ~19440 samples)
    n_test = int(baseline * 8 * 0.1)
    X_test = inputs_scaled[-n_test:]
    y_test = outputs[-n_test:]
    X_pool = inputs_scaled[:-n_test]
    y_pool = outputs[:-n_test]

    cases = list(range(len(multipliers)))
    train_mse_means, train_mse_stds = [], []
    test_mse_means, test_mse_stds = [], []
    r2_means, r2_stds = [], []

    for case_idx, mult in enumerate(multipliers):
        n_train = int(baseline * mult)
        print(f"\n[Case {case_idx}] Training size = {n_train} (x{mult})")

        case_train_mse = []
        case_test_mse = []
        case_r2 = []

        for seed in range(n_seeds):
            # Subsample training data
            rng = np.random.default_rng(seed + 100)
            idx = rng.choice(len(X_pool), size=min(n_train, len(X_pool)), replace=False)
            X_train = X_pool[idx]
            y_train = y_pool[idx]

            train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True, device=device)

            model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1)
            model = model.to(device)

            history = train_model(
                model, train_loader,
                epochs=epochs, lr_start=1e-3, lr_end=1e-5,
                device=device, print_every=9999,  # silent
            )

            # Training MSE (last epoch)
            case_train_mse.append(history["train_losses"][-1])

            # Test metrics
            y_pred = predict(model, X_test, device=device)
            metrics = compute_metrics(y_test, y_pred)
            case_test_mse.append(metrics["MSE"])
            case_r2.append(metrics["R2"])

            print(f"    Seed {seed}: train MSE={history['train_losses'][-1]:.2e}, "
                  f"test MSE={metrics['MSE']:.2e}, R2={metrics['R2']:.6f}")

        train_mse_means.append(np.mean(case_train_mse))
        train_mse_stds.append(np.std(case_train_mse))
        test_mse_means.append(np.mean(case_test_mse))
        test_mse_stds.append(np.std(case_test_mse))
        r2_means.append(np.mean(case_r2))
        r2_stds.append(np.std(case_r2))

    # --- Print Table 3 ---
    print("\n" + "=" * 60)
    print("TABLE 3: Dataset size study results")
    print(f"{'Case':>6s} {'Train Size':>12s} {'Train MSE':>12s} {'Test MSE':>12s} {'R2 (%)':>10s}")
    print("-" * 55)
    for i, mult in enumerate(multipliers):
        n_train = int(baseline * mult)
        print(f"{i:6d} {n_train:12d} {train_mse_means[i]:12.2e} "
              f"{test_mse_means[i]:12.2e} {r2_means[i]*100:10.4f}")

    # --- Figure 6 ---
    print("\n[2] Generating Figure 6...")
    plot_dataset_size_study(
        cases, train_mse_means, test_mse_means, r2_means,
        train_std=train_mse_stds, test_std=test_mse_stds, r2_std=r2_stds,
        filename="fig6_dataset_size.png",
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 8 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
