"""Experiment 4: Heston-ANN (Table 10, Figure 9).

Faithful to the article (Section 4.4.1):
- 1,000,000 samples via LHS (Table 9)
- COS method with N=1500, L=50 (robust, no integration-range clamp)
- Split 80/10/10
- 3000 epochs, step-decay LR (Figure 4)
- Minimal filtering: prices finite and in [intrinsic, S]
- No normalization, no V<0.67 hack — those were workarounds for a COS bug
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from heston import generate_heston_data
from model import PricingMLP
from train import train_model, predict
from metrics import compute_metrics, print_metrics
from utils import (
    get_device, set_seed, split_data_3way, make_dataloader,
    plot_error_histogram, plot_scatter, plot_training_history,
    MODELS_DIR, RESULTS_DIR, DATA_DIR,
)


def main():
    print("=" * 60)
    print("EXPERIMENT 4: Heston-ANN (Table 10, Figure 9)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    cache = os.path.join(DATA_DIR, "heston_data_1M.npz")
    if os.path.exists(cache):
        print(f"\n[1] Loading cached Heston data from {cache}")
        d = np.load(cache)
        X_all, y_all = d["inputs"], d["outputs"]
    else:
        print("\n[1] Generating 1,000,000 Heston samples via robust COS method...")
        t0 = time.time()
        X_all, y_all = generate_heston_data(n_samples=1_000_000, seed=42)
        np.savez(cache, inputs=X_all, outputs=y_all)
        print(f"    Generation took {(time.time()-t0)/60:.1f} min")
    print(f"    Shape: X={X_all.shape}, y={y_all.shape}")
    print(f"    Price range: [{y_all.min():.4f}, {y_all.max():.4f}]")
    print(f"    Price stats: mean={y_all.mean():.4f}, std={y_all.std():.4f}")

    # Split 80/10/10 (Section 4.4.1)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_3way(
        X_all, y_all, ratios=(0.8, 0.1, 0.1), seed=42
    )
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False)

    model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1).to(device)
    print(f"\n[2] Model: {model.count_parameters():,} parameters")

    print("\n[3] Training for 3000 epochs (MultiStepLR 1e-3 -> 1e-4 -> 1e-5 at 1000, 2000)...")
    t0 = time.time()
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=3000, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=50, log_prefix="[Heston] ",
    )
    print(f"    Training took {(time.time()-t0)/3600:.2f} hours")

    # Evaluate
    print("\n[4] Evaluating...")
    y_pred_train = predict(model, X_train[:100_000], device=device)
    m_train = compute_metrics(y_train[:100_000], y_pred_train)
    print_metrics(m_train, "Training (100K subsample)")

    y_pred_test = predict(model, X_test, device=device)
    m_test = compute_metrics(y_test, y_pred_test)
    print_metrics(m_test, "Testing")

    # Table 10
    print("\n" + "=" * 60)
    print("TABLE 10: Heston-ANN performance")
    print(f"{'':12s} {'MSE':>12s} {'RMSE':>12s} {'MAE':>12s} {'MAPE':>12s} {'R2':>12s}")
    print("-" * 78)
    print(f"{'Training':12s} {m_train['MSE']:12.2e} {m_train['RMSE']:12.2e} "
          f"{m_train['MAE']:12.2e} {m_train['MAPE']:12.2e} {m_train['R2']:12.7f}")
    print(f"{'Testing':12s} {m_test['MSE']:12.2e} {m_test['RMSE']:12.2e} "
          f"{m_test['MAE']:12.2e} {m_test['MAPE']:12.2e} {m_test['R2']:12.7f}")

    # Figures
    print("\n[5] Generating figures...")
    plot_scatter(y_test, y_pred_test,
                 title="Heston-ANN: COS vs ANN prices (1M training)",
                 filename="fig9a_heston_scatter.png")
    plot_error_histogram(y_test, y_pred_test,
                         title="Heston-ANN Error Distribution (1M training)",
                         filename="fig9b_heston_error.png")
    plot_training_history(history["train_losses"], history["val_losses"],
                          title="Heston-ANN Training & Validation Loss (1M)",
                          filename="fig5_heston_training_loss.png")

    # Persist
    model_path = os.path.join(MODELS_DIR, "heston_ann.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    with open(os.path.join(RESULTS_DIR, "heston_ann_metrics.json"), "w") as f:
        json.dump({"train": m_train, "test": m_test}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "heston_ann_history.json"), "w") as f:
        json.dump({
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "lrs": history["lrs"],
        }, f)

    print("\n" + "=" * 60)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
