"""Experiment 2: IV-ANN with gradient-squash (Table 7, Figure 8).

Train one IV-ANN with raw input (baseline) and one with log(V_hat/K) (scaling).
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
from metrics import compute_metrics, print_metrics
from utils import (
    get_device, set_seed, split_data, make_dataloader,
    plot_error_histogram, plot_scatter, plot_training_history,
    MODELS_DIR, RESULTS_DIR, DATA_DIR,
)


def train_iv_model(X, y, label, device, epochs=3000):
    # Re-seed before each model so the two trainings (raw vs scaled) start
    # from the same Glorot init -- isolates the gradient-squash effect.
    set_seed(42)
    X_train_all, y_train_all, X_test, y_test = split_data(X, y, train_ratio=0.9, seed=42)
    n_val = len(X_train_all) // 10
    X_val, y_val = X_train_all[-n_val:], y_train_all[-n_val:]
    X_train, y_train = X_train_all[:-n_val], y_train_all[:-n_val]

    train_loader = make_dataloader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=2048, shuffle=False)

    model = PricingMLP(input_dim=X.shape[1], hidden_dim=400, n_hidden=4, output_dim=1).to(device)

    print(f"\n  Training IV-ANN ({label}) for {epochs} epochs...")
    history = train_model(
        model, train_loader, val_loader=val_loader,
        epochs=epochs, lr_start=1e-3, lr_end=1e-5,
        device=device, print_every=50, log_prefix=f"[IV-{label[:5]}] ",
    )

    # Re-train on train+val (article p.9)
    print(f"  Re-training on train+val ({label}, 200 epochs @ 1e-5)...")
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    tv_loader = make_dataloader(X_tv, y_tv, batch_size=1024, shuffle=True)
    train_model(
        model, tv_loader,
        epochs=200, lr_start=1e-5, lr_end=1e-5, schedule="constant",
        device=device, print_every=50, log_prefix=f"[IV-{label[:5]}-ft] ",
    )

    y_pred = predict(model, X_test, device=device)
    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, f"IV-ANN ({label})")

    return model, y_test, y_pred, metrics, history


def main():
    print("=" * 60)
    print("EXPERIMENT 2: IV-ANN with Gradient-Squash (Table 7, Figure 8)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    cache = os.path.join(DATA_DIR, "iv_data_1M.npz")
    if os.path.exists(cache):
        print(f"\n[1] Loading cached IV data from {cache}")
        d = np.load(cache)
        inputs_raw, inputs_scaled, outputs = d["raw"], d["scaled"], d["outputs"]
    else:
        print("\n[1] Generating IV data (1M samples, LHS)...")
        inputs_raw, inputs_scaled, outputs = generate_iv_data(n_samples=1_000_000, seed=42)
        np.savez(cache, raw=inputs_raw, scaled=inputs_scaled, outputs=outputs)
    print(f"    Raw input shape: {inputs_raw.shape}")
    print(f"    Scaled input shape: {inputs_scaled.shape}")

    # 2. Without gradient-squash
    print("\n[2] WITHOUT gradient-squash (V/K -> sigma*):")
    _, y_test_raw, y_pred_raw, m_raw, _ = train_iv_model(
        inputs_raw, outputs, "without scaling", device, epochs=1000
    )

    # 3. With gradient-squash
    print("\n[3] WITH gradient-squash (log(V_hat/K) -> sigma*):")
    model_scaled, y_test_sc, y_pred_sc, m_scaled, history = train_iv_model(
        inputs_scaled, outputs, "with scaling", device, epochs=3000
    )

    # 4. Table 7
    print("\n" + "=" * 60)
    print("TABLE 7 COMPARISON:")
    print(f"{'':30s} {'MSE':>12s} {'MAE':>12s} {'MAPE':>12s} {'R2':>12s}")
    print("-" * 78)
    print(f"{'Without scaling':30s} {m_raw['MSE']:12.2e} {m_raw['MAE']:12.2e} "
          f"{m_raw['MAPE']:12.2e} {m_raw['R2']:12.5f}")
    print(f"{'With gradient-squash':30s} {m_scaled['MSE']:12.2e} {m_scaled['MAE']:12.2e} "
          f"{m_scaled['MAPE']:12.2e} {m_scaled['R2']:12.7f}")
    print(f"\nMSE improvement factor: {m_raw['MSE'] / m_scaled['MSE']:.0f}x")

    # 5. Figures
    print("\n[4] Generating figures...")
    plot_scatter(y_test_sc, y_pred_sc,
                 title="IV-ANN: Predicted vs Actual (with scaling)",
                 filename="fig8a_iv_scatter.png")
    plot_error_histogram(y_test_sc, y_pred_sc,
                         title="IV-ANN Error Distribution (with scaling)",
                         filename="fig8b_iv_error.png")
    plot_training_history(history["train_losses"],
                          title="IV-ANN Training Loss (with scaling)",
                          filename="fig_iv_training_loss.png")

    # 6. Persist
    model_path = os.path.join(MODELS_DIR, "iv_ann.pt")
    torch.save(model_scaled.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    with open(os.path.join(RESULTS_DIR, "iv_ann_metrics.json"), "w") as f:
        json.dump({"without_scaling": m_raw, "with_scaling": m_scaled}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "iv_ann_history.json"), "w") as f:
        json.dump({
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "lrs": history["lrs"],
        }, f)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
