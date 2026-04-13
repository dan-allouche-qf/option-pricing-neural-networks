"""Experiment 5: Reproduce Heston implied volatility surface (Table 11, Figures 11-12).

Two-ANN pipeline: Heston-ANN -> IV-ANN to compute BS implied volatility
from Heston model parameters.
Also generates the implied volatility surface (Figure 12) and
error distributions per case (Figure 11, Table 11).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

from heston_cos import heston_call_price
from black_scholes import bs_price
from implied_vol import brent_iv
from model import PricingMLP
from train import predict
from metrics import compute_metrics, print_metrics
from utils import get_device, FIGURES_DIR, plot_error_histogram


def compute_iv_surface_reference(m_range, tau_range, r, rho, kappa, vbar, gamma, v0, K=1.0):
    """Compute IV surface using COS + Brent (ground truth)."""
    mm, tt = np.meshgrid(m_range, tau_range)
    iv_surface = np.zeros_like(mm)

    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            m, tau = mm[i, j], tt[i, j]
            S = m * K
            price = heston_call_price(S, K, tau, r, kappa, vbar, gamma, rho, v0)
            if price > 1e-8:
                sigma_iv = brent_iv(price, S, K, tau, r, a=0.001, b=2.0)
                iv_surface[i, j] = sigma_iv if np.isfinite(sigma_iv) else np.nan
            else:
                iv_surface[i, j] = np.nan

    return mm, tt, iv_surface


def compute_iv_surface_ann(m_range, tau_range, r, rho, kappa, vbar, gamma, v0,
                           heston_model, iv_model, device, K=1.0):
    """Compute IV surface using Heston-ANN + IV-ANN pipeline."""
    mm, tt = np.meshgrid(m_range, tau_range)
    n = mm.size

    # Prepare Heston-ANN input: (m, tau, r, rho, kappa, vbar, gamma, v0)
    heston_input = np.column_stack([
        mm.flatten(),
        tt.flatten(),
        np.full(n, r),
        np.full(n, rho),
        np.full(n, kappa),
        np.full(n, vbar),
        np.full(n, gamma),
        np.full(n, v0),
    ])

    # Step 1: Heston-ANN predicts prices
    prices = predict(heston_model, heston_input, device=device).flatten()

    # Step 2: IV-ANN predicts implied volatilities
    S = mm.flatten() * K
    intrinsic = np.maximum(S - K * np.exp(-r * tt.flatten()), 0.0)
    V_hat = prices - intrinsic
    V_hat = np.maximum(V_hat, 1e-12)
    log_V_hat = np.log(V_hat)

    iv_input = np.column_stack([
        mm.flatten(),
        tt.flatten(),
        np.full(n, r),
        log_V_hat,
    ])

    iv_pred = predict(iv_model, iv_input, device=device).flatten()
    return mm, tt, iv_pred.reshape(mm.shape)


def evaluate_heston_iv_case(case_name, m_bounds, tau_bounds, heston_params,
                            heston_model, iv_model, device, n_test=5000):
    """Evaluate Heston-ANN + IV-ANN pipeline on a specific parameter sub-range (Table 11).

    Returns metrics dict and error array.
    """
    rho, kappa, gamma, vbar, v0, r = heston_params
    K = 1.0

    # Generate test points using LHS within the sub-range
    sampler = LatinHypercube(d=2, seed=123)
    samples = sampler.random(n=n_test)
    m_test = samples[:, 0] * (m_bounds[1] - m_bounds[0]) + m_bounds[0]
    tau_test = samples[:, 1] * (tau_bounds[1] - tau_bounds[0]) + tau_bounds[0]

    # Ground truth: COS + Brent
    iv_true = np.full(n_test, np.nan)
    for i in range(n_test):
        S = m_test[i] * K
        price = heston_call_price(S, K, tau_test[i], r, kappa, vbar, gamma, rho, v0)
        if price > 1e-8:
            sigma = brent_iv(price, S, K, tau_test[i], r, a=0.001, b=2.0)
            iv_true[i] = sigma if np.isfinite(sigma) else np.nan

    # ANN prediction: Heston-ANN + IV-ANN
    heston_input = np.column_stack([
        m_test,
        tau_test,
        np.full(n_test, r),
        np.full(n_test, rho),
        np.full(n_test, kappa),
        np.full(n_test, vbar),
        np.full(n_test, gamma),
        np.full(n_test, v0),
    ])
    prices_ann = predict(heston_model, heston_input, device=device).flatten()

    S = m_test * K
    intrinsic = np.maximum(S - K * np.exp(-r * tau_test), 0.0)
    V_hat = prices_ann - intrinsic
    V_hat = np.maximum(V_hat, 1e-12)
    log_V_hat = np.log(V_hat)

    iv_input = np.column_stack([m_test, tau_test, np.full(n_test, r), log_V_hat])
    iv_pred = predict(iv_model, iv_input, device=device).flatten()

    # Filter valid
    mask = np.isfinite(iv_true) & np.isfinite(iv_pred) & (iv_true > 0)
    metrics = compute_metrics(iv_true[mask], iv_pred[mask])

    print(f"\n  {case_name}:")
    print(f"    m in [{m_bounds[0]}, {m_bounds[1]}], tau in [{tau_bounds[0]}, {tau_bounds[1]}]")
    print(f"    RMSE = {metrics['RMSE']:.2e}, MAE = {metrics['MAE']:.2e}, "
          f"MAPE = {metrics['MAPE']:.2e}, R2 = {metrics['R2']:.6f}")

    errors = iv_pred[mask] - iv_true[mask]
    return metrics, errors


def main():
    print("=" * 60)
    print("EXPERIMENT 5: Heston IV Surface (Table 11, Figures 11-12)")
    print("=" * 60)

    device = get_device()
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- 1. Load models ---
    print("\n[1] Loading trained models...")
    heston_path = os.path.join(code_dir, "heston_ann_model.pt")
    iv_path = os.path.join(code_dir, "iv_ann_model.pt")

    if not os.path.exists(heston_path):
        print("  ERROR: heston_ann_model.pt not found. Run exp4 first!")
        return
    if not os.path.exists(iv_path):
        print("  ERROR: iv_ann_model.pt not found. Run exp2 first!")
        return

    heston_model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1)
    heston_model.load_state_dict(torch.load(heston_path, weights_only=True, map_location="cpu"))
    heston_model = heston_model.to(device)
    heston_model.eval()

    iv_model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1)
    iv_model.load_state_dict(torch.load(iv_path, weights_only=True, map_location="cpu"))
    iv_model = iv_model.to(device)
    iv_model.eval()

    # --- 2. Heston parameters for smile surface (article p.18) ---
    rho, kappa, gamma, vbar, v0, r = -0.05, 1.5, 0.3, 0.1, 0.1, 0.02
    heston_params = (rho, kappa, gamma, vbar, v0, r)

    # --- 3. Table 11: Two cases ---
    print("\n[2] Evaluating Table 11 cases...")

    # Case 1: tau in [0.3, 1.1], m in [0.7, 1.3]
    m1, e1 = evaluate_heston_iv_case(
        "Case 1", m_bounds=(0.7, 1.3), tau_bounds=(0.3, 1.1),
        heston_params=heston_params,
        heston_model=heston_model, iv_model=iv_model, device=device,
    )

    # Case 2: tau in [0.4, 1.0], m in [0.75, 1.25]
    m2, e2 = evaluate_heston_iv_case(
        "Case 2", m_bounds=(0.75, 1.25), tau_bounds=(0.4, 1.0),
        heston_params=heston_params,
        heston_model=heston_model, iv_model=iv_model, device=device,
    )

    # --- Print Table 11 ---
    print("\n" + "=" * 60)
    print("TABLE 11: Out-of-sample Heston-ANN + IV-ANN performance")
    print(f"{'Case':>30s} {'RMSE':>12s} {'MAE':>12s} {'MAPE':>12s} {'R2':>10s}")
    print("-" * 78)
    print(f"{'Case 1 (wide)':>30s} {m1['RMSE']:12.2e} {m1['MAE']:12.2e} "
          f"{m1['MAPE']:12.2e} {m1['R2']:10.6f}")
    print(f"{'Case 2 (narrow)':>30s} {m2['RMSE']:12.2e} {m2['MAE']:12.2e} "
          f"{m2['MAPE']:12.2e} {m2['R2']:10.6f}")

    # --- Figure 11: Error distributions ---
    print("\n[3] Generating Figure 11...")
    plot_error_histogram(np.zeros_like(e1), -e1,  # trick: true=0, pred=-error => error=e1
                         title="Case 1: The error distribution",
                         filename="fig11a_case1_error.png")
    # Simpler: directly plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, errors, title in [(ax1, e1, "Case 1"), (ax2, e2, "Case 2")]:
        ax_hist = ax
        ax_hist.hist(errors, bins=60, density=True, alpha=0.7, color="#2980B9", edgecolor="white")
        ax_hist.set_xlabel("diff", fontsize=12)
        ax_hist.set_ylabel("Density", fontsize=12)
        ax_hist.set_title(f"({title}): The error distribution", fontsize=12)

        ax_cdf = ax_hist.twinx()
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax_cdf.plot(sorted_err, cdf, color="#E67E22", linewidth=2)
        ax_cdf.set_ylabel("Distribution", fontsize=12)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig11_case_errors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    # --- 4. Compute reference surface (COS + Brent) ---
    m_range = np.linspace(0.7, 1.3, 30)
    tau_range = np.linspace(0.5, 1.0, 20)

    print("\n[4] Computing reference IV surface (COS + Brent)...")
    mm_ref, tt_ref, iv_ref = compute_iv_surface_reference(
        m_range, tau_range, r, rho, kappa, vbar, gamma, v0
    )

    # --- 5. Compute ANN surface ---
    print("\n[5] Computing ANN IV surface (Heston-ANN + IV-ANN)...")
    mm_ann, tt_ann, iv_ann = compute_iv_surface_ann(
        m_range, tau_range, r, rho, kappa, vbar, gamma, v0,
        heston_model, iv_model, device
    )

    # --- 6. Compare ---
    mask = np.isfinite(iv_ref) & np.isfinite(iv_ann)
    if mask.sum() > 0:
        metrics = compute_metrics(iv_ref[mask], iv_ann[mask])
        print_metrics(metrics, "Heston-ANN + IV-ANN vs COS + Brent (surface)")
        diff = iv_ann - iv_ref
        print(f"  Max deviation: {np.nanmax(np.abs(diff)):.6f}")

    # --- 7. Plot IV surface (Figure 12a) ---
    print("\n[6] Generating Figure 12...")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(mm_ann, tt_ann, iv_ann, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Moneyness (m)", fontsize=11)
    ax.set_ylabel("Time to maturity (τ)", fontsize=11)
    ax.set_zlabel("Implied volatility (σ*)", fontsize=11)
    ax.set_title("Implied volatility(IV-ANN)", fontsize=13)
    path = os.path.join(FIGURES_DIR, "fig12a_iv_surface.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    # Plot difference (Figure 12b)
    fig, ax = plt.subplots(figsize=(8, 6))
    diff = iv_ann - iv_ref
    c = ax.pcolormesh(mm_ref, tt_ref, diff, cmap="RdBu_r", shading="auto")
    plt.colorbar(c, ax=ax, label="IV difference")
    ax.set_xlabel("Moneyness (m)", fontsize=11)
    ax.set_ylabel("Time to maturity (τ)", fontsize=11)
    ax.set_title("implied volatility difference", fontsize=13)
    path = os.path.join(FIGURES_DIR, "fig12b_iv_difference.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
