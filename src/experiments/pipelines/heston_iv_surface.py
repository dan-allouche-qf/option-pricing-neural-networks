"""Experiment 5: Heston IV surface via Heston-ANN + IV-ANN (Table 11, Figures 11-12).

Uses the pre-trained Heston-ANN (exp4) and IV-ANN (exp2).
Reference: COS + Brent.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

from heston import heston_call_price
from implied_vol import brent_iv
from model import PricingMLP
from train import predict
from metrics import compute_metrics, print_metrics
from utils import get_device, FIGURES_DIR, MODELS_DIR, RESULTS_DIR


def compute_iv_surface_reference(m_range, tau_range, r, rho, kappa, vbar, gamma, v0, K=1.0):
    mm, tt = np.meshgrid(m_range, tau_range)
    iv = np.zeros_like(mm)
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            m, tau = mm[i, j], tt[i, j]
            S = m * K
            price = heston_call_price(S, K, tau, r, kappa, vbar, gamma, rho, v0)
            if np.isfinite(price) and price > 1e-8:
                sig = brent_iv(price, S, K, tau, r, a=0.001, b=2.0)
                iv[i, j] = sig if np.isfinite(sig) else np.nan
            else:
                iv[i, j] = np.nan
    return mm, tt, iv


def compute_iv_surface_ann(m_range, tau_range, r, rho, kappa, vbar, gamma, v0,
                            heston_model, iv_model, device, K=1.0):
    mm, tt = np.meshgrid(m_range, tau_range)
    n = mm.size
    hin = np.column_stack([
        mm.flatten(), tt.flatten(),
        np.full(n, r), np.full(n, rho), np.full(n, kappa),
        np.full(n, vbar), np.full(n, gamma), np.full(n, v0),
    ])
    prices = predict(heston_model, hin, device=device).flatten()

    S = mm.flatten() * K
    intrinsic = np.maximum(S - K * np.exp(-r * tt.flatten()), 0.0)
    V_hat = np.maximum(prices - intrinsic, 1e-12)
    iv_in = np.column_stack([mm.flatten(), tt.flatten(), np.full(n, r), np.log(V_hat)])
    iv_pred = predict(iv_model, iv_in, device=device).flatten()
    return mm, tt, iv_pred.reshape(mm.shape)


def evaluate_case(name, m_bounds, tau_bounds, hparams, heston_model, iv_model, device, n_test=5000):
    rho, kappa, gamma, vbar, v0, r = hparams
    K = 1.0
    sampler = LatinHypercube(d=2, seed=123)
    s = sampler.random(n=n_test)
    m_t = s[:, 0] * (m_bounds[1] - m_bounds[0]) + m_bounds[0]
    tau_t = s[:, 1] * (tau_bounds[1] - tau_bounds[0]) + tau_bounds[0]

    iv_true = np.full(n_test, np.nan)
    for i in range(n_test):
        S = m_t[i] * K
        price = heston_call_price(S, K, tau_t[i], r, kappa, vbar, gamma, rho, v0)
        if np.isfinite(price) and price > 1e-8:
            sig = brent_iv(price, S, K, tau_t[i], r, a=0.001, b=2.0)
            iv_true[i] = sig if np.isfinite(sig) else np.nan

    hin = np.column_stack([
        m_t, tau_t,
        np.full(n_test, r), np.full(n_test, rho), np.full(n_test, kappa),
        np.full(n_test, vbar), np.full(n_test, gamma), np.full(n_test, v0),
    ])
    p_ann = predict(heston_model, hin, device=device).flatten()

    S = m_t * K
    intrinsic = np.maximum(S - K * np.exp(-r * tau_t), 0.0)
    V_hat = np.maximum(p_ann - intrinsic, 1e-12)
    iv_in = np.column_stack([m_t, tau_t, np.full(n_test, r), np.log(V_hat)])
    iv_pred = predict(iv_model, iv_in, device=device).flatten()

    mask = np.isfinite(iv_true) & np.isfinite(iv_pred) & (iv_true > 0)
    metrics = compute_metrics(iv_true[mask], iv_pred[mask])
    print(f"\n  {name}: m in [{m_bounds[0]}, {m_bounds[1]}], tau in [{tau_bounds[0]}, {tau_bounds[1]}]")
    print(f"    RMSE = {metrics['RMSE']:.2e}, MAE = {metrics['MAE']:.2e}, "
          f"MAPE = {metrics['MAPE']:.2e}, R2 = {metrics['R2']:.6f}")
    return metrics, iv_pred[mask] - iv_true[mask]


def main():
    print("=" * 60)
    print("EXPERIMENT 5: Heston IV Surface (Table 11, Figures 11-12)")
    print("=" * 60)

    device = get_device()

    heston_path = os.path.join(MODELS_DIR, "heston_ann.pt")
    iv_path = os.path.join(MODELS_DIR, "iv_ann.pt")
    if not os.path.exists(heston_path) or not os.path.exists(iv_path):
        print("ERROR: Models not found. Run exp2 and exp4 first.")
        return

    heston_model = PricingMLP(input_dim=8, hidden_dim=400, n_hidden=4, output_dim=1)
    heston_model.load_state_dict(torch.load(heston_path, weights_only=True, map_location="cpu"))
    heston_model = heston_model.to(device)
    heston_model.eval()

    iv_model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1)
    iv_model.load_state_dict(torch.load(iv_path, weights_only=True, map_location="cpu"))
    iv_model = iv_model.to(device)
    iv_model.eval()

    # Heston params for smile (article p.18)
    rho, kappa, gamma, vbar, v0, r = -0.05, 1.5, 0.3, 0.1, 0.1, 0.02
    hparams = (rho, kappa, gamma, vbar, v0, r)

    m1, e1 = evaluate_case("Case 1", (0.7, 1.3), (0.3, 1.1), hparams, heston_model, iv_model, device)
    m2, e2 = evaluate_case("Case 2", (0.75, 1.25), (0.4, 1.0), hparams, heston_model, iv_model, device)

    print("\n" + "=" * 60)
    print("TABLE 11: Heston-ANN + IV-ANN performance")
    print(f"{'Case':>30s} {'RMSE':>12s} {'MAE':>12s} {'MAPE':>12s} {'R2':>10s}")
    print("-" * 78)
    print(f"{'Case 1 (wide)':>30s} {m1['RMSE']:12.2e} {m1['MAE']:12.2e} "
          f"{m1['MAPE']:12.2e} {m1['R2']:10.6f}")
    print(f"{'Case 2 (narrow)':>30s} {m2['RMSE']:12.2e} {m2['MAE']:12.2e} "
          f"{m2['MAPE']:12.2e} {m2['R2']:10.6f}")

    # Figure 11
    print("\n[3] Generating Figure 11...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, errors, title in [(ax1, e1, "Case 1"), (ax2, e2, "Case 2")]:
        ax.hist(errors, bins=60, density=True, alpha=0.7, color="#2980B9", edgecolor="white")
        ax.set_xlabel("diff", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"({title}): The error distribution", fontsize=12)
        ax_cdf = ax.twinx()
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax_cdf.plot(sorted_err, cdf, color="#E67E22", linewidth=2)
        ax_cdf.set_ylabel("Distribution", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig11_case_errors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    # IV surface
    m_range = np.linspace(0.7, 1.3, 30)
    tau_range = np.linspace(0.5, 1.0, 20)
    print("\n[4] Computing reference IV surface (COS + Brent)...")
    mm_ref, tt_ref, iv_ref = compute_iv_surface_reference(m_range, tau_range, r, rho, kappa, vbar, gamma, v0)
    print("\n[5] Computing ANN IV surface (Heston-ANN + IV-ANN)...")
    mm_ann, tt_ann, iv_ann = compute_iv_surface_ann(m_range, tau_range, r, rho, kappa, vbar, gamma, v0,
                                                     heston_model, iv_model, device)

    mask = np.isfinite(iv_ref) & np.isfinite(iv_ann)
    surface_metrics = None
    if mask.sum() > 0:
        surface_metrics = compute_metrics(iv_ref[mask], iv_ann[mask])
        print_metrics(surface_metrics, "Heston-ANN + IV-ANN vs COS + Brent (surface)")
        diff = iv_ann - iv_ref
        print(f"  Max deviation: {np.nanmax(np.abs(diff)):.6f}")

    print("\n[6] Generating Figure 12...")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(mm_ann, tt_ann, iv_ann, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Moneyness (m)", fontsize=11)
    ax.set_ylabel(r"Time to maturity ($\tau$)", fontsize=11)
    ax.set_zlabel(r"Implied volatility ($\sigma^*$)", fontsize=11)
    ax.set_title("Implied volatility(IV-ANN)", fontsize=13)
    path = os.path.join(FIGURES_DIR, "fig12a_iv_surface.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    diff = iv_ann - iv_ref
    c = ax.pcolormesh(mm_ref, tt_ref, diff, cmap="RdBu_r", shading="auto")
    plt.colorbar(c, ax=ax, label="IV difference")
    ax.set_xlabel("Moneyness (m)", fontsize=11)
    ax.set_ylabel(r"Time to maturity ($\tau$)", fontsize=11)
    ax.set_title("implied volatility difference", fontsize=13)
    path = os.path.join(FIGURES_DIR, "fig12b_iv_difference.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    with open(os.path.join(RESULTS_DIR, "heston_iv_surface_metrics.json"), "w") as f:
        out = {"case1": m1, "case2": m2}
        if surface_metrics is not None:
            out["surface"] = surface_metrics
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
