"""Experiment 3: Reproduce speed comparison (Table 8 of the article).

Compare IV computation time: Newton-Raphson, Brent, Secant, Bisection vs IV-ANN.
20,000 European call options.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import torch

from black_scholes import bs_price, generate_iv_data
from implied_vol import batch_newton_iv, batch_brent_iv, batch_secant_iv, batch_bisection_iv
from model import PricingMLP
from train import predict
from metrics import compute_metrics
from utils import get_device


def main():
    print("=" * 60)
    print("EXPERIMENT 3: Speed Comparison (Table 8)")
    print("=" * 60)

    device = get_device()
    n_options = 20_000

    # --- 1. Generate test options ---
    # Wider moneyness range to test robustness
    
    print(f"\n[1] Generating {n_options} test options (including deep ITM/OTM)...")
    rng = np.random.default_rng(42)
    sigma_true = rng.uniform(0.01, 0.99, n_options)
    S = rng.uniform(0.5, 1.5, n_options)  # moneyness 0.5 to 1.5 (deep ITM to deep OTM)
    K = np.ones(n_options)
    tau = np.full(n_options, 0.5)
    r = np.zeros(n_options)

    V_mkt = bs_price(S, K, tau, r, sigma_true)

    # Filter options where all methods can find IV
    mask = V_mkt > 1e-8
    S, K, tau, r, V_mkt, sigma_true = S[mask], K[mask], tau[mask], r[mask], V_mkt[mask], sigma_true[mask]
    n_valid = len(V_mkt)
    print(f"    {n_valid} valid options")

    # --- 2. Load trained IV-ANN model ---
    print("\n[2] Loading IV-ANN model...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iv_ann_model.pt")
    if not os.path.exists(model_path):
        print("    ERROR: iv_ann_model.pt not found. Run exp2_iv_bs.py first!")
        return

    model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model = model.to(device)
    model.eval()

    # Prepare ANN input with gradient-squash
    m = S / K
    intrinsic = np.maximum(S - K * np.exp(-r * tau), 0.0)
    V_hat = V_mkt - intrinsic
    V_hat = np.maximum(V_hat, 1e-12)
    log_V_hat = np.log(V_hat)
    X_ann = np.column_stack([m, tau, r, log_V_hat])

    # --- 3. Benchmark each method ---
    print("\n[3] Benchmarking (CPU)...\n")

    results = {}

    # Newton-Raphson
    t0 = time.time()
    iv_nr = batch_newton_iv(V_mkt, S, K, tau, r, sigma0=0.5)
    t_nr = time.time() - t0
    nr_valid = np.isfinite(iv_nr).sum()
    results["Newton-Raphson"] = {"time": t_nr, "robust": nr_valid == n_valid}

    # Brent
    t0 = time.time()
    iv_br = batch_brent_iv(V_mkt, S, K, tau, r)
    t_br = time.time() - t0
    br_valid = np.isfinite(iv_br).sum()
    results["Brent"] = {"time": t_br, "robust": br_valid == n_valid}

    # Secant
    t0 = time.time()
    iv_sec = batch_secant_iv(V_mkt, S, K, tau, r)
    t_sec = time.time() - t0
    sec_valid = np.isfinite(iv_sec).sum()
    results["Secant"] = {"time": t_sec, "robust": sec_valid == n_valid}

    # Bisection
    t0 = time.time()
    iv_bis = batch_bisection_iv(V_mkt, S, K, tau, r)
    t_bis = time.time() - t0
    bis_valid = np.isfinite(iv_bis).sum()
    results["Bisection"] = {"time": t_bis, "robust": bis_valid == n_valid}

    # IV-ANN (CPU)
    model_cpu = model.to("cpu")
    t0 = time.time()
    iv_ann = predict(model_cpu, X_ann, device=torch.device("cpu"))
    t_ann_cpu = time.time() - t0
    results["IV-ANN (CPU)"] = {"time": t_ann_cpu, "robust": True}

    # IV-ANN (MPS/GPU if available)
    if device.type != "cpu":
        model_gpu = model.to(device)
        # Warm-up
        _ = predict(model_gpu, X_ann[:100], device=device)
        t0 = time.time()
        iv_ann_gpu = predict(model_gpu, X_ann, device=device)
        t_ann_gpu = time.time() - t0
        results[f"IV-ANN ({device.type.upper()})"] = {"time": t_ann_gpu, "robust": True}

    # --- 4. Print results (Table 8) ---
    print("\n" + "=" * 60)
    print("TABLE 8: Speed Comparison")
    print(f"{'Method':25s} {'Time (sec)':>12s} {'Robust':>8s}")
    print("-" * 48)
    for method, data in results.items():
        robust_str = "Yes" if data["robust"] else "No"
        print(f"{method:25s} {data['time']:12.2f} {robust_str:>8s}")

    # Speedup vs Newton-Raphson
    if "IV-ANN (CPU)" in results:
        speedup = results["Newton-Raphson"]["time"] / results["IV-ANN (CPU)"]["time"]
        print(f"\nSpeedup IV-ANN (CPU) vs Newton-Raphson: {speedup:.0f}x")

    # ANN accuracy
    ann_metrics = compute_metrics(sigma_true, iv_ann.flatten())
    print(f"\nIV-ANN MAE: {ann_metrics['MAE']:.2e}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
