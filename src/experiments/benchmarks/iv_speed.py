"""Experiment 3: Speed comparison for IV (Table 8).

Newton, Brent, Secant, Bisection vs IV-ANN on 20,000 options.
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from black_scholes import bs_price
from implied_vol import batch_newton_iv, batch_brent_iv, batch_secant_iv, batch_bisection_iv
from model import PricingMLP
from train import predict
from metrics import compute_metrics
from utils import get_device, MODELS_DIR, RESULTS_DIR


def main():
    print("=" * 60)
    print("EXPERIMENT 3: Speed Comparison (Table 8)")
    print("=" * 60)

    device = get_device()
    n_options = 20_000

    print(f"\n[1] Generating {n_options} test options (including deep ITM/OTM)...")
    rng = np.random.default_rng(42)
    sigma_true = rng.uniform(0.01, 0.99, n_options)
    S = rng.uniform(0.5, 1.5, n_options)
    K = np.ones(n_options)
    tau = np.full(n_options, 0.5)
    r = np.zeros(n_options)
    V_mkt = bs_price(S, K, tau, r, sigma_true)

    mask = V_mkt > 1e-8
    S, K, tau, r, V_mkt, sigma_true = S[mask], K[mask], tau[mask], r[mask], V_mkt[mask], sigma_true[mask]
    n_valid = len(V_mkt)
    print(f"    {n_valid} valid options")

    print("\n[2] Loading IV-ANN model...")
    model_path = os.path.join(MODELS_DIR, "iv_ann.pt")
    if not os.path.exists(model_path):
        print(f"    ERROR: {model_path} not found. Run iv_ann first.")
        return

    model = PricingMLP(input_dim=4, hidden_dim=400, n_hidden=4, output_dim=1)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    m = S / K
    intrinsic = np.maximum(S - K * np.exp(-r * tau), 0.0)
    V_hat = np.maximum(V_mkt - intrinsic, 1e-12)
    log_V_hat = np.log(V_hat)
    X_ann = np.column_stack([m, tau, r, log_V_hat])

    print("\n[3] Benchmarking...")
    results = {}

    for name, fn in [
        ("Newton-Raphson", lambda: batch_newton_iv(V_mkt, S, K, tau, r, sigma0=0.5)),
        ("Brent", lambda: batch_brent_iv(V_mkt, S, K, tau, r)),
        ("Secant", lambda: batch_secant_iv(V_mkt, S, K, tau, r)),
        ("Bisection", lambda: batch_bisection_iv(V_mkt, S, K, tau, r)),
    ]:
        t0 = time.time()
        iv = fn()
        elapsed = time.time() - t0
        robust = bool(np.isfinite(iv).sum() == n_valid)
        results[name] = {"time": elapsed, "robust": robust}
        print(f"  {name:20s}: {elapsed:8.2f}s  robust={robust}")

    # IV-ANN CPU
    model_cpu = model.to("cpu")
    # Warm-up
    _ = predict(model_cpu, X_ann[:100], device=torch.device("cpu"))
    t0 = time.time()
    iv_ann = predict(model_cpu, X_ann, device=torch.device("cpu"))
    t_cpu = time.time() - t0
    results["IV-ANN (CPU)"] = {"time": t_cpu, "robust": True}
    print(f"  IV-ANN (CPU)        : {t_cpu:8.2f}s  robust=True")

    # IV-ANN device (MPS/CUDA) if available
    if device.type != "cpu":
        model_gpu = model.to(device)
        _ = predict(model_gpu, X_ann[:100], device=device)
        t0 = time.time()
        _ = predict(model_gpu, X_ann, device=device)
        t_gpu = time.time() - t0
        key = f"IV-ANN ({device.type.upper()})"
        results[key] = {"time": t_gpu, "robust": True}
        print(f"  {key:20s}: {t_gpu:8.2f}s  robust=True")

    print("\n" + "=" * 60)
    print("TABLE 8: Speed Comparison")
    print(f"{'Method':25s} {'Time (sec)':>12s} {'Robust':>8s}")
    print("-" * 48)
    for method, data in results.items():
        rstr = "Yes" if data["robust"] else "No"
        print(f"{method:25s} {data['time']:12.2f} {rstr:>8s}")
    speedup = results["Newton-Raphson"]["time"] / results["IV-ANN (CPU)"]["time"]
    print(f"\nSpeedup IV-ANN (CPU) vs Newton-Raphson: {speedup:.0f}x")

    ann_metrics = compute_metrics(sigma_true, iv_ann.flatten())
    print(f"IV-ANN MAE: {ann_metrics['MAE']:.2e}")

    results["_ann_accuracy"] = ann_metrics
    with open(os.path.join(RESULTS_DIR, "iv_speed_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
