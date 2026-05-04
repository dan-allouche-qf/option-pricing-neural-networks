"""Black-Scholes analytical pricing and data generation (Section 2.1 of the article)."""

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube


def bs_price(S, K, tau, r, sigma):
    """Black-Scholes European call price (Eq. 4a-4b)."""
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_vega(S, K, tau, r, sigma):
    """Black-Scholes Vega = dV/dsigma (used in Newton-Raphson for IV)."""
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return S * np.sqrt(tau) * norm.pdf(d1)


def generate_bs_data(n_samples=1_000_000, param_range="wide", seed=42):
    """Generate Black-Scholes training/test data (Table 4 of the article).

    Returns:
        inputs: (n, 4) with columns [S0/K, tau, r, sigma]
        outputs: (n, 1) with column [V/K]
    """
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_samples)

    if param_range == "wide":
        ranges = np.array([
            [0.4, 1.6],    # S0/K
            [0.2, 1.1],    # tau
            [0.02, 0.1],   # r
            [0.01, 1.0],   # sigma
        ])
    elif param_range == "narrow":
        ranges = np.array([
            [0.5, 1.5],
            [0.3, 0.95],
            [0.03, 0.08],
            [0.02, 0.9],
        ])
    else:
        raise ValueError(f"Unknown param_range: {param_range}")

    params = samples * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    S_K, tau, r, sigma = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    V_K = bs_price(S_K, 1.0, tau, r, sigma)

    return np.column_stack([S_K, tau, r, sigma]), V_K.reshape(-1, 1)


def generate_iv_data(n_samples=1_000_000, seed=42):
    """Generate data for implied volatility learning (Table 6).

    Uses gradient-squash: input includes log(V_hat/K), output is sigma.

    Returns:
        inputs_raw: (n, 4) with [S0/K, tau, r, V/K]
        inputs_scaled: (n, 4) with [S0/K, tau, r, log(V_hat/K)]
        outputs: (n, 1) with sigma
    """
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_samples)

    ranges = np.array([
        [0.5, 1.4],    # S0/K
        [0.05, 1.0],   # tau
        [0.0, 0.1],    # r
        [0.05, 1.0],   # sigma
    ])
    params = samples * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    S_K, tau, r, sigma = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

    V_K = bs_price(S_K, 1.0, tau, r, sigma)

    intrinsic = np.maximum(S_K - np.exp(-r * tau), 0.0)
    V_hat = V_K - intrinsic

    mask = V_hat > 1e-7
    S_K, tau, r, sigma, V_K, V_hat = S_K[mask], tau[mask], r[mask], sigma[mask], V_K[mask], V_hat[mask]

    log_V_hat = np.log(V_hat)

    inputs_raw = np.column_stack([S_K, tau, r, V_K])
    inputs_scaled = np.column_stack([S_K, tau, r, log_V_hat])
    outputs = sigma.reshape(-1, 1)

    return inputs_raw, inputs_scaled, outputs
