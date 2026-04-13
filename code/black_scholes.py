"""Black-Scholes analytical pricing and data generation (Section 2.1 of the article)."""

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube


def bs_price(S, K, tau, r, sigma):
    """Black-Scholes European call price (Eq. 4a-4b).

    All inputs can be numpy arrays (vectorized).
    """
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return price


def bs_vega(S, K, tau, r, sigma):
    """Black-Scholes Vega = dV/dsigma (used in Newton-Raphson for IV)."""
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return S * np.sqrt(tau) * norm.pdf(d1)


def generate_bs_data(n_samples=1_000_000, param_range="wide", seed=42):
    """Generate Black-Scholes training/test data (Table 4 of the article).

    Args:
        n_samples: number of samples
        param_range: "wide" or "narrow" (Table 4)
        seed: random seed

    Returns:
        inputs: array of shape (n, 4) with columns [S0/K, tau, r, sigma]
        outputs: array of shape (n, 1) with column [V/K]
    """
    # Latin Hypercube Sampling (Section 4.1 of the article)
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_samples)

    if param_range == "wide":
        # Table 4: Wide range
        ranges = np.array([
            [0.4, 1.6],    # S0/K
            [0.2, 1.1],    # tau (years)
            [0.02, 0.1],   # r
            [0.01, 1.0],   # sigma
        ])
    elif param_range == "narrow":
        # Table 4: Narrow range
        ranges = np.array([
            [0.5, 1.5],    # S0/K
            [0.3, 0.95],   # tau
            [0.03, 0.08],  # r
            [0.02, 0.9],   # sigma
        ])
    else:
        raise ValueError(f"Unknown param_range: {param_range}")

    params = samples * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    S_K = params[:, 0]
    tau = params[:, 1]
    r = params[:, 2]
    sigma = params[:, 3]

    # Compute V/K using BS formula with S=S0/K, K=1
    V_K = bs_price(S_K, 1.0, tau, r, sigma)

    inputs = np.column_stack([S_K, tau, r, sigma])
    outputs = V_K.reshape(-1, 1)

    return inputs, outputs


def generate_iv_data(n_samples=1_000_000, seed=42):
    """Generate data for implied volatility learning (Table 6).

    Uses gradient-squash: input includes log(V_hat/K), output is sigma.

    Returns:
        inputs_raw: (n, 4) with [S0/K, tau, r, V/K] (for comparison without scaling)
        inputs_scaled: (n, 4) with [S0/K, tau, r, log(V_hat/K)] (gradient-squash)
        outputs: (n, 1) with sigma
    """
    # Latin Hypercube Sampling (Section 4.1) with Table 6 ranges
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_samples)

    ranges = np.array([
        [0.5, 1.4],    # S0/K
        [0.05, 1.0],   # tau (years)
        [0.0, 0.1],    # r
        [0.05, 1.0],   # sigma
    ])
    params = samples * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    S_K = params[:, 0]
    tau = params[:, 1]
    r = params[:, 2]
    sigma = params[:, 3]

    # Compute option price V/K
    V_K = bs_price(S_K, 1.0, tau, r, sigma)

    # Compute time value: V_hat = V - max(S - K*exp(-r*tau), 0) with K=1
    intrinsic = np.maximum(S_K - np.exp(-r * tau), 0.0)
    V_hat = V_K - intrinsic

    # Filter out samples with very small time value (article: V_hat < 1e-7)
    mask = V_hat > 1e-7
    S_K = S_K[mask]
    tau = tau[mask]
    r = r[mask]
    sigma = sigma[mask]
    V_K = V_K[mask]
    V_hat = V_hat[mask]

    log_V_hat = np.log(V_hat)  # K=1 so log(V_hat/K) = log(V_hat)

    inputs_raw = np.column_stack([S_K, tau, r, V_K])
    inputs_scaled = np.column_stack([S_K, tau, r, log_V_hat])
    outputs = sigma.reshape(-1, 1)

    return inputs_raw, inputs_scaled, outputs
