"""Heston model pricing via the COS method (Sections 2.2, 4.4 of the article).

Implements:
- Heston characteristic function using the "Little Heston Trap" formulation
  (Albrecher, Mayer, Schoutens, Tistaert 2007) - numerically stable
- COS method for the European put (Fang & Oosterlee 2009)
- Call price recovered from put via put-call parity (the direct call chi
  coefficient contains exp(b_upper) which overflows for extreme parameters)
- Data generation for Heston-ANN training (Table 9)
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube
from multiprocessing import Pool, cpu_count


def heston_char_func(u, tau, r, kappa, vbar, gamma, rho, v0):
    """Heston characteristic function phi(u) = E[exp(i*u*x)] with x = ln(S_T/K).

    Uses the Little Heston Trap formulation (Albrecher et al. 2007):
        g = (b - d) / (b + d)    with  b = kappa - rho*gamma*i*u
    This avoids the log branch-cut discontinuity for long maturities.
    """
    u = np.asarray(u, dtype=np.complex128)
    i = 1j

    b = kappa - rho * gamma * i * u
    d = np.sqrt(b * b + gamma * gamma * (i * u + u * u))

    g = (b - d) / (b + d)
    exp_dt = np.exp(-d * tau)

    C = i * u * r * tau + (kappa * vbar / (gamma * gamma)) * (
        (b - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = (b - d) / (gamma * gamma) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

    return np.exp(C + D * v0)


def heston_cumulants(tau, r, kappa, vbar, gamma, rho, v0):
    """First and second cumulants of the Heston log-price process.

    Formula from Fang & Oosterlee (2009), used to determine the COS
    integration range [a, b] = [c1 - L*sqrt(c2), c1 + L*sqrt(c2)].
    """
    c1 = r * tau + (1 - np.exp(-kappa * tau)) * (vbar - v0) / (2 * kappa) - 0.5 * vbar * tau

    c2 = (1.0 / (8.0 * kappa ** 3)) * (
        gamma * tau * kappa * np.exp(-kappa * tau) * (v0 - vbar) * (8 * kappa * rho - 4 * gamma)
        + kappa * rho * gamma * (1 - np.exp(-kappa * tau)) * (16 * vbar - 8 * v0)
        + 2 * vbar * kappa * tau * (-4 * kappa * rho * gamma + gamma ** 2 + 4 * kappa ** 2)
        + gamma ** 2 * ((vbar - 2 * v0) * np.exp(-2 * kappa * tau)
                        + vbar * (6 * np.exp(-kappa * tau) - 7) + 2 * v0)
        + 8 * kappa ** 2 * (v0 - vbar) * (1 - np.exp(-kappa * tau))
    )
    return c1, c2


def _chi(k, a, b, c, d):
    """COS chi coefficients: int_c^d exp(y) cos(k*pi*(y-a)/(b-a)) dy."""
    omega = k * np.pi / (b - a)
    denom = 1.0 + omega * omega
    return (1.0 / denom) * (
        np.cos(omega * (d - a)) * np.exp(d)
        - np.cos(omega * (c - a)) * np.exp(c)
        + omega * np.sin(omega * (d - a)) * np.exp(d)
        - omega * np.sin(omega * (c - a)) * np.exp(c)
    )


def _psi(k, a, b, c, d):
    """COS psi coefficients: int_c^d cos(k*pi*(y-a)/(b-a)) dy."""
    omega = k * np.pi / (b - a)
    result = np.zeros_like(k, dtype=float)
    result[k == 0] = d - c
    mask = k > 0
    result[mask] = (np.sin(omega[mask] * (d - a)) - np.sin(omega[mask] * (c - a))) / omega[mask]
    return result


def cos_price_put(S, K, tau, r, kappa, vbar, gamma, rho, v0, N=1500, L=50):
    """Price a European put via the COS method."""
    x = np.log(S / K)

    c1, c2 = heston_cumulants(tau, r, kappa, vbar, gamma, rho, v0)

    if not np.isfinite(c1) or not np.isfinite(c2) or c2 <= 0:
        return np.nan

    sqrt_c2 = np.sqrt(c2)
    a = c1 - L * sqrt_c2
    b = c1 + L * sqrt_c2

    if b - a < 1e-6 or not (np.isfinite(a) and np.isfinite(b)):
        return np.nan

    k_vec = np.arange(0, N)
    omega = k_vec * np.pi / (b - a)

    char_vals = heston_char_func(omega, tau, r, kappa, vbar, gamma, rho, v0)
    phi_adj = np.real(char_vals * np.exp(1j * omega * x) * np.exp(-1j * k_vec * np.pi * a / (b - a)))

    chi = _chi(k_vec, a, b, a, 0.0)
    psi = _psi(k_vec, a, b, a, 0.0)
    V_k = 2.0 / (b - a) * (-chi + psi)
    V_k[0] *= 0.5

    price = K * np.exp(-r * tau) * np.sum(phi_adj * V_k)

    if not np.isfinite(price):
        return np.nan
    return float(np.real(price))


def heston_call_price(S, K, tau, r, kappa, vbar, gamma, rho, v0, N=1500, L=50):
    """Heston European call price via put + put-call parity.

    The direct call chi coefficient involves exp(b_upper) where
    b_upper = c1 + L*sqrt(c2); for extreme parameters (high v0, long tau,
    high gamma) it overflows. The put chi only involves exp(0) - exp(a) with
    a << 0, hence numerically stable. We recover C = P + S - K*exp(-r*tau).
    """
    put = cos_price_put(S, K, tau, r, kappa, vbar, gamma, rho, v0, N, L)
    if not np.isfinite(put):
        return np.nan
    return put + S - K * np.exp(-r * tau)


def _compute_single_heston_price(params_row):
    """Worker for multiprocessing. Accept/reject based on validity of COS output."""
    m, tau, r, rho, kappa, vbar, gamma_h, v0 = params_row
    K = 1.0
    S = m * K
    try:
        price = heston_call_price(S, K, tau, r, kappa, vbar, gamma_h, rho, v0)
    except Exception:
        return None
    if not np.isfinite(price):
        return None
    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    if price < intrinsic - 1e-8 or price > S + 1e-8:
        return None
    return (params_row, max(price, 0.0))


def generate_heston_data(n_samples=1_000_000, seed=42, verbose=True):
    """Generate Heston training data using LHS sampling (Table 9 of the article).

    Fixed K=1. Inputs: (m, tau, r, rho, kappa, vbar, gamma, v0). Output: V (call price).
    Uses multiprocessing to leverage all CPU cores.
    Minimal filtering: keep prices that are finite and in [intrinsic, S].
    """
    if verbose:
        print(f"  Generating {n_samples:,} Heston samples via COS method (N=1500, L=50)")

    sampler = LatinHypercube(d=8, seed=seed)
    samples = sampler.random(n=n_samples)

    ranges = np.array([
        [0.6, 1.4],      # moneyness m = S/K
        [0.1, 1.4],      # tau
        [0.0, 0.10],     # r
        [-0.95, 0.0],    # rho
        [0.01, 2.0],     # kappa
        [0.01, 0.5],     # vbar
        [0.01, 0.5],     # gamma
        [0.05, 0.5],     # v0
    ])
    params = samples * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    params_list = [params[i] for i in range(n_samples)]

    n_workers = min(cpu_count(), 8)
    if verbose:
        print(f"  Using {n_workers} workers")

    results = []
    batch_size = 50_000
    with Pool(n_workers) as pool:
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_results = pool.map(
                _compute_single_heston_price,
                params_list[start:end],
                chunksize=500,
            )
            results.extend(batch_results)
            if verbose:
                valid = sum(1 for r in results if r is not None)
                print(f"    {end:,}/{n_samples:,} done ({valid:,} valid, {end - valid:,} skipped)",
                      flush=True)

    valid_results = [r for r in results if r is not None]
    inputs = np.array([r[0] for r in valid_results])
    outputs = np.array([r[1] for r in valid_results]).reshape(-1, 1)

    if verbose:
        print(f"  Done. {len(inputs):,} valid samples ({n_samples - len(inputs):,} skipped)")

    return inputs, outputs
