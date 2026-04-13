"""Heston model pricing via the COS method (Section 2.2, 4.4 of the article).

Implements:
- Heston characteristic function (Heston 1993)
- COS method for European call pricing (Fang & Oosterlee 2009)
- Data generation for Heston-ANN training (Table 9)
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube
from multiprocessing import Pool, cpu_count


def heston_char_func(u, tau, r, kappa, vbar, gamma, rho, v0):
    """Heston characteristic function phi(u) = E[exp(i*u*x)] where x = ln(S_T/K)."""
    d = np.sqrt((rho * gamma * 1j * u - kappa) ** 2 + gamma ** 2 * (1j * u + u ** 2))
    g = (kappa - rho * gamma * 1j * u - d) / (kappa - rho * gamma * 1j * u + d)

    C = (1j * u * r * tau
         + (kappa * vbar / gamma ** 2)
         * ((kappa - rho * gamma * 1j * u - d) * tau
            - 2.0 * np.log((1.0 - g * np.exp(-d * tau)) / (1.0 - g))))

    D = ((kappa - rho * gamma * 1j * u - d) / gamma ** 2) \
        * ((1.0 - np.exp(-d * tau)) / (1.0 - g * np.exp(-d * tau)))

    return np.exp(C + D * v0)


def _heston_cumulants(tau, r, kappa, vbar, gamma, rho, v0):
    """First 4 cumulants of the Heston log-price process (for truncation range)."""
    # c1 = E[x]
    c1 = r * tau + (1 - np.exp(-kappa * tau)) * (vbar - v0) / (2 * kappa) - 0.5 * vbar * tau
    # c2 = Var[x]
    c2 = (1.0 / (8 * kappa ** 3)) * (
        gamma * tau * kappa * np.exp(-kappa * tau) * (v0 - vbar) * (8 * kappa * rho - 4 * gamma)
        + kappa * rho * gamma * (1 - np.exp(-kappa * tau)) * (16 * vbar - 8 * v0)
        + 2 * vbar * kappa * tau * (-4 * kappa * rho * gamma + gamma ** 2 + 4 * kappa ** 2)
        + gamma ** 2 * ((vbar - 2 * v0) * np.exp(-2 * kappa * tau)
                        + vbar * (6 * np.exp(-kappa * tau) - 7) + 2 * v0)
        + 8 * kappa ** 2 * (v0 - vbar) * (1 - np.exp(-kappa * tau))
    )
    return c1, max(c2, 1e-8)


def _chi(k, a, b, c, d):
    """Helper: chi coefficients for COS method."""
    omega = k * np.pi / (b - a)
    val = np.zeros_like(omega, dtype=float)
    denom = 1.0 + omega ** 2
    val = (1.0 / denom) * (
        np.cos(omega * (d - a)) * np.exp(d)
        - np.cos(omega * (c - a)) * np.exp(c)
        + omega * np.sin(omega * (d - a)) * np.exp(d)
        - omega * np.sin(omega * (c - a)) * np.exp(c)
    )
    return val


def _psi(k, a, b, c, d):
    """Helper: psi coefficients for COS method."""
    omega = k * np.pi / (b - a)
    result = np.zeros_like(k, dtype=float)
    result[k == 0] = d - c
    mask = k > 0
    result[mask] = (np.sin(omega[mask] * (d - a)) - np.sin(omega[mask] * (c - a))) / omega[mask]
    return result


def cos_price_call(S, K, tau, r, kappa, vbar, gamma, rho, v0, N=1500, L=50):
    """Price a European call option using the COS method.

    Uses cumulant-based truncation and log-moneyness shift x = ln(S/K).
    """
    x = np.log(S / K)

    # Truncation range based on cumulants
    c1, c2 = _heston_cumulants(tau, r, kappa, vbar, gamma, rho, v0)
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)

    a = max(a, -30)
    b = min(b, 30)
    if b - a < 0.1:
        b = a + 0.1

    k_vec = np.arange(0, N)
    omega = k_vec * np.pi / (b - a)

    # phi(omega_k) * exp(i * omega_k * x) to account for the log-moneyness shift
    char_vals = heston_char_func(omega, tau, r, kappa, vbar, gamma, rho, v0)
    phi_adj = np.real(char_vals * np.exp(1j * omega * x) * np.exp(-1j * k_vec * np.pi * a / (b - a)))

    # Call payoff coefficients on [0, b]
    chi = _chi(k_vec, a, b, 0, b)
    psi = _psi(k_vec, a, b, 0, b)
    V_k = 2.0 / (b - a) * (chi - psi)
    V_k[0] *= 0.5

    price = K * np.exp(-r * tau) * np.sum(phi_adj * V_k)
    return max(float(np.real(price)), 0.0)


def cos_price_put(S, K, tau, r, kappa, vbar, gamma, rho, v0, N=1500, L=50):
    """Price a European put via COS method."""
    x = np.log(S / K)
    c1, c2 = _heston_cumulants(tau, r, kappa, vbar, gamma, rho, v0)
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)
    a = max(a, -30)
    b = min(b, 30)
    if b - a < 0.1:
        b = a + 0.1

    k_vec = np.arange(0, N)
    omega = k_vec * np.pi / (b - a)

    char_vals = heston_char_func(omega, tau, r, kappa, vbar, gamma, rho, v0)
    phi_adj = np.real(char_vals * np.exp(1j * omega * x) * np.exp(-1j * k_vec * np.pi * a / (b - a)))

    # Put payoff coefficients on [a, 0]
    chi = _chi(k_vec, a, b, a, 0)
    psi = _psi(k_vec, a, b, a, 0)
    V_k = 2.0 / (b - a) * (-chi + psi)
    V_k[0] *= 0.5

    price = K * np.exp(-r * tau) * np.sum(phi_adj * V_k)
    return max(float(np.real(price)), 0.0)


def heston_call_price(S, K, tau, r, kappa, vbar, gamma, rho, v0, N=1500, L=50):
    """Price a call, using put-call parity for deep OTM (article p.16)."""
    m = S / K
    if m < 0.85:
        put = cos_price_put(S, K, tau, r, kappa, vbar, gamma, rho, v0, N, L)
        call = put + S - K * np.exp(-r * tau)
        return max(call, 0.0)
    else:
        return cos_price_call(S, K, tau, r, kappa, vbar, gamma, rho, v0, N, L)


def _compute_single_heston_price(params_row):
    """Worker function for multiprocessing. Returns (params, price) or None."""
    m, tau, r, rho, kappa, vbar, gamma_h, v0 = params_row
    K = 1.0
    S = m * K
    try:
        price = heston_call_price(S, K, tau, r, kappa, vbar, gamma_h, rho, v0)
        if np.isfinite(price) and 0 <= price < 2.0:
            return (params_row, price)
    except Exception:
        pass
    return None


def generate_heston_data(n_samples=1_000_000, seed=42):
    """Generate Heston training data using LHS sampling (Table 9).

    Fixed K=1. Inputs: (m, tau, r, rho, kappa, vbar, gamma, v0). Output: V (call price).
    Uses multiprocessing to leverage all CPU cores.
    """
    print(f"  Generating {n_samples} Heston samples via COS method (N=1500, L=50)...")

    sampler = LatinHypercube(d=8, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale to parameter ranges (Table 9 of the article)
    ranges = np.array([
        [0.6, 1.4],      # moneyness m = S/K
        [0.1, 1.4],      # tau (years)
        [0.0, 0.10],     # r (Table 9: 0.0% to 10%)
        [-0.95, 0.0],    # rho (Table 9: -0.95 to 0.0)
        [0.01, 2.0],     # kappa (Table 9: 0.0 to 2.0, open)
        [0.01, 0.5],     # vbar (Table 9: 0.0 to 0.5, open)
        [0.01, 0.5],     # gamma (Table 9: 0.0 to 0.5, open)
        [0.05, 0.5],     # v0
    ])

    params = samples * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    # Parallel computation using all CPU cores
    n_workers = min(cpu_count(), 8)
    print(f"  Using {n_workers} workers...")

    params_list = [params[i] for i in range(n_samples)]

    with Pool(n_workers) as pool:
        results = []
        batch_size = 50_000
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_results = pool.map(
                _compute_single_heston_price,
                params_list[start:end],
                chunksize=500,
            )
            results.extend(batch_results)
            valid = sum(1 for r in results if r is not None)
            skipped = len(results) - valid
            print(f"    {end}/{n_samples} done ({valid} valid, {skipped} skipped)", flush=True)

    # Filter valid results
    valid_results = [r for r in results if r is not None]
    inputs = np.array([r[0] for r in valid_results])
    outputs = np.array([r[1] for r in valid_results]).reshape(-1, 1)
    print(f"  Done. {len(inputs)} valid samples ({n_samples - len(inputs)} skipped)")

    return inputs, outputs
