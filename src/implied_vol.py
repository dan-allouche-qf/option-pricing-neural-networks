"""Implied volatility computation methods (Section 2.3 of the article).

Implements Newton-Raphson, Brent, Secant, Bisection for benchmarking.
"""

import numpy as np
from scipy.optimize import brentq

from black_scholes import bs_price, bs_vega


def newton_raphson_iv(V_mkt, S, K, tau, r, sigma0=0.5, tol=1e-10, max_iter=100):
    """Newton-Raphson for implied volatility (Eq. 9)."""
    sigma = sigma0
    for _ in range(max_iter):
        price = bs_price(S, K, tau, r, sigma)
        vega = bs_vega(S, K, tau, r, sigma)
        if abs(vega) < 1e-15:
            return np.nan
        sigma_new = sigma - (price - V_mkt) / vega
        if sigma_new < 0:
            return np.nan
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new
    return np.nan


def secant_iv(V_mkt, S, K, tau, r, sigma0=0.4, sigma1=0.6, tol=1e-10, max_iter=100):
    """Secant method for implied volatility (Eq. 11)."""
    s0, s1 = sigma0, sigma1
    f0 = bs_price(S, K, tau, r, s0) - V_mkt
    f1 = bs_price(S, K, tau, r, s1) - V_mkt
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-15:
            return np.nan
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        if s2 < 0:
            return np.nan
        if abs(s2 - s1) < tol:
            return s2
        s0, f0 = s1, f1
        s1 = s2
        f1 = bs_price(S, K, tau, r, s1) - V_mkt
    return np.nan


def bisection_iv(V_mkt, S, K, tau, r, a=0.001, b=5.0, tol=1e-10, max_iter=200):
    """Bisection method for implied volatility."""
    fa = bs_price(S, K, tau, r, a) - V_mkt
    fb = bs_price(S, K, tau, r, b) - V_mkt
    if fa * fb > 0:
        return np.nan
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = bs_price(S, K, tau, r, c) - V_mkt
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return np.nan


def brent_iv(V_mkt, S, K, tau, r, a=0.001, b=5.0):
    """Brent's method for implied volatility using scipy (Eq. 10-11)."""
    try:
        return brentq(lambda sig: bs_price(S, K, tau, r, sig) - V_mkt, a, b, xtol=1e-10)
    except ValueError:
        return np.nan


def batch_newton_iv(V_mkt, S, K, tau, r, sigma0=0.5, tol=1e-10, max_iter=100):
    """Vectorized Newton-Raphson for arrays."""
    n = len(V_mkt)
    results = np.full(n, np.nan)
    for i in range(n):
        results[i] = newton_raphson_iv(V_mkt[i], S[i], K[i], tau[i], r[i], sigma0, tol, max_iter)
    return results


def batch_brent_iv(V_mkt, S, K, tau, r):
    n = len(V_mkt)
    results = np.full(n, np.nan)
    for i in range(n):
        results[i] = brent_iv(V_mkt[i], S[i], K[i], tau[i], r[i])
    return results


def batch_secant_iv(V_mkt, S, K, tau, r):
    n = len(V_mkt)
    results = np.full(n, np.nan)
    for i in range(n):
        results[i] = secant_iv(V_mkt[i], S[i], K[i], tau[i], r[i])
    return results


def batch_bisection_iv(V_mkt, S, K, tau, r):
    n = len(V_mkt)
    results = np.full(n, np.nan)
    for i in range(n):
        results[i] = bisection_iv(V_mkt[i], S[i], K[i], tau[i], r[i])
    return results
