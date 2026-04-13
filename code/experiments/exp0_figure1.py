"""Experiment 0: Reproduce Figure 1 of the article.

Figure 1a: Option price vs. volatility for different moneyness values.
Figure 1b: Vega vs. Moneyness, showing ITM/ATM/OTM regions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import bs_price, bs_vega
from utils import FIGURES_DIR


def main():
    print("=" * 60)
    print("EXPERIMENT 0: Figure 1 (Option price vs volatility, Vega vs Moneyness)")
    print("=" * 60)

    # ---- Figure 1a: Option price vs. volatility ----
    print("\n[1] Generating Figure 1a...")
    sigma_range = np.linspace(0.01, 10.0, 500)
    moneyness_values = [1.30, 1.20, 1.10, 1.00, 0.90, 0.80, 0.70]
    tau, r, K = 0.5, 0.02, 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in moneyness_values:
        S = m * K
        prices = bs_price(S, K, tau, r, sigma_range)
        ax.plot(sigma_range, prices, label=f"Moneyness ={m:.2f}")

    ax.set_xlabel("Volatility", fontsize=12)
    ax.set_ylabel("Option Price", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1a_price_vs_vol.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    # ---- Figure 1b: Vega vs. Moneyness ----
    print("\n[2] Generating Figure 1b...")
    m_range = np.linspace(0.01, 2.5, 500)
    sigma_fixed = 0.3

    vegas = []
    for m in m_range:
        S = m * K
        v = bs_vega(S, K, tau, r, sigma_fixed)
        vegas.append(float(v))
    vegas = np.array(vegas)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m_range, vegas, color="#2980B9", linewidth=2)
    ax.set_xlabel("Moneyness", fontsize=12)
    ax.set_ylabel("Vega", fontsize=12)

    # Mark ITM, ATM, OTM regions
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7)
    y_max = vegas.max()
    ax.text(0.5, y_max * 0.6, "ITM", fontsize=12, ha="center", fontweight="bold")
    ax.text(1.0, y_max * 0.85, "ATM", fontsize=12, ha="center", fontweight="bold")
    ax.text(1.7, y_max * 0.6, "OTM", fontsize=12, ha="center", fontweight="bold")

    ax.set_xlim(0, 2.5)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1b_vega_vs_moneyness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 0 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
