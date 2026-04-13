# Option Pricing with Neural Networks

Reproduction of **Liu, Oosterlee & Bohte (2019)** — *Pricing Options and Computing Implied Volatilities Using Neural Networks* ([paper](paper.pdf)).

## Overview

This project trains feedforward neural networks (MLP 4x400, ReLU, Adam) to approximate option pricing functions, replacing expensive numerical solvers with fast batch inference.

Three models are implemented:

| Model | Input | Output | Test R² |
|-------|-------|--------|---------|
| **BS-ANN** | S/K, τ, r, σ | V/K (Black-Scholes price) | 0.9999999 |
| **IV-ANN** | S/K, τ, r, log(V̂/K) | σ* (implied volatility) | 0.9999986 |
| **Heston-ANN** | m, τ, r, ρ, κ, ν̄, γ, ν₀ | V (Heston price via COS) | 0.9914 |

The IV-ANN achieves a **208x speedup** over Newton-Raphson on CPU, with full robustness on deep ITM/OTM options.

## Repository structure

```
code/               Python source (PyTorch)
  ├── model.py            MLP architecture
  ├── black_scholes.py    BS formula + LHS data generation
  ├── heston_cos.py       Heston characteristic function + COS method
  ├── implied_vol.py      Newton-Raphson, Brent, Secant, Bisection
  ├── train.py            Training loop (Decay/Constant/Cyclical LR)
  ├── metrics.py          MSE, RMSE, MAE, MAPE, R²
  ├── utils.py            Device selection, data splits, plotting
  ├── run_all.py          Run all experiments sequentially
  └── experiments/        One script per experiment (exp0–exp9)
figures/             Generated figures (Figures 1–12)
rapport/             Final report (PDF, in French)
paper.pdf            Original article (Liu et al., 2019)
```

## Key results

- **BS-ANN**: MSE = 5.54×10⁻⁹, slightly better than the article (8.21×10⁻⁹)
- **IV-ANN**: gradient-squash transformation improves MSE by 116x
- **Heston-ANN**: R² = 0.9914 vs article's 0.9999993 — gap attributed to the "little trap" formulation of the Heston characteristic function (see report Section 5.2)
- **Speed**: IV-ANN computes 20,000 implied volatilities in 0.03s (208x faster than Newton-Raphson)

## How to run

```bash
cd code
python run_all.py      # all experiments (~24h on M3 CPU)
python run_final.py    # BS + IV + speed + Heston IV surface only (~12h)
```

Requires Python 3.12+ and PyTorch 2.x. Install dependencies:
```bash
pip install torch numpy scipy matplotlib
```

## Hardware

All experiments were run on an Apple MacBook Air M3 (8-core CPU, 16GB unified memory), **CPU only** — GPU (MPS/CUDA) is slower for this model size due to kernel launch overhead.

## Report

The full report (19 pages, in French) is available in [`rapport/rapport.pdf`](rapport/rapport.pdf). It includes all tables and figures from the article, a detailed analysis of the Heston difficulty, and proposals for improvement (residual connections, PINNs, transfer learning).

## Reference

> S. Liu, C.W. Oosterlee, S.M. Bohte. *Pricing Options and Computing Implied Volatilities Using Neural Networks*. Risks 7(1):16, 2019. [arXiv:1901.08943](https://arxiv.org/abs/1901.08943)
