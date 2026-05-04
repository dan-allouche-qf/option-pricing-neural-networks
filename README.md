# Option Pricing with Neural Networks

Reproduction of **Liu, Oosterlee & Bohte (2019)** — *Pricing Options and Computing Implied Volatilities Using Neural Networks* ([paper](paper.pdf)).

## Overview

This project trains feedforward neural networks (MLP 4×400, ReLU, Adam, MSE) to approximate three option-pricing maps, replacing expensive numerical solvers with fast batch inference.

| Model       | Input                                         | Output                         | Test R²      |
|-------------|-----------------------------------------------|--------------------------------|--------------|
| **BS-ANN**      | S/K, τ, r, σ                                 | V/K (Black–Scholes price)       | 0.99999985   |
| **IV-ANN**      | S/K, τ, r, log(V̂/K)                         | σ* (implied volatility)         | 0.9999981    |
| **Heston-ANN**  | m, τ, r, ρ, κ, ν̄, γ, ν₀                     | V (Heston price via COS)        | 0.9999990    |

IV-ANN computes 20,000 implied volatilities in **24 ms**, a **270× speed-up** over Newton–Raphson, with no failures on deep ITM/OTM options.

## Repository structure

```
option-pricing-neural-networks/
├── run.py                           orchestrator (runs all 10 experiments)
├── paper.pdf                        Liu, Oosterlee & Bohte (2019)
├── report/                          LaTeX + PDF report (English)
├── src/
│   ├── black_scholes.py             BS analytics + LHS data generation
│   ├── heston.py                    Heston char. func., COS method
│   ├── implied_vol.py               Newton, Brent, secant, bisection
│   ├── model.py                     PricingMLP (4×400, ReLU, Glorot)
│   ├── train.py                     Adam + MultiStepLR / CyclicLR
│   ├── metrics.py                   MSE, RMSE, MAE, MAPE, R²
│   ├── utils.py                     splits, data loaders, plots
│   └── experiments/
│       ├── figures/                 illustrative plots
│       │   ├── payoff_vega.py         (Figure 1)
│       │   └── pipeline_diagram.py    (Figure 10)
│       ├── training/                the three ANN trainings
│       │   ├── bs_ann.py              (Table 5, Figure 7)
│       │   ├── iv_ann.py              (Table 7, Figure 8)
│       │   └── heston_ann.py          (Table 10, Figure 9)
│       ├── benchmarks/              studies and comparisons
│       │   ├── lr_finder.py           (Figure 3)
│       │   ├── lr_schedules.py        (Figures 4–5)
│       │   ├── iv_speed.py            (Table 8)
│       │   └── dataset_size.py        (Table 3, Figure 6)
│       └── pipelines/
│           └── heston_iv_surface.py   (Table 11, Figures 11–12)
└── outputs/                         everything produced by run.py
    ├── data/                        cached .npz datasets
    ├── models/                      trained .pt weights
    ├── figures/                     all generated plots
    └── results/                     per-experiment metrics + logs + SUMMARY.md
```

## Key results

| Metric                        | Article target         | This work              |
|-------------------------------|------------------------|------------------------|
| BS-ANN test MSE               | ~8 × 10⁻⁹              | 7.23 × 10⁻⁹            |
| IV-ANN scaled test MSE        | ~1.5 × 10⁻⁸            | 1.31 × 10⁻⁷            |
| IV-ANN speed-up vs Newton     | > 100×                 | **270×**               |
| Heston-ANN test MSE           | ~1.5 × 10⁻⁸            | 2.35 × 10⁻⁸            |
| Heston-ANN test R²            | > 0.9999993            | 0.9999990              |

See [`report/report.pdf`](report/report.pdf) for the full write-up.

## How to run

```bash
pip install -r requirements.txt

# Single-command full pipeline (~52 h on an M3 laptop)
caffeinate -dimsu python3 run.py 2>&1 | tee outputs/results/run.log
```

Requirements: Python 3.12+, PyTorch 2.x, NumPy, SciPy, Matplotlib, tqdm
(see `requirements.txt`).

## Reproducibility

Each experiment calls `set_seed(42)` (see `src/utils.py`) before any
`torch` allocation, so both LHS data generation (numpy) and model
initialisation / DataLoader shuffling (PyTorch) are deterministic.
Two consecutive runs from a clean state produce bit-identical metrics.

The `outputs/data/*.npz` LHS caches are not versioned (regenerable from
`seed=42`); they are produced on the first run. Trained weights
(`outputs/models/*.pt`), figures, and per-experiment metrics are
versioned so the report can be re-built without retraining.

## Hardware

All experiments were run on an Apple MacBook Air M3 (8-core CPU, 16 GB unified memory), **CPU only** — on this model size, GPU (MPS / CUDA) is slower than CPU due to kernel-launch overhead.

## Reference

> S. Liu, C.W. Oosterlee, S.M. Bohte. *Pricing Options and Computing Implied Volatilities Using Neural Networks*. Risks 7(1):16, 2019. [arXiv:1901.08943](https://arxiv.org/abs/1901.08943)
