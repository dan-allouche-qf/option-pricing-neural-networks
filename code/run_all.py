#!/usr/bin/env python3
"""Run all experiments sequentially. Usage: python3 run_all.py

Reproduces all figures and tables from:
Liu et al. (2019) "Pricing options and computing implied volatilities using neural networks"
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

experiments = [
    ("Fig 1: Price/Vega visualization", "experiments/exp0_figure1.py"),
    ("Exp1: BS-ANN Pricing (Table 5, Figure 7)", "experiments/exp1_bs_pricing.py"),
    ("Exp2: IV-ANN (Table 7, Figure 8)", "experiments/exp2_iv_bs.py"),
    ("Exp3: Speed Comparison (Table 8)", "experiments/exp3_speed_compare.py"),
    ("Exp4d: Heston-ANN 1M filtered (Table 10, Figure 9)", "experiments/exp4d_heston_filtered.py"),
    ("Exp5: Heston IV Surface (Table 11, Figures 11-12)", "experiments/exp5_heston_iv.py"),
    ("Exp6: LR Finder (Figure 3)", "experiments/exp6_lr_finder.py"),
    ("Exp7: LR Schedule Comparison (Figures 4-5)", "experiments/exp7_lr_schedules.py"),
    ("Exp8: Dataset Size Study (Table 3, Figure 6)", "experiments/exp8_dataset_size.py"),
    ("Fig 10: Pipeline Diagram", "experiments/exp9_figure10_diagram.py"),
]

for name, script in experiments:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    ret = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    if ret.returncode != 0:
        print(f"  WARNING: {script} exited with code {ret.returncode}")

print("\n\nALL EXPERIMENTS COMPLETE.")
print(f"Figures saved in: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')}")
