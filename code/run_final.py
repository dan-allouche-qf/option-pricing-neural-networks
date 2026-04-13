#!/usr/bin/env python3
"""Final run: BS-ANN and IV-ANN with 1M samples, then exp3 and exp5."""

import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

experiments = [
    ("BS-ANN 1M (Table 5)", "experiments/exp1_bs_pricing.py"),
    ("IV-ANN 1M (Table 7)", "experiments/exp2_iv_bs.py"),
    ("Speed Comparison (Table 8)", "experiments/exp3_speed_compare.py"),
    ("Heston IV Surface (Table 11)", "experiments/exp5_heston_iv.py"),
]

t_total = time.time()

for name, script in experiments:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    t0 = time.time()
    ret = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    t1 = time.time()
    print(f"\n  Duration: {(t1-t0)/60:.1f} min")
    if ret.returncode != 0:
        print(f"  WARNING: {script} exited with code {ret.returncode}")

print(f"\n\n{'='*60}")
print(f"FINAL RUN COMPLETE. Total time: {(time.time()-t_total)/3600:.1f} hours")
print(f"{'='*60}")
