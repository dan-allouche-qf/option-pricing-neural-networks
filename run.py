"""Master script: run all 10 experiments end-to-end and summarize results.

Usage:
    caffeinate -dimsu python3 run.py 2>&1 | tee outputs/results/run.log

`caffeinate -dimsu` prevents macOS sleep so the run (24 h+) finishes.

Experiments are grouped by role under src/experiments/:
    figures/       - illustrative plots (instant)
    training/      - the three ANN trainings (BS, IV, Heston)
    benchmarks/    - LR / dataset / speed studies
    pipelines/     - end-to-end composition of trained ANNs

Order (dependencies respected):
    figures.pipeline_diagram         - instant
    figures.payoff_vega              - instant
    benchmarks.lr_finder             - ~20 min
    training.bs_ann                  - ~1.5 h
    training.iv_ann                  - ~1.5 h
    benchmarks.iv_speed              - ~10 min  [needs training.iv_ann]
    training.heston_ann              - ~12-15 h [needs Heston data]
    pipelines.heston_iv_surface      - ~30 min  [needs iv_ann + heston_ann]
    benchmarks.lr_schedules          - ~9 h     [needs Heston data]
    benchmarks.dataset_size          - ~3 h

Each experiment writes:
    outputs/data/<name>.npz           (cached datasets)
    outputs/models/<name>.pt          (trained model weights)
    outputs/results/<name>_metrics.json  (all metrics)
    outputs/results/<name>_history.json  (training curves)
    outputs/results/<name>_log.txt       (stdout/stderr capture)
    outputs/figures/<name>.png        (plots)

At the end: outputs/results/SUMMARY.md (all metrics side-by-side vs article).
"""

import os
import sys
import json
import time
import importlib
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "outputs", "results")
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, SRC)


# Execution order, display name, and the module to import.
# Short name is used as the result-file prefix (bs_ann_metrics.json, etc.).
EXPERIMENTS = [
    ("pipeline_diagram",   "figures.pipeline_diagram",        "Figure 10 pipeline diagram"),
    ("payoff_vega",        "figures.payoff_vega",             "Figure 1 payoff / vega"),
    ("lr_finder",          "benchmarks.lr_finder",            "LR range test (Figure 3)"),
    ("bs_ann",             "training.bs_ann",                 "BS-ANN (Table 5, Figure 7)"),
    ("iv_ann",             "training.iv_ann",                 "IV-ANN (Table 7, Figure 8)"),
    ("iv_speed",           "benchmarks.iv_speed",             "IV solver speed (Table 8)"),
    ("heston_ann",         "training.heston_ann",             "Heston-ANN (Table 10, Figure 9)"),
    ("heston_iv_surface",  "pipelines.heston_iv_surface",     "Heston + IV surface (Table 11, Figures 11-12)"),
    ("lr_schedules",       "benchmarks.lr_schedules",         "LR schedules (Figures 4-5)"),
    ("dataset_size",       "benchmarks.dataset_size",         "Dataset size study (Table 3, Figure 6)"),
]


class Tee:
    """Duplicate writes to both the terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def run_experiment(short_name, module_path, title):
    ts_start = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "#" * 72)
    print(f"# [{ts_start}] START: {short_name} - {title}")
    print("#" * 72, flush=True)

    log_path = os.path.join(RESULTS, f"{short_name}_log.txt")
    log_file = open(log_path, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(old_stdout, log_file)
    sys.stderr = Tee(old_stderr, log_file)

    full_module = f"experiments.{module_path}"

    t0 = time.time()
    success = False
    try:
        if full_module in sys.modules:
            importlib.reload(sys.modules[full_module])
        else:
            importlib.import_module(full_module)
        mod = importlib.import_module(full_module)
        mod.main()
        success = True
    except Exception as e:
        print(f"\nERROR in {full_module}: {e}")
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()

    elapsed = time.time() - t0
    ts_end = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if success else "FAILED"
    print(f"\n[{ts_end}] END: {short_name} ({status}) - {elapsed / 60:.1f} min")
    print(f"  log: {log_path}")
    return success, elapsed


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_summary(run_times):
    lines = []
    lines.append("# Experimental Results Summary\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    total = sum(run_times.values())
    lines.append(f"Total runtime: {total / 3600:.2f} hours ({total / 60:.1f} min)\n\n")

    lines.append("## Runtime per experiment\n")
    lines.append("| Experiment | Time (min) |\n|---|---|\n")
    for name, t in run_times.items():
        lines.append(f"| {name} | {t / 60:.1f} |\n")
    lines.append("\n")

    # BS-ANN
    m = load_json(os.path.join(RESULTS, "bs_ann_metrics.json"))
    if m:
        lines.append("## Table 5: BS-ANN\n\n")
        lines.append("| Dataset | MSE | RMSE | MAE | MAPE | R2 |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for k in ["train_wide", "test_wide", "test_narrow"]:
            if k in m:
                v = m[k]
                lines.append(f"| {k} | {v['MSE']:.3e} | {v['RMSE']:.3e} | "
                             f"{v['MAE']:.3e} | {v['MAPE']:.3e} | {v['R2']:.7f} |\n")
        lines.append("\nArticle target: MSE ~8e-9, R2 > 0.99999\n\n")

    # IV-ANN
    m = load_json(os.path.join(RESULTS, "iv_ann_metrics.json"))
    if m:
        lines.append("## Table 7: IV-ANN\n\n")
        lines.append("| Input | MSE | MAE | MAPE | R2 |\n|---|---|---|---|---|\n")
        for label, key in [("Without scaling", "without_scaling"),
                           ("With gradient-squash", "with_scaling")]:
            if key in m:
                v = m[key]
                lines.append(f"| {label} | {v['MSE']:.3e} | {v['MAE']:.3e} | "
                             f"{v['MAPE']:.3e} | {v['R2']:.7f} |\n")
        lines.append("\nArticle target (scaled): MSE ~1.5e-8, R2 > 0.9999998\n\n")

    # Speed
    m = load_json(os.path.join(RESULTS, "iv_speed_metrics.json"))
    if m:
        lines.append("## Table 8: Speed comparison (IV computation)\n\n")
        lines.append("| Method | Time (s) | Robust |\n|---|---|---|\n")
        for method, v in m.items():
            if method.startswith("_"):
                continue
            robust = "Yes" if v.get("robust") else "No"
            lines.append(f"| {method} | {v['time']:.2f} | {robust} |\n")
        lines.append("\n")

    # Heston-ANN
    m = load_json(os.path.join(RESULTS, "heston_ann_metrics.json"))
    if m:
        lines.append("## Table 10: Heston-ANN\n\n")
        lines.append("| Dataset | MSE | RMSE | MAE | MAPE | R2 |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for k in ["train", "test"]:
            if k in m:
                v = m[k]
                lines.append(f"| {k} | {v['MSE']:.3e} | {v['RMSE']:.3e} | "
                             f"{v['MAE']:.3e} | {v['MAPE']:.3e} | {v['R2']:.7f} |\n")
        lines.append("\nArticle target: MSE ~1.5e-8, R2 > 0.9999993\n\n")

    # Heston + IV pipeline
    m = load_json(os.path.join(RESULTS, "heston_iv_surface_metrics.json"))
    if m:
        lines.append("## Table 11: Heston-ANN + IV-ANN pipeline\n\n")
        lines.append("| Case | RMSE | MAE | MAPE | R2 |\n|---|---|---|---|---|\n")
        for k in ["case1", "case2", "surface"]:
            if k in m:
                v = m[k]
                lines.append(f"| {k} | {v['RMSE']:.3e} | {v['MAE']:.3e} | "
                             f"{v['MAPE']:.3e} | {v['R2']:.7f} |\n")
        lines.append("\n")

    # Dataset size
    m = load_json(os.path.join(RESULTS, "dataset_size_metrics.json"))
    if m:
        lines.append("## Table 3: Dataset size study\n\n")
        lines.append("| Case | Train size (x baseline) | Train MSE | Test MSE | R2 (%) |\n")
        lines.append("|---|---|---|---|---|\n")
        for i, mult in enumerate(m["multipliers"]):
            lines.append(
                f"| {i} | x{mult} | {m['train_mse_mean'][i]:.3e} | "
                f"{m['test_mse_mean'][i]:.3e} | {m['r2_mean'][i] * 100:.4f} |\n"
            )
        lines.append("\n")

    path = os.path.join(RESULTS, "SUMMARY.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\n  Summary written: {path}")


def main():
    print("=" * 72)
    print("OPTION PRICING WITH NEURAL NETWORKS - FULL RUN")
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    t_total = time.time()
    run_times = {}
    failures = []

    for short_name, module_path, title in EXPERIMENTS:
        ok, dt = run_experiment(short_name, module_path, title)
        run_times[short_name] = dt
        if not ok:
            failures.append(short_name)

    total = time.time() - t_total
    print("\n" + "=" * 72)
    print(f"ALL EXPERIMENTS COMPLETE - total time: {total / 3600:.2f} h")
    if failures:
        print(f"FAILURES ({len(failures)}): {', '.join(failures)}")
    else:
        print("All experiments succeeded.")
    print("=" * 72)

    build_summary(run_times)


if __name__ == "__main__":
    main()
