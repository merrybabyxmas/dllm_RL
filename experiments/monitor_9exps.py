#!/usr/bin/env python3
"""Monitor progress of the official experiments."""
import datetime, json, os, subprocess
from pathlib import Path

OUT = Path("/home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl/experiments/outputs/official_9exp")
LOG = Path("/home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl/experiments/outputs/official_9exp")
DATASETS = ["mbpp", "humaneval", "svamp"]
METHODS  = ["baseline", "diffu_grpo", "stage2"]

def gpu_info():
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader",
            shell=True, capture_output=True, text=True
        )
        return r.stdout.strip()
    except:
        return "N/A"

def tail_log(log_path, n=5):
    try:
        lines = Path(log_path).read_text().splitlines()
        return lines[-n:]
    except:
        return []

def read_result(ds, method):
    p = OUT / ds / method / "result.json"
    if p.exists():
        return json.loads(p.read_text())
    return None

ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n{'='*70}")
print(f"9-Experiment Monitor  |  {ts}")
print(f"GPU: {gpu_info()}")
print(f"{'='*70}")

# Results table
print("\n--- Results (completed) ---")
print(f"{'':15}  {'baseline':>10}  {'diffu_grpo':>10}  {'stage2':>10}")
for ds in DATASETS:
    row = f"{ds:15}"
    for m in METHODS:
        r = read_result(ds, m)
        if r:
            row += f"  {r['mean_score']:>10.4f}"
        else:
            row += f"  {'--':>10}"
    print(row)

# Per-experiment status
print("\n--- Per-experiment status ---")
for ds in DATASETS:
    for m in METHODS:
        lf  = LOG / f"{ds}_{m}.log"
        res = read_result(ds, m)
        if res:
            print(f"  [{ds}/{m}] DONE  score={res['mean_score']:.4f}  "
                  f"train={res.get('train_time_s',0)/3600:.1f}h")
        elif lf.exists():
            lines = tail_log(lf, 3)
            last  = lines[-1] if lines else "(empty)"
            print(f"  [{ds}/{m}] RUNNING")
            print(f"    {last}")
        else:
            print(f"  [{ds}/{m}] NOT STARTED")

# Check PIDs
print("\n--- Running Python processes ---")
try:
    r = subprocess.run(
        "ps aux | grep run_experiment.py | grep -v grep",
        shell=True, capture_output=True, text=True
    )
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        pid   = parts[1]
        cmd   = " ".join(parts[10:])
        print(f"  PID {pid}: {cmd[:80]}")
except:
    pass
print()
