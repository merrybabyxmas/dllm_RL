# cc_rl Main Experiment – Agent Execution Guide

> Target: 4x RTX 4090 server (24 GB VRAM each)  
> Goal: reproduce `baseline` vs `delta_v_only` across 6 datasets × 3 gen-lengths (36 runs total)

---

## 1. One-time environment setup

```bash
# ── Clone ──────────────────────────────────────────────────────────────────
git clone https://github.com/merrybabyxmas/dllm_RL.git confidence_credit_dllm_rl
cd confidence_credit_dllm_rl

# ── d1 (DiffuGRPO base trainer) ────────────────────────────────────────────
git clone https://github.com/HKUNLP/d1.git ../d1

# ── Conda env (Python 3.10 recommended) ────────────────────────────────────
conda create -n cc_rl python=3.10 -y
conda activate cc_rl

# ── Python dependencies ────────────────────────────────────────────────────
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.0 peft trl==1.6.0 accelerate datasets
pip install streamlit plotly pandas numpy

# d1 package install
pip install -e ../d1/diffu-grpo

# cc_rl package install
pip install -e .
```

---

## 2. Model

Download **LLaDA-8B-Instruct** (≈16 GB) and place (or symlink) at:

```
../LLaDA-8B-Instruct/
```

Or edit `_MODEL_PATH` in `experiments/run_experiment.py` line ~40:

```python
_MODEL_PATH = "/your/path/to/LLaDA-8B-Instruct"
```

The model is publicly available on HuggingFace:
```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ../LLaDA-8B-Instruct
```

---

## 3. Datasets

All datasets are loaded automatically at runtime (HuggingFace Hub + local cache).  
No manual download needed. On first run each dataset is fetched once:

| Dataset | Source | Notes |
|---------|--------|-------|
| `countdown` | `Jiayi-Pan/Countdown-Tasks-3to4` | 490K examples, 90/10 split |
| `gsm8k` | `openai/gsm8k` | 7.4K train / 1.3K eval |
| `mbpp` | `google-research-datasets/mbpp` | sanitized split |
| `humaneval` | `openai/openai_humaneval` | code generation |
| `svamp` | `ChilleD/SVAMP` | math word problems |
| `spider` | `spider` | text-to-SQL |

If the server has no internet access, pre-download on a connected machine:
```bash
python3 -c "
from datasets import load_dataset
for ds in ['Jiayi-Pan/Countdown-Tasks-3to4', 'openai/gsm8k',
           'google-research-datasets/mbpp', 'openai/openai_humaneval',
           'ChilleD/SVAMP', 'spider']:
    load_dataset(ds)
    print(f'cached: {ds}')
"
```

---

## 4. Smoke test (verify before full run)

Takes ~5 minutes, uses GPU.

```bash
cd confidence_credit_dllm_rl

PYTHONPATH=src:../d1/diffu-grpo \
  python3 experiments/smoke_test.py
```

Expected output ends with:
```
=== ALL SMOKE TESTS PASSED ✅ ===
```

If any test fails, do NOT start the full launcher — diagnose first.

---

## 5. Launch main experiments

```bash
cd confidence_credit_dllm_rl

# Review config at the top of the script (NUM_GPUS, GPU_IDS, etc.)
head -30 experiments/launch_main_experiments.sh

# Launch (runs in foreground; use tmux/screen for long jobs)
bash experiments/launch_main_experiments.sh
```

**Default config** (edit top of script if needed):

```bash
NUM_GPUS=4
GPU_IDS="0,1,2,3"
METHODS=("baseline" "delta_v_only")
DATASETS=("mbpp" "humaneval" "svamp" "gsm8k" "countdown" "spider")
GEN_LENGTHS=("128" "256" "512")
MAX_TRAIN=10000        # cap training examples per dataset
```

**What it does:**
- Builds a queue of 36 experiments (2 methods × 6 datasets × 3 gen-lengths)
- Assigns each to a free GPU via `CUDA_VISIBLE_DEVICES=N`
- Skips any experiment where `result.json` already exists (safe to restart)
- Logs each run to `experiments/outputs/main_experiments/launcher_logs/<name>.log`
- Prints a summary table when all runs complete

**Expected wall-clock** (4x RTX 4090, 24h budget):
- `baseline` (inference only): ~1-2h per dataset × gen_length
- `delta_v_only` (1-epoch RL): ~6-12h per dataset × gen_length
- Total: ~20-30h with 4 GPUs in parallel

---

## 6. Monitor progress

### Option A – terminal log tail
```bash
# Live log for a specific run
tail -f experiments/outputs/main_experiments/launcher_logs/gsm8k_gl256_delta_v_only.log

# Check how many results are done
find experiments/outputs/main_experiments -name "result.json" | wc -l

# Print all completed scores
for f in $(find experiments/outputs/main_experiments -name "result.json" | sort); do
    score=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['mean_score']:.4f}\")")
    echo "$score  $f"
done
```

### Option B – Streamlit dashboard (port 8503)
```bash
# Start dashboard
streamlit run experiments/dashboard_main.py \
    --server.port 8503 \
    --server.headless true \
    --server.address 0.0.0.0 &

# External access via serveo.net (no auth needed)
ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -R 80:localhost:8503 serveo.net &
# → URL printed: https://XXXX.serveousercontent.com
```

---

## 7. Output structure

```
experiments/outputs/main_experiments/
├── launcher_logs/
│   ├── gsm8k_gl128_baseline.log
│   ├── gsm8k_gl128_delta_v_only.log
│   └── ...
├── gsm8k/
│   ├── gl128/
│   │   ├── baseline/result.json
│   │   └── delta_v_only/result.json
│   ├── gl256/
│   └── gl512/
├── countdown/
│   └── ...
└── ...
```

Each `result.json`:
```json
{
  "method": "delta_v_only",
  "dataset": "gsm8k",
  "mean_score": 0.823,
  "n_eval": 1319,
  "eval_history": [[100, 0.71], [200, 0.79], ...],
  "train_time_s": 14832.5,
  "eval_time_s": 421.3,
  "total_time_s": 15253.8,
  "gen_length": 256
}
```

---

## 8. Known issues & fixes

| Issue | Fix |
|-------|-----|
| `DatasetNotFoundError` for countdown | Normal — uses cached version. Not an error if cache exists. |
| `trust_remote_code` warning on model load | Harmless FutureWarning, ignore |
| `torch_dtype deprecated` | Harmless FutureWarning, ignore |
| OOM on humaneval/gen_length=512 | Reduce `num_generations` in `build_config()` from 4 to 2 |
| Two experiments on same GPU → OOM | `wait_for_free_gpu()` in launcher prevents this; safe |
| Restart after crash | Re-run `bash experiments/launch_main_experiments.sh` — skips completed runs |

---

## 9. Single-run manual command (for debugging)

```bash
cd confidence_credit_dllm_rl

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:../d1/diffu-grpo \
TOKENIZERS_PARALLELISM=false \
  python3 experiments/run_experiment.py \
    --dataset gsm8k \
    --method delta_v_only \
    --gen_length 256 \
    --output_dir experiments/outputs/main_experiments
```

Supported values:
- `--dataset`: `gsm8k` `countdown` `mbpp` `humaneval` `svamp` `spider`
- `--method`: `baseline` `delta_v_only`
- `--gen_length`: `128` `256` `512`
- `--max_train_examples`: default 100000 (cap training set size)

---

## 10. Repo structure (key files)

```
confidence_credit_dllm_rl/
├── src/cc_rl/
│   ├── algorithms/
│   │   ├── stage1_cw_grpo.py       # CWGRPOTrainer (confidence weighting)
│   │   └── stage2_value_credit.py  # ValueCreditTrainer (delta-V, our method)
│   └── data/
│       ├── humaneval.py
│       └── svamp.py
├── experiments/
│   ├── run_experiment.py           # main entry point (all datasets + methods)
│   ├── launch_main_experiments.sh  # 4-GPU parallel launcher
│   ├── dashboard_main.py           # streamlit dashboard
│   └── smoke_test.py               # end-to-end verification
└── AGENT_GUIDE.md                  # this file
```
