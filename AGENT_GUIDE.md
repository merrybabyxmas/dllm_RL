# cc_rl Agent Execution Guide

> **Current status (2026-06-27):** 9 official experiments underway (3 datasets × 3 methods).  
> 5 complete, 4 remain. Remote agent should pick up from **Section 5 – Resume remaining runs**.

---

## Experiment matrix

| Dataset | baseline | diffu_grpo | stage2 |
|---------|----------|------------|--------|
| mbpp | ✅ 0.1889 | ✅ 0.2333 | ✅ 0.2333 |
| humaneval | ✅ 0.1768 | ✅ 0.1707 | ✅ 0.2012 |
| svamp | ✅ 0.8050 | ✅ 0.8300 | ✅ 0.8300 |
| gsm8k | ✅ 0.1865 | ❌ missing | ❌ missing |
| spider | ✅ 0.0130 | ❌ missing | ❌ missing |

**4 runs left: `gsm8k/diffu_grpo`, `gsm8k/stage2`, `spider/diffu_grpo`, `spider/stage2`**

---

## 1. One-time environment setup

```bash
# ── Clone ──────────────────────────────────────────────────────────────────
git clone https://github.com/merrybaxyxmas/dllm_RL.git confidence_credit_dllm_rl
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

# d1 package
pip install -e ../d1/diffu-grpo

# cc_rl package
pip install -e .
```

---

## 2. Model

Download **LLaDA-8B-Instruct** (≈16 GB):

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ../LLaDA-8B-Instruct
```

Expected path: `../LLaDA-8B-Instruct/` (one level above the repo root).  
To override, set `_MODEL_PATH` in `experiments/run_experiment.py` line ~40.

---

## 3. Datasets

All loaded automatically from HuggingFace Hub at runtime.  
For offline servers, pre-cache on a connected machine:

```bash
python3 -c "
from datasets import load_dataset
for ds in ['openai/gsm8k', 'spider']:
    load_dataset(ds)
    print(f'cached: {ds}')
"
```

---

## 4. Smoke test (verify before training)

```bash
cd confidence_credit_dllm_rl
PYTHONPATH=src:../d1/diffu-grpo python3 experiments/smoke_test.py
```

Expected: `=== ALL SMOKE TESTS PASSED ✅ ===`

---

## 5. Resume remaining 4 runs (START HERE for remote agent)

The launcher skips experiments that already have `result.json`, so running it again is safe.

```bash
cd confidence_credit_dllm_rl

conda activate cc_rl

# Run remaining 4 experiments sequentially (one GPU)
# Expected total time: ~130h (gsm8k ~56h each, spider ~75h each)
bash experiments/launch_fast.sh 2>&1 | tee experiments/outputs/fast_launch_resume.log
```

Or run them individually in parallel (if multiple GPUs available):

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false

# GPU 0: gsm8k/diffu_grpo
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiment.py \
    --dataset gsm8k --method diffu_grpo \
    --max_value_states 2 \
    --output_dir experiments/outputs/official_9exp \
    > experiments/outputs/gsm8k_diffu_grpo.log 2>&1 &

# GPU 1: gsm8k/stage2
CUDA_VISIBLE_DEVICES=1 python experiments/run_experiment.py \
    --dataset gsm8k --method stage2 \
    --max_value_states 2 \
    --output_dir experiments/outputs/official_9exp \
    > experiments/outputs/gsm8k_stage2.log 2>&1 &

# GPU 2: spider/diffu_grpo
CUDA_VISIBLE_DEVICES=2 python experiments/run_experiment.py \
    --dataset spider --method diffu_grpo \
    --max_value_states 2 \
    --output_dir experiments/outputs/official_9exp \
    > experiments/outputs/spider_diffu_grpo.log 2>&1 &

# GPU 3: spider/stage2
CUDA_VISIBLE_DEVICES=3 python experiments/run_experiment.py \
    --dataset spider --method stage2 \
    --max_value_states 2 \
    --output_dir experiments/outputs/official_9exp \
    > experiments/outputs/spider_stage2.log 2>&1 &

wait
echo "All 4 experiments done"
```

---

## 6. Monitor progress

```bash
# Check how many result.json files exist (target: 15 total = 5 datasets × 3 methods)
find experiments/outputs/official_9exp -name "result.json" | wc -l

# Print all completed scores
for f in $(find experiments/outputs/official_9exp -name "result.json" | sort); do
    score=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d.get('mean_score','?'):.4f}\")" 2>/dev/null || echo "?")
    echo "$score  $f"
done

# Live tail for a specific run
tail -f experiments/outputs/gsm8k_stage2.log
```

Streamlit dashboard (if installed):

```bash
streamlit run experiments/dashboard.py --server.port 8503 --server.headless true &
ssh -R 80:localhost:8503 serveo.net &
# → URL printed: https://XXXX.serveousercontent.com
```

---

## 7. Output structure

```
experiments/outputs/official_9exp/
├── {dataset}/
│   ├── baseline/result.json
│   ├── diffu_grpo/result.json
│   └── stage2/result.json
└── {dataset}_{method}.log
```

Each `result.json` contains:
```json
{
  "method": "stage2",
  "dataset": "gsm8k",
  "mean_score": 0.823,
  "n_eval": 1319,
  "eval_history": [[100, 0.71], [200, 0.79]],
  "train_time_s": 14832.5,
  "eval_time_s": 421.3,
  "total_time_s": 15253.8
}
```

---

## 8. Hyperparameters

All training methods use identical config for fair comparison:

| Parameter | Value |
|-----------|-------|
| Model | LLaDA-8B-Instruct (bf16) |
| `learning_rate` | 1e-6 |
| `num_generations` | 4 |
| `max_completion_length` | 256 |
| `diffusion_steps` | 64 |
| `block_length` | 32 |
| `beta` (KL penalty) | 0.04 |
| `epsilon` (PPO clip) | 0.2 |
| epochs | 1 |
| `max_value_states` | 2 |

Stage 2 extras: `value_hidden_size=256`, `critic_lr=5e-6`, `critic_loss_coef=0.5`, `delta_v_gate=0.01`

---

## 9. Known issues

| Issue | Fix |
|-------|-----|
| OOM during generate | `generate_with_confidence` has `@torch.no_grad()` — already patched |
| `dtype` mismatch (bf16/fp32) | `hidden.float()` cast before value_head — already patched |
| Spider `DatasetNotFoundError` | Usually cache miss; re-run once with internet access |
| Crash mid-run | Re-run the same command — skips completed `result.json` |
| `trust_remote_code` warning | Harmless FutureWarning, ignore |

---

## 10. Repo structure

```
confidence_credit_dllm_rl/
├── src/cc_rl/
│   ├── algorithms/
│   │   ├── stage1_cw_grpo.py         # CWGRPOTrainer (confidence weighting)
│   │   ├── stage2_value_credit.py    # ValueCreditTrainer (delta-V credit)
│   │   └── stage3_q_credit.py        # QCreditTrainer (Q-value, future)
│   └── data/
│       ├── humaneval.py
│       └── svamp.py
├── experiments/
│   ├── run_experiment.py             # main entry point
│   ├── launch_fast.sh               # sequential launcher (skips done)
│   ├── launch_9exps.sh              # 2-at-a-time parallel launcher
│   ├── monitor_9exps.py             # progress monitor
│   ├── dashboard.py                  # Streamlit dashboard
│   └── smoke_test.py                # end-to-end verification
└── AGENT_GUIDE.md                   # this file
```

---

## 11. After experiments finish

Collect results into a summary table:

```bash
python3 - <<'EOF'
import json, os

datasets = ["mbpp", "humaneval", "svamp", "gsm8k", "spider"]
methods  = ["baseline", "diffu_grpo", "stage2"]
base     = "experiments/outputs/official_9exp"

header = f"{'Dataset':<12}" + "".join(f"{m:>12}" for m in methods)
print(header)
print("-" * len(header))
for ds in datasets:
    row = f"{ds:<12}"
    for m in methods:
        f = f"{base}/{ds}/{m}/result.json"
        if os.path.exists(f):
            s = json.load(open(f)).get("mean_score", 0)
            row += f"{s:>12.4f}"
        else:
            row += f"{'--':>12}"
    print(row)
EOF
```
