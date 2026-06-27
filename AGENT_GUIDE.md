# cc_rl Agent Execution Guide

> **Repo:** https://github.com/merrybabyxmas/dllm_RL  
> **Target hardware:** 4× RTX 4090 (24 GB each)  
> **Goal:** 36 experiments — 6 datasets × 2 methods × 3 gen-lengths

---

## Experiment matrix

| Dataset | baseline ×3 GL | delta_v_only ×3 GL |
|---------|----------------|---------------------|
| mbpp | ✅ prev runs | (new run) |
| humaneval | ✅ prev runs | (new run) |
| svamp | ✅ prev runs | (new run) |
| gsm8k | (new run) | (new run) |
| countdown | (new run) | (new run) |
| spider | (new run) | (new run) |

Gen lengths: 128, 256, 512 → **36 total experiments**  
Output: `experiments/outputs/main_experiments/{dataset}/gl{128,256,512}/{method}/result.json`

---

## 1. One-time setup

```bash
# Clone
git clone https://github.com/merrybabyxmas/dllm_RL.git confidence_credit_dllm_rl
cd confidence_credit_dllm_rl

# d1 base trainer
git clone https://github.com/HKUNLP/d1.git ../d1

# Conda env
conda create -n cc_rl python=3.10 -y
conda activate cc_rl

# Python deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.0 peft trl==1.6.0 accelerate datasets
pip install streamlit plotly pandas numpy

# Install packages
pip install -e ../d1/diffu-grpo
pip install -e .
```

---

## 2. Model

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ../LLaDA-8B-Instruct
```

Expected at `../LLaDA-8B-Instruct/` (one level above repo root).  
To override: edit `_MODEL_PATH` in `experiments/run_experiment.py` line ~47.

---

## 3. Datasets

All loaded from HuggingFace Hub automatically. For offline servers:

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

## 4. Smoke test

```bash
PYTHONPATH=src:../d1/diffu-grpo python3 experiments/smoke_test.py
```

Expected: `=== ALL SMOKE TESTS PASSED ✅ ===`  
If any test fails, diagnose before launching full experiments.

---

## 5. Launch all 36 experiments (START HERE)

The launcher uses a 4-GPU pool. Each slot runs one experiment at a time.  
Already-complete `result.json` files are skipped automatically (safe to restart).

```bash
conda activate cc_rl
cd confidence_credit_dllm_rl

# Recommended: run inside tmux or screen
tmux new -s cc_rl_exp
bash experiments/launch_main_experiments.sh 2>&1 | tee experiments/outputs/main_experiments_launch.log
```

**Config at top of the script** (edit if needed):

```bash
NUM_GPUS=4
GPU_IDS="0,1,2,3"
METHODS=("baseline" "delta_v_only")
DATASETS=("mbpp" "humaneval" "svamp" "gsm8k" "countdown" "spider")
GEN_LENGTHS=("128" "256" "512")
MAX_TRAIN=10000
```

**Expected wall-clock (4× RTX 4090):**

| Dataset | baseline (eval only) | delta_v_only (1-epoch train) |
|---------|----------------------|------------------------------|
| mbpp | ~0.5h | ~4h |
| humaneval | ~0.3h | ~2h |
| svamp | ~0.5h | ~4h |
| gsm8k | ~1h | ~18h |
| countdown | ~1h | ~20h |
| spider | ~1h | ~25h |

Total with 4 GPUs in parallel: ~25-35h.

---

## 6. Manual single run (debugging)

```bash
cd confidence_credit_dllm_rl
export PYTHONPATH="$(pwd)/src:$(pwd)/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python3 experiments/run_experiment.py \
    --dataset gsm8k \
    --method delta_v_only \
    --gen_length 256 \
    --max_train_examples 10000 \
    --max_value_states 2 \
    --output_dir experiments/outputs/main_experiments
```

Supported values:
- `--dataset`: `gsm8k` `mbpp` `humaneval` `svamp` `countdown` `spider`
- `--method`: `baseline` `delta_v_only`
- `--gen_length`: `128` `256` `512`

---

## 7. Monitor progress

```bash
# Count completed (target: 36)
find experiments/outputs/main_experiments -name "result.json" | wc -l

# Print all scores
for f in $(find experiments/outputs/main_experiments -name "result.json" | sort); do
    score=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d.get('mean_score',0):.4f}\")" 2>/dev/null)
    echo "$score  $f"
done

# Live log for a specific run
tail -f experiments/outputs/main_experiments/launcher_logs/gsm8k_gl256_delta_v_only.log
```

Streamlit dashboard:

```bash
streamlit run experiments/dashboard.py --server.port 8503 --server.headless true &
ssh -R 80:localhost:8503 serveo.net &
# prints → https://XXXX.serveousercontent.com
```

---

## 8. Output structure

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
├── mbpp/
└── ...
```

Each `result.json`:

```json
{
  "method": "delta_v_only",
  "dataset": "gsm8k",
  "mean_score": 0.231,
  "n_eval": 1319,
  "eval_history": [[100, 0.19], [500, 0.22]],
  "train_time_s": 14832.5,
  "eval_time_s": 421.3,
  "total_time_s": 15253.8,
  "gen_length": 256
}
```

---

## 9. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | LLaDA-8B-Instruct (bf16) |
| `learning_rate` | 1e-6 |
| `num_generations` | 4 |
| `diffusion_steps` | 64 |
| `block_length` | 32 |
| `beta` (KL) | 0.04 |
| `epsilon` (PPO clip) | 0.2 |
| `max_prompt_length` | 256 |
| `max_value_states` | **2** (set by launcher for OOM safety) |
| epochs | 1 |
| `max_train_examples` | 10000 per dataset |

---

## 10. Known issues

| Issue | Fix |
|-------|-----|
| OOM during generate | `@torch.no_grad()` already patched in `generate_with_confidence` |
| bf16/fp32 dtype mismatch | `hidden.float()` cast already patched in value head |
| Spider `DatasetNotFoundError` | Cache miss — re-run with internet |
| Crash mid-run | Re-run launcher — already-done experiments are skipped |
| `trust_remote_code` warning | Harmless FutureWarning |
| gen_length=512 OOM on 4090 | `--max_value_states 2` already in launcher |

---

## 11. Summary table after completion

```bash
python3 - <<'EOF'
import json, os

datasets   = ["mbpp", "humaneval", "svamp", "gsm8k", "countdown", "spider"]
methods    = ["baseline", "delta_v_only"]
gen_lengths = [128, 256, 512]
base       = "experiments/outputs/main_experiments"

for gl in gen_lengths:
    print(f"\n=== gen_length={gl} ===")
    header = f"{'Dataset':<12}" + "".join(f"{m:>14}" for m in methods)
    print(header)
    print("-" * len(header))
    for ds in datasets:
        row = f"{ds:<12}"
        for m in methods:
            f = f"{base}/{ds}/gl{gl}/{m}/result.json"
            if os.path.exists(f):
                s = json.load(open(f)).get("mean_score", 0)
                row += f"{s:>14.4f}"
            else:
                row += f"{'--':>14}"
        print(row)
EOF
```

---

## 12. Repo structure

```
confidence_credit_dllm_rl/
├── src/cc_rl/
│   ├── algorithms/
│   │   ├── stage1_cw_grpo.py          # CWGRPOTrainer (base)
│   │   ├── stage2_value_credit.py     # ValueCreditTrainer (delta_v_only uses this)
│   │   └── stage3_q_credit.py
│   └── data/
│       ├── humaneval.py
│       └── svamp.py
├── experiments/
│   ├── run_experiment.py              # main entry point
│   ├── launch_main_experiments.sh    # 4-GPU parallel launcher (START HERE)
│   ├── monitor_9exps.py              # progress monitor
│   ├── dashboard.py                  # Streamlit dashboard
│   └── smoke_test.py                 # end-to-end verification
└── AGENT_GUIDE.md                    # this file
```
