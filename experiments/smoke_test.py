"""
Smoke test: verifies countdown+gsm8k baseline inference + 3-step delta_v_only training.
Run: PYTHONPATH=src:../d1/diffu-grpo python3 experiments/smoke_test.py
"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, '/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo')
sys.path.insert(0, os.path.dirname(__file__))

from run_experiment import (
    load_countdown, load_gsm8k,
    score_completion, build_baseline_model, build_base_model,
    _diffusion_generate, _MASK_ID, _MODEL_PATH,
    build_config, TRAIN_REWARD_FUNCS,
    ValueCreditTrainer,
)
from transformers import AutoTokenizer
from peft import get_peft_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_LENGTH = 128

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_baseline_inference(dataset_name, n_eval=5):
    section(f"Baseline Inference — {dataset_name} ({n_eval} examples)")

    if dataset_name == "countdown":
        _, eval_rows = load_countdown(seed=42, max_train=10)
    else:
        _, eval_rows = load_gsm8k(seed=42, max_train=10)
    eval_rows = eval_rows[:n_eval]

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = build_baseline_model(_MODEL_PATH, DEVICE)
    model.eval()

    scores = []
    for i, ex in enumerate(eval_rows):
        t0 = time.time()
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = enc["input_ids"].to(DEVICE)[:, -256:]

        full_ids = _diffusion_generate(model, prompt_ids,
                                       gen_length=GEN_LENGTH, block_length=32,
                                       steps=64, mask_id=_MASK_ID)
        comp_text = tokenizer.decode(full_ids[0, prompt_ids.size(1):], skip_special_tokens=True)
        sc = score_completion(comp_text, ex, dataset_name)
        scores.append(sc)
        elapsed = time.time() - t0

        if dataset_name == "countdown":
            extra = f"nums={ex['nums']} target={ex['target']}"
        else:
            extra = f"answer={str(ex.get('answer',''))[:30]}"
        print(f"  [{i+1}/{n_eval}]  score={sc:.1f}  t={elapsed:.1f}s  {extra}")
        print(f"    completion: {comp_text[:100].strip()!r}")

    print(f"\n  mean_score={sum(scores)/len(scores):.4f}  (n={len(scores)})")
    del model
    torch.cuda.empty_cache()
    return scores


def test_training_steps(dataset_name, n_train=8, n_steps=3):
    section(f"delta_v_only Training ({n_steps} steps) — {dataset_name}")

    if dataset_name == "countdown":
        train_ds, eval_rows = load_countdown(seed=42, max_train=n_train)
    else:
        train_ds, eval_rows = load_gsm8k(seed=42, max_train=n_train)
    eval_rows = eval_rows[:3]

    print(f"  train={len(train_ds)}  eval_subset={len(eval_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _, peft_config = build_base_model(_MODEL_PATH, DEVICE)

    cfg = build_config("delta_v_only", dataset_name,
                       output_dir=f"/tmp/smoke_{dataset_name}",
                       seed=42, num_train_examples=n_train, gen_length=GEN_LENGTH)
    # Override: only run n_steps
    cfg.max_steps = n_steps
    cfg.num_train_epochs = 9999  # ignored when max_steps set
    cfg.logging_steps = 1
    cfg.save_steps = 9999
    cfg.eval_steps = 9999

    _CW_KWARGS = dict(credit_alpha=1.0, credit_eps=1e-6,
                      credit_clip_min=0.25, credit_clip_max=4.0)
    _DV_KWARGS = dict(value_hidden_size=256, value_mlp_layers=2,
                      critic_lr=5e-6, critic_loss_coef=0.5, delta_v_gate=0.01,
                      max_value_states=2)

    trainer = ValueCreditTrainer(
        model=model,
        reward_funcs=TRAIN_REWARD_FUNCS[dataset_name],
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        use_confidence_weight=False,
        **_CW_KWARGS,
        **_DV_KWARGS,
    )

    # TRL compatibility patches
    if not hasattr(trainer, "max_prompt_length") or trainer.max_prompt_length is None:
        trainer.max_prompt_length = cfg.max_prompt_length
    if not hasattr(trainer, "_buffered_inputs") or trainer._buffered_inputs is None:
        ga = max(1, cfg.gradient_accumulation_steps)
        trainer._buffered_inputs = [None] * ga
    if not hasattr(trainer, "epsilon"):
        trainer.epsilon = getattr(trainer, "epsilon_low", cfg.epsilon)
    if not hasattr(trainer, "log_completions"):
        trainer.log_completions = getattr(cfg, "log_completions", False)

    print(f"  Starting training ({n_steps} steps) ...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # Quick inference check post-training
    print(f"  Post-training inference on {len(eval_rows)} examples ...")
    model.eval()
    scores = []
    from trl.models import unwrap_model_for_generation
    for ex in eval_rows:
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = enc["input_ids"].to(DEVICE)[:, -cfg.max_prompt_length:]
        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as uw:
            full_ids = trainer.generate(
                model=uw,
                prompt=prompt_ids,
                steps=cfg.diffusion_steps,
                gen_length=cfg.max_completion_length,
                block_length=cfg.block_length,
                temperature=0.0,
                mask_id=_MASK_ID,
            )
        comp = tokenizer.decode(full_ids[0, prompt_ids.size(1):], skip_special_tokens=True)
        sc = score_completion(comp, ex, dataset_name)
        scores.append(sc)
        print(f"    score={sc:.1f}  comp={comp[:80].strip()!r}")

    del model, trainer
    torch.cuda.empty_cache()
    return scores


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  cc_rl SMOKE TEST")
    print("="*60)

    results = {}

    # Phase 1: Baseline inference
    for ds in ["countdown", "gsm8k"]:
        sc = test_baseline_inference(ds, n_eval=5)
        results[f"baseline_{ds}"] = sc

    # Phase 2: delta_v_only training (3 steps)
    for ds in ["countdown", "gsm8k"]:
        sc = test_training_steps(ds, n_train=8, n_steps=3)
        results[f"delta_v_only_{ds}"] = sc

    section("SMOKE TEST SUMMARY")
    all_ok = True
    for k, v in results.items():
        status = "✅ OK" if v is not None else "❌ FAILED"
        print(f"  {status}  {k}  scores={[f'{s:.1f}' for s in v]}")
    print("\nSmoke test complete.")
