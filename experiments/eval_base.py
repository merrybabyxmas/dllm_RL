"""
Evaluate base LLaDA-8B-Instruct on GSM8K (no training).

Usage:
  python eval_base.py --model_path /path/to/LLaDA-8B-Instruct --n_examples 256
"""
import sys
import os
import argparse
import json
import re
import torch
import numpy as np
from tqdm import tqdm

_D1_PATH = os.path.join(os.path.dirname(__file__), "../../d1/diffu-grpo")
_D1_PATH = os.path.abspath(_D1_PATH)
if _D1_PATH not in sys.path:
    sys.path.insert(0, _D1_PATH)

_CC_RL_SRC = os.path.join(os.path.dirname(__file__), "../src")
_CC_RL_SRC = os.path.abspath(_CC_RL_SRC)
if _CC_RL_SRC not in sys.path:
    sys.path.insert(0, _CC_RL_SRC)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def add_gumbel_noise(logits, temperature, dtype):
    if temperature == 0.0:
        return logits
    logits = logits.to(dtype)
    noise = torch.rand_like(logits, dtype=dtype)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def generate_diffusion(model, prompt, tokenizer, gen_length=256, steps=64, block_length=32,
                       temperature=0.0, remasking="low_confidence", mask_id=126336, device="cuda"):
    model.eval()
    with torch.no_grad():
        bs = prompt.shape[0]
        dtype = model.dtype
        total_len = prompt.shape[1] + gen_length
        x = torch.full((bs, total_len), mask_id, dtype=torch.long, device=device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = x != mask_id

        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask = x[:, start_idx:end_idx] == mask_id
            mask_num = block_mask.sum(dim=1, keepdim=True)
            base = mask_num // steps_per_block
            remainder = mask_num % steps_per_block
            num_transfer = base.expand(-1, steps_per_block).clone()
            for j in range(bs):
                for s in range(remainder[j, 0].item()):
                    num_transfer[j, s] += 1

            for i in range(steps_per_block):
                mask_index = x == mask_id
                logits = model(x).logits
                logits_n = add_gumbel_noise(logits, temperature, dtype)
                x0 = torch.argmax(logits_n, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(dtype), dim=-1)
                    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                else:
                    x0_p = torch.rand_like(x0, dtype=torch.float)

                x0_p[:, end_idx:] = -float("inf")
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(bs):
                    n = num_transfer[j, i].item()
                    if n > 0:
                        _, idx = torch.topk(confidence[j], k=int(n))
                        transfer_index[j, idx] = True

                x[transfer_index] = x0[transfer_index]

        return x


def extract_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        val = m.group(1).strip()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", val)
        if nums:
            return nums[-1].replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def extract_gold(answer_str):
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_str.replace(",", ""))
    if nums:
        return nums[-1]
    return None


def main():
    # Clear expired HF token so public datasets load without auth errors
    import os
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct")
    parser.add_argument("--n_examples", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/base_eval_gsm8k.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.config.use_cache = False
    model.eval()

    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42)
    if args.n_examples:
        ds = ds.select(range(min(args.n_examples, len(ds))))

    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        mask_id = 126336  # LLaDA default

    results = []
    correct = 0

    for i, example in enumerate(tqdm(ds, desc="Evaluating")):
        question = example["question"]
        gold = example["answer"]
        gold_num = extract_gold(gold)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(args.device)

        # Trim if too long
        if prompt_ids.shape[1] > 256:
            prompt_ids = prompt_ids[:, -256:]

        out_ids = generate_diffusion(
            model, prompt_ids, tokenizer,
            gen_length=args.gen_length,
            steps=args.diffusion_steps,
            block_length=32,
            temperature=0.0,
            mask_id=mask_id,
            device=args.device,
        )

        completion_ids = out_ids[:, prompt_ids.shape[1]:]
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        pred_num = extract_answer(completion_text)

        is_correct = False
        if pred_num is not None and gold_num is not None:
            try:
                is_correct = abs(float(pred_num) - float(gold_num)) < 1e-6
            except:
                is_correct = str(pred_num).strip() == str(gold_num).strip()

        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "gold": gold,
            "gold_num": gold_num,
            "completion": completion_text,
            "pred_num": pred_num,
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(ds)}] Running acc: {acc:.3f}")

    accuracy = correct / len(ds)
    summary = {
        "method": "base",
        "model": args.model_path,
        "n_examples": len(ds),
        "accuracy": accuracy,
        "correct": correct,
    }
    print(f"\n===== BASE EVAL RESULT =====")
    print(f"  GSM8K Accuracy: {accuracy:.4f} ({correct}/{len(ds)})")

    output = {"summary": summary, "results": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
