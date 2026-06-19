"""
DiffusionSampler: wraps a masked diffusion LLM and produces TrajectoryRecord
objects with per-step confidence scores recorded during generation.

The confidence score for each revealed token equals the softmax probability
of the chosen token at the denoising step in which it was committed.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cc_rl.rollout.trajectory import TrajectoryRecord, TrajectoryStep


class DiffusionSampler:
    """
    Generates completions from a masked diffusion LLM and records per-token
    confidence scores for downstream credit assignment.

    Parameters
    ----------
    model       : The masked-diffusion model (e.g., LLaDA-8B-Instruct).
    tokenizer   : Tokenizer matching the model vocabulary.
    mask_id     : Vocabulary index of the [MASK] token (default 126336 for LLaDA).
    device      : torch.device on which to run generation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        mask_id: int = 126336,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_group(
        self,
        prompt_ids: torch.Tensor,   # [1, prompt_len] or [batch, prompt_len]
        prompt_id: str,
        n_samples: int = 8,
        gen_length: int = 128,
        steps: int = 64,
        block_length: int = 64,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
    ) -> List[TrajectoryRecord]:
        """
        Draw n_samples completions for the same prompt, returning TrajectoryRecord
        objects with per-token confidence metadata.
        """
        # Expand prompt to n_samples
        if prompt_ids.shape[0] == 1:
            batch_prompt = prompt_ids.expand(n_samples, -1)
        else:
            assert prompt_ids.shape[0] == n_samples
            batch_prompt = prompt_ids

        completion_ids, token_confidence = self._generate_with_confidence(
            prompt=batch_prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
        )

        prompt_len = batch_prompt.shape[1]
        records = []
        for i in range(n_samples):
            comp = completion_ids[i, prompt_len:]           # [gen_length]
            conf = token_confidence[i, prompt_len:]         # [gen_length]
            final_text = self.tokenizer.decode(comp, skip_special_tokens=True)

            steps_list = []
            for t in range(gen_length):
                state = f"step_{t}"
                action = comp[t].item()
                next_state = f"step_{t + 1}"
                step = TrajectoryStep(
                    prompt_id=prompt_id,
                    sample_id=i,
                    step_idx=t,
                    state=state,
                    action=action,
                    next_state=next_state,
                    confidence=float(conf[t].clamp(1e-6, 1.0)),
                    old_logprob=0.0,
                    done=(t == gen_length - 1),
                )
                steps_list.append(step)

            record = TrajectoryRecord(
                prompt_id=prompt_id,
                sample_id=i,
                prompt_text=self.tokenizer.decode(batch_prompt[i], skip_special_tokens=True),
                final_text=final_text,
                reward=0.0,   # to be filled by reward function
                steps=steps_list,
                metadata={"gen_length": gen_length, "steps": steps},
            )
            records.append(record)

        return records

    # ------------------------------------------------------------------
    # Internal generation with confidence tracking
    # ------------------------------------------------------------------

    def _generate_with_confidence(
        self,
        prompt: torch.Tensor,  # [batch, prompt_len]
        steps: int = 64,
        gen_length: int = 128,
        block_length: int = 64,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
    ):
        """
        Iterative denoising generation following LLaDA's algorithm.

        Returns
        -------
        x               : [batch, prompt_len + gen_length]  final token IDs
        token_confidence: [batch, prompt_len + gen_length]  per-token confidence
                          scores (0 for prompt, confidence-at-reveal for completion)
        """
        bs = prompt.shape[0]
        prompt_len = prompt.shape[1]
        total_len = prompt_len + gen_length
        dtype = self.model.dtype

        x = torch.full((bs, total_len), self.mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_len] = prompt.clone()

        # Track the confidence at the time each token was committed
        token_confidence = torch.zeros(bs, total_len, device=self.device)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        for block_idx in range(num_blocks):
            start_idx = prompt_len + block_idx * block_length
            end_idx = prompt_len + (block_idx + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == self.mask_id
            num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step_i in range(steps_per_block):
                mask_index = x == self.mask_id  # [bs, total_len]

                with torch.cuda.amp.autocast(enabled=True):
                    logits = self.model(x).logits  # [bs, total_len, vocab]
                    logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature, dtype=dtype)
                    x0 = torch.argmax(logits_with_noise, dim=-1)  # [bs, total_len]

                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(dtype), dim=-1)  # [bs, total_len, vocab]
                        # Confidence = softmax prob of the chosen token
                        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    else:
                        x0_p = torch.rand(x0.shape, device=self.device)

                    # Mask out positions beyond current block
                    x0_p[:, end_idx:] = -float("inf")

                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(bs):
                        n_transfer = num_transfer_tokens[j, step_i].item()
                        if n_transfer > 0:
                            _, sel = torch.topk(confidence[j], k=int(n_transfer))
                            transfer_index[j, sel] = True

                    # Record confidence at time of reveal
                    revealed_now = transfer_index & (x == self.mask_id)
                    token_confidence[revealed_now] = x0_p[revealed_now].float().clamp(0.0, 1.0)

                    x[transfer_index] = x0[transfer_index]

        return x, token_confidence

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_gumbel_noise(logits: torch.Tensor, temperature: float, dtype: torch.dtype) -> torch.Tensor:
        if temperature == 0.0:
            return logits
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        """Precompute tokens to reveal per denoising step (uniform schedule)."""
        mask_num = mask_index.sum(dim=1, keepdim=True)  # [bs, 1]
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer = base.expand(-1, steps).clone()
        indices = torch.arange(steps, device=mask_index.device)
        num_transfer[indices.unsqueeze(0) < remainder] += 1
        return num_transfer.to(torch.int64)
