# cc_rl Experiment Status (2026-06-22)

## 실험 구조

### 목적
LLaDA-8B-Instruct 모델에 세 가지 방법을 적용해 수학/코딩 벤치마크 성능을 비교한다.

| Method | 설명 |
|--------|------|
| `baseline` | 학습 없음, zero-shot eval |
| `diffu_grpo` | Diffusion GRPO (d1 패키지 `DiffuGRPOTrainer`) |
| `stage2` | Delta-V Credit Assignment (`ValueCreditTrainer`) |

### 데이터셋 × 메서드 = 9개 실험

| Dataset | 크기 | 예상 학습 시간 |
|---------|------|---------------|
| MBPP | 500 train / 90 eval | ~7h (stage2) |
| HumanEval | 164 train / 164 eval | ~3h (stage2) |
| SVAMP | 800 train / 200 eval | ~13h (stage2) |

---

## 실행 중인 코드

### 실행 명령 (2026-06-22 11:20 시작)
```bash
cd /home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl
nohup bash experiments/launch_fast.sh > experiments/outputs/fast_launch3.log 2>&1 &
# PID: 2314127
```

### Launcher: `experiments/launch_fast.sh`
- Phase 1 (Baselines): 이미 완료 → SKIP
- Phase 2 (Diffu-GRPO): 이미 완료 → SKIP
- Phase 3 (Stage 2): **현재 실행 중**
  - `mbpp/stage2` → `humaneval/stage2` → `svamp/stage2` 순차 실행
  - 각 실험: `python run_experiment.py --dataset {DS} --method stage2 --max_value_states 2`

### 핵심 파일
```
experiments/
  launch_fast.sh          # 9개 실험 순차 실행 launcher
  run_experiment.py       # 실험 진입점 (build_config, run_training)
  outputs/
    official_9exp/
      mbpp/stage2/        # 현재 실행 중
      humaneval/stage2/   # 대기 중
      svamp/stage2/       # 대기 중
    fast_launch3.log      # launcher 전체 로그

src/cc_rl/
  algorithms/
    stage1_cw_grpo.py     # CWGRPOTrainer: confidence-weighted GRPO
    stage2_value_credit.py # ValueCreditTrainer: delta-V credit
  critics/
    value_head.py          # MLP value head (hidden→256→1)

/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo/
  diffu_grpo_trainer.py   # 기반 DiffuGRPOTrainer (d1 패키지)
```

---

## 하이퍼파라미터

### 공통 (diffu_grpo / stage2 동일)
| 파라미터 | 값 |
|---------|-----|
| 모델 | LLaDA-8B-Instruct (bf16) |
| `learning_rate` | 1e-6 |
| `num_generations` | 4 |
| `generation_batch_size` | 4 |
| `per_device_train_batch_size` | 1 |
| `max_completion_length` | 256 |
| `max_prompt_length` | 256 |
| `diffusion_steps` | 64 |
| `block_length` | 32 |
| `beta` (KL penalty) | 0.04 |
| `epsilon` (PPO clip) | 0.2 |
| `temperature` | 0.9 |
| `remasking` | low_confidence |
| epochs | 1 |

### Stage 2 추가 파라미터
| 파라미터 | 값 |
|---------|-----|
| `value_hidden_size` | 256 |
| `value_mlp_layers` | 2 |
| `critic_lr` | 5e-6 |
| `critic_loss_coef` | 0.5 |
| `delta_v_gate` | 0.01 |
| `max_value_states` | 2 (launch_fast.sh에서 지정) |
| `credit_alpha` | 1.0 |
| `credit_eps` | 1e-6 |
| `credit_clip_min/max` | 0.25 / 4.0 |

---

## Stage 2 알고리즘 요약

```
ValueCreditTrainer (stage2_value_credit.py)
  └─ extends CWGRPOTrainer (stage1_cw_grpo.py)
       └─ extends DiffuGRPOTrainer (d1/diffu-grpo)

학습 루프 per step:
  1. generate_with_confidence()  → 4개 completion + 토큰별 confidence
     [no_grad] diffusion 64step으로 생성, 각 토큰 reveal 시 confidence 기록
  2. _compute_delta_v_advantages()  → block-level delta-V
     [no_grad] max_value_states=2 block boundary에서 V(s) 계산 (CPU staging)
     delta_v[b] = V(s_{b+1}) - V(s_b), terminal: r - V(s_{B-1})
  3. compute_loss() step 1: value head 업데이트
     backbone [no_grad], hidden.float() → value_head(fp32) → Huber loss
  4. compute_loss() step 2: policy 업데이트
     GRPO loss with delta_v weighted by confidence
```

### Value Head 구조
```
mean_pool(hidden_states[-1])  [batch, H=4096]
  → Linear(4096, 256) + GELU
  → Linear(256, 1)
  → squeeze → scalar V(s)
  (fp32, 별도 AdamW optimizer lr=5e-6)
```

---

## 현재 실험 결과 (2026-06-22 11:38)

```
                   baseline  diffu_grpo    stage2
  mbpp               0.1889      0.2333        -- (학습중, step 30/2000)
  humaneval          0.1768      0.1707        --
  svamp              0.8050      0.8300        --
```

GPU: 19.2 GB / 94.98 GB, 99% 사용 중

### 진행 상황
| 실험 | 상태 | 완료 시각 (예상) |
|------|------|-----------------|
| mbpp/stage2 | **RUNNING** step 30/2000 | ~18:50 |
| humaneval/stage2 | NOT STARTED | ~22:00 |
| svamp/stage2 | NOT STARTED | +13h 후 |

---

## 오늘 수정한 버그 (OOM 해결)

### 문제 1: OOM in generate_with_confidence
- **원인**: `generate_with_confidence()`에 `torch.no_grad()` 없음 → 64 diffusion step × 32 transformer layer 전체 gradient graph 저장 (~94 GB)
- **수정**: `stage1_cw_grpo.py`의 `generate_with_confidence`에 `@torch.no_grad()` 데코레이터 추가

### 문제 2: dtype 불일치
- **원인**: 모델 hidden states는 bf16, value_head는 fp32 → `F.linear` dtype 불일치
- **수정**: `stage2_value_credit.py`의 value_head 호출 시 `hidden.float()` 캐스팅 추가 (2곳: `_value_at_prefix`, `compute_loss`)

### 수정 파일
- `src/cc_rl/algorithms/stage1_cw_grpo.py` line 86: `@torch.no_grad()` 추가
- `src/cc_rl/algorithms/stage2_value_credit.py` line 132, 300: `hidden.float()` 추가
