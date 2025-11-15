# Auto DAgger Training Pipeline

자동으로 모든 milestone에 대해 DAgger policy를 학습하는 파이프라인입니다.

## 작동 방식

```
각 milestone에 대해:
1. Expert policy 평가 (5-10 episodes)
   → 성공률 & 평균 step 수 측정

2. max_episode_steps 계산
   → mean_success_steps × multiplier (기본값: 2.0)

3. DAgger training 실행
   → 계산된 max_episode_steps로 학습
```

## 기본 사용법

```bash
# 기본 실행
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py

# 파라미터 조정
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --eval-episodes 10 \
    --step-multiplier 2.5 \
    --dagger-iterations 15 \
    --episodes-per-iter 20
```

## 주요 파라미터

### Expert Evaluation
- `--eval-episodes`: Expert를 평가할 에피소드 수 (기본값: 5)
- `--step-multiplier`: Expert의 평균 step에 곱할 배수 (기본값: 2.0)
- `--min-success-rate`: 학습을 진행할 최소 expert 성공률 (기본값: 0.6)

### DAgger Training
- `--dagger-iterations`: DAgger 반복 횟수 (기본값: 10)
- `--episodes-per-iter`: 반복당 rollout 에피소드 수 (기본값: 10)
- `--train-steps-per-iter`: 반복당 학습 step 수 (기본값: 2000)
- `--batch-size`: 배치 크기 (기본값: 64)
- `--learning-rate`: 학습률 (기본값: 3e-4)
- `--hidden-dim`: 은닉층 차원 (기본값: 256)

### 경로 설정
- `--milestone-config`: milestone 설정 JSON 파일 (기본값: milestone_config.json)
- `--state-dir`: milestone state 파일 디렉토리 (기본값: .milestone_trainer_cache/milestone_states)
- `--policy-dir`: expert policy 파일 디렉토리 (기본값: .milestone_trainer_cache/successful_policies)
- `--output-dir`: 결과 저장 디렉토리 (기본값: dagger_pipeline_results)

### 재시작 & 스킵
- `--milestone MILESTONE_ID`: 특정 milestone 하나만 실행 (resume/skip 무시)
- `--resume-from MILESTONE_ID`: 특정 milestone부터 재시작
- `--skip-milestones [ID1, ID2, ...]`: 특정 milestone들을 건너뛰기

### 기타
- `--track`: Weights & Biases 추적 활성화
- `--wandb-project`: W&B 프로젝트 이름 (기본값: pokemon-dagger-pipeline)
- `--no-record-video`: 비디오 녹화 비활성화
- `--no-cuda`: CUDA 비활성화

## 예시

### 1. 빠른 테스트 (적은 에피소드)
```bash
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --eval-episodes 3 \
    --dagger-iterations 5 \
    --episodes-per-iter 5 \
    --train-steps-per-iter 1000
```

### 2. 전체 학습 (높은 품질)
```bash
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --eval-episodes 10 \
    --step-multiplier 2.0 \
    --dagger-iterations 20 \
    --episodes-per-iter 20 \
    --train-steps-per-iter 3000
```

### 3. 특정 milestone 하나만 실행 (빠른 테스트)
```bash
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --milestone RIVAL_HOUSE \
    --eval-episodes 5 \
    --dagger-iterations 10
```

### 4. 특정 milestone부터 재시작
```bash
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --resume-from RIVAL_BEDROOM
```

### 5. 특정 milestone들 건너뛰기
```bash
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --skip-milestones GAME_RUNNING PLAYER_NAME_SET
```

### 6. W&B 추적 활성화
```bash
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --track \
    --wandb-project my-pokemon-dagger \
    --wandb-entity my-username
```

### 7. 비디오 녹화 없이 실행 (빠름)
```bash
/home/heatz123/anaconda3/envs/pokeagent/bin/python3 rl_training/auto_dagger_pipeline.py \
    --no-record-video
```

## 출력 결과

파이프라인 실행 후 `dagger_pipeline_results/` 디렉토리에 다음 파일들이 생성됩니다:

### 1. `pipeline_results.csv`
```csv
milestone_id,index,timestamp,status,expert_success_rate,expert_mean_steps,max_episode_steps,dagger_final_success_rate
RIVAL_HOUSE,10,2024-11-14 12:34:56,completed,1.0,45.2,90,0.85
...
```

### 2. `pipeline_results.json`
```json
[
  {
    "milestone_id": "RIVAL_HOUSE",
    "index": 10,
    "timestamp": "2024-11-14 12:34:56",
    "status": "completed",
    "expert_success_rate": 1.0,
    "expert_mean_steps": 45.2,
    "expert_std_steps": 8.3,
    "max_episode_steps": 90,
    "dagger_final_success_rate": 0.85
  },
  ...
]
```

### 3. 디렉토리 구조
```
dagger_pipeline_results/
├── pipeline_results.csv
├── pipeline_results.json
├── expert_eval_videos/
│   ├── RIVAL_HOUSE/
│   │   ├── expert_eval_ep1_success_steps45.mp4
│   │   └── ...
│   └── ...
├── dagger_videos/
│   ├── RIVAL_HOUSE/
│   │   ├── rollout_iter0_ep0_RIVAL_HOUSE.mp4
│   │   └── ...
│   └── ...
└── models/
    ├── RIVAL_HOUSE/
    │   ├── dagger_RIVAL_HOUSE_best.pth
    │   └── dagger_RIVAL_HOUSE_final.pth
    └── ...
```

## 파이프라인 상태

각 milestone은 다음 중 하나의 상태를 가집니다:

- ✅ **completed**: 성공적으로 완료
- ❌ **failed**: 실패 (expert 성공률 낮음, 파일 없음 등)
- ⏭️ **skipped**: 건너뜀 (skip list에 있거나 resume point 이전)

## 문제 해결

### Expert evaluation이 실패하는 경우
1. Expert policy 파일이 존재하는지 확인: `.milestone_trainer_cache/successful_policies/{MILESTONE_ID}.py`
2. Starting state 파일이 존재하는지 확인: `.milestone_trainer_cache/milestone_states/{PREV_MILESTONE}_completed.state`
3. `--eval-episodes`를 늘려서 재시도

### Expert 성공률이 너무 낮은 경우
- `--min-success-rate`를 낮춰서 학습 진행 (예: `--min-success-rate 0.4`)
- Expert policy를 수정하여 더 안정적으로 만들기

### 학습이 너무 느린 경우
- `--no-record-video`로 비디오 녹화 비활성화
- `--episodes-per-iter`를 줄이기
- `--train-steps-per-iter`를 줄이기

### 메모리 부족
- `--batch-size`를 줄이기 (예: 32)
- `--hidden-dim`을 줄이기 (예: 128)

## 고급 사용법

### Step multiplier 동적 조정
Expert가 매우 안정적이면 (성공률 > 95%) multiplier를 낮춰도 됩니다:
```bash
--step-multiplier 1.5
```

Expert가 불안정하면 (성공률 60-80%) multiplier를 높이세요:
```bash
--step-multiplier 3.0
```

### 품질 vs 속도 트레이드오프

**고품질 학습 (느림)**:
```bash
--eval-episodes 10 \
--dagger-iterations 20 \
--episodes-per-iter 20 \
--train-steps-per-iter 5000
```

**빠른 학습 (낮은 품질)**:
```bash
--eval-episodes 3 \
--dagger-iterations 5 \
--episodes-per-iter 5 \
--train-steps-per-iter 1000 \
--no-record-video
```

## 참고

- 전체 파이프라인 실행은 milestone 수에 따라 수 시간에서 수 일이 걸릴 수 있습니다
- 중간에 중단되면 `--resume-from`으로 이어서 실행할 수 있습니다
- 각 milestone별 결과는 `pipeline_results.csv`에서 확인 가능합니다
