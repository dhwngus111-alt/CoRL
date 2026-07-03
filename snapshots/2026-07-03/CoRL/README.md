# CoRL Workspace Notes

이 문서는 `/home/isl_jhoh/CoRL` 작업공간에 받아둔 `HSP` 저장소를 빠르게 이해하기 위한 정리다.

## HSP 저장소 위치

- 로컬 경로: `/home/isl_jhoh/CoRL/HSP`
- 원본 저장소: `https://github.com/samjia2000/HSP`
- 기준 커밋: `79be728`
- 논문: `Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased`
- 핵심 환경: Overcooked
- 핵심 알고리즘 흐름: SP/FCP/MEP/HSP policy pool 생성 후 adaptive policy 학습

## 전체 구조

```text
HSP/
├── README.md
├── setup.py
└── hsp/
    ├── algorithms/
    ├── envs/
    ├── policy_pool/
    ├── runner/
    ├── scripts/
    ├── utils/
    ├── config.py
    └── __init__.py
```

## hsp 폴더별 역할

### `hsp/config.py`

전체 학습에 공통으로 쓰이는 argparse 설정 파일이다. 알고리즘 이름, PPO 옵션, recurrent policy, CNN/MLP 설정, rollout thread 수, logging, evaluation, rendering 관련 옵션이 여기 모인다.

### `hsp/scripts/`

실험을 실제로 실행하는 shell script와 Python entrypoint가 들어 있다.

- `train_overcooked_sp.sh`: self-play policy 학습
- `train_overcooked_mep_stage_1.sh`: MEP population stage-1 학습
- `train_overcooked_mep_stage_2.sh`: MEP adaptive stage-2 학습
- `train_overcooked_fcp_stage_2.sh`: FCP adaptive stage-2 학습
- `train_hsp_all_S1.sh`: HSP biased policy stage-1 전체 layout 실행
- `train_hsp_all_S2.sh`: HSP adaptive stage-2 전체 layout 실행
- `eval_overcooked.sh`: policy 평가
- `eval_events.sh`: HSP policy pair의 행동 event feature 평가
- `extract_*_models.py`: wandb/result checkpoint를 `policy_pool` 형식으로 추출

하위 폴더:

- `scripts/train/`: SP, HSP, MEP, adaptive, traj 학습용 Python entrypoint
- `scripts/eval/`: 평가 entrypoint
- `scripts/render/`: policy 렌더링 entrypoint
- `scripts/hsp/s1/`: layout별 HSP stage-1 실행 스크립트
- `scripts/hsp/s2/`: layout별 HSP stage-2 실행 스크립트
- `scripts/hsp/greedy_select.py`: event count 기반으로 HSP policy subset을 greedy selection

### `hsp/runner/`

학습 루프의 중심이다. 환경 reset, rollout 수집, replay buffer insert, return 계산, trainer update, save/eval/logging을 담당한다.

- `runner/shared/base_runner.py`: shared-policy 공통 runner
- `runner/shared/overcooked_runner.py`: Overcooked shared-policy 학습/eval 루프
- `runner/separated/base_runner.py`: agent별 policy 분리 runner
- `runner/separated/overcooked_runner.py`: Overcooked separated-policy 학습/eval 루프

대부분의 실험 흐름을 따라가려면 `runner/shared/overcooked_runner.py`를 먼저 보면 된다.

### `hsp/algorithms/`

학습 알고리즘과 neural network 구성 요소가 들어 있다.

- `algorithms/r_mappo/`: MAPPO/RMAPPO 구현
- `algorithms/r_mappo/r_mappo.py`: PPO update, loss 계산, optimization
- `algorithms/r_mappo/algorithm/rMAPPOPolicy.py`: actor/critic policy wrapper
- `algorithms/r_mappo/algorithm/r_actor_critic.py`: actor/critic network 정의
- `algorithms/population/`: policy pool 기반 학습 구현
- `algorithms/population/policy_pool.py`: YAML 기반 policy pool load/register
- `algorithms/population/mep.py`: MEP 및 adaptive population trainer
- `algorithms/population/trainer_pool.py`: 여러 trainer를 묶는 pool trainer
- `algorithms/population/traj.py`: trajectory 기반 실험 코드
- `algorithms/utils/`: MLP, CNN, RNN, attention, distribution, PopArt 등 모델 building block

### `hsp/envs/`

Overcooked 환경과 vectorized environment wrapper가 들어 있다.

- `envs/env_wrappers.py`: subprocess/dummy vector env wrapper
- `envs/wrappers/env_policy.py`: 일부 agent를 fixed policy로 굴리는 wrapper
- `envs/overcooked/`: old Overcooked 환경
- `envs/overcooked_new/`: new Overcooked 환경

HSP 실험에서는 layout에 따라 old/new 환경을 나눠 쓴다.

- old env: `unident_s`, `random1`, `random3`
- new env: `distant_tomato`, `many_orders`

중요 파일:

- `envs/overcooked/Overcooked_Env.py`
- `envs/overcooked_new/Overcooked_Env.py`
- `envs/overcooked*/script_agent/`: scripted partner policies
- `envs/overcooked*/overcooked_ai_py` 또는 `src/overcooked_ai_py`: Overcooked MDP, planner, agent, visualization
- `envs/overcooked*/data/layouts/*.layout`: map/layout 정의

### `hsp/policy_pool/`

layout별 policy pool 설정과 policy config가 들어 있다. 실제 checkpoint `.pt`가 모두 들어 있는 것은 아니고, 주로 YAML config와 pickle config를 포함한다.

구조 예:

```text
policy_pool/
├── unident_s/
├── random1/
├── random3/
├── distant_tomato/
└── many_orders/
```

각 layout 아래에는 보통 다음이 있다.

- `fcp/s1/eval.yml`
- `fcp/s2/train.yml`, `fcp/s2/eval.yml`
- `mep/s1/train.yml`, `mep/s1/eval.yml`
- `mep/s2/train.yml`, `mep/s2/eval.yml`
- `hsp/s1/eval.yml`
- `hsp/s2/train.yml`, `hsp/s2/eval.yml`
- `policy_config/mlp_policy_config.pkl`
- `policy_config/rnn_policy_config.pkl`

### `hsp/utils/`

학습 보조 유틸이다.

- `shared_buffer.py`: shared-policy PPO replay buffer
- `separated_buffer.py`: separated-policy PPO replay buffer
- `valuenorm.py`: value normalization
- `multi_discrete.py`: MultiDiscrete action space 보조
- `util.py`: learning-rate schedule, initialization, shape helper 등

## HSP 학습 흐름

HSP README 기준 흐름은 다음과 같다.

1. Self-play 또는 MEP/HSP stage-1 policy들을 여러 seed/layout으로 학습한다.
2. `extract_*_models.py`로 checkpoint를 `policy_pool` 구조에 맞게 추출한다.
3. HSP의 경우 `eval_events_all.sh`로 biased policies 사이의 event feature를 평가한다.
4. `scripts/hsp/greedy_select.py`로 다양한 행동 특성을 가진 policy subset을 선택한다.
5. 선택된 policy pool을 상대로 adaptive policy를 stage-2에서 학습한다.
6. `eval_overcooked.sh`로 adaptive policy를 평가한다.

## 먼저 볼 파일

HSP 아이디어와 구현 연결을 빠르게 보려면 아래 순서가 좋다.

1. `hsp/scripts/train/train_overcooked_hsp.py`
2. `hsp/scripts/train/train_overcooked_adaptive.py`
3. `hsp/envs/wrappers/env_policy.py`
4. `hsp/algorithms/population/policy_pool.py`
5. `hsp/algorithms/population/mep.py`
6. `hsp/runner/shared/overcooked_runner.py`
7. `hsp/scripts/hsp/greedy_select.py`

## 주의할 점

- 이 저장소는 연구 재현용 코드 성격이 강하다.
- README 설치 예시에 `conda create -n hsp` 후 `conda activate marl`로 환경명이 불일치한다.
- 여러 shell script에 `--log_inerval` 오타가 있다. 실제 option은 `--log_interval`이다.
- `runner/*/base_runner.py`에 Slack webhook URL이 하드코딩되어 있다.
- PyTorch `1.5.1+cu101` 기준이라 최신 환경에서 바로 실행이 어려울 수 있다.
- open issues 기준으로 human proxy 평가 코드, human study 코드, traj script, MEP `population_size` 관련 재현성 이슈가 남아 있다.

## 한 줄 요약

`HSP`는 Overcooked에서 사람의 숨은 편향 utility를 가정해 다양한 biased policy pool을 만들고, 그 pool을 상대로 zero-shot cooperation용 adaptive policy를 학습하는 연구 코드다.
