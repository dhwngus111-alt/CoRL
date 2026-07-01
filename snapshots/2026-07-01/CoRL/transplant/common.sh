#!/usr/bin/env bash

set -euo pipefail

# 이 파일은 transplant/scripts/*.sh에서 공통으로 source되는 설정 파일이다.
# 여기서는 경로, Python 실행기, risked 환경 인자, PPO/HSP/MEP 기본값을 한곳에서 관리한다.

# 프로젝트 루트 경로들.
# TRANSPLANT_ROOT: 이번 transplant 실험 코드와 결과가 놓이는 루트.
# CORL_ROOT: CoRL 작업 디렉터리 루트.
# HSP_ROOT/RISKED_ROOT: 원본 HSP 코드와 Risked Overcooked 코드 위치.
# POLICY_POOL: 학습/추출된 policy yaml과 .pt 파일이 모이는 위치.
export TRANSPLANT_ROOT="${TRANSPLANT_ROOT:-/home/isl_jhoh/CoRL/transplant}"
export CORL_ROOT="${CORL_ROOT:-/home/isl_jhoh/CoRL}"
export HSP_ROOT="${HSP_ROOT:-${CORL_ROOT}/HSP}"
export RISKED_ROOT="${RISKED_ROOT:-${CORL_ROOT}/risked_overcooked}"
export RISKY_ROOT="${RISKED_ROOT}"
export POLICY_POOL="${POLICY_POOL:-${TRANSPLANT_ROOT}/policy_pool}"

# transplant, HSP, risked_overcooked/src를 import할 수 있도록 Python path를 묶는다.
export PYTHONPATH="${CORL_ROOT}:${HSP_ROOT}:${RISKED_ROOT}/src:${PYTHONPATH:-}"

# 기본 Python은 corl conda env의 Python을 우선 사용한다.
# 외부에서 PYTHON=/path/to/python을 export하면 그 값이 우선된다.
DEFAULT_PYTHON="/home/isl_jhoh/miniconda3/envs/corl/bin/python"
if [[ -z "${PYTHON:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON="${DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

# Risked Overcooked 환경 기본값.
# LAYOUT은 subgoal layout 기본값이고 Stage 1/2와 평가가 같은 layout을 보게 한다.
# EPISODE_LENGTH=200은 새 Risked transplant 실험 기본 horizon이다.
# SUBGOAL_DISABLE_STEPS는 G subgoal interact 후 puddle이 비활성화되는 step 수이다.
# P_SLIP/TIME_COST는 Risked 환경의 확률적 slip과 step penalty 계열 설정이다.
LAYOUT="${LAYOUT:-risky_dualpath_subgoal}"
NUM_AGENTS="${NUM_AGENTS:-2}"
EPISODE_LENGTH="${EPISODE_LENGTH:-200}"
SUBGOAL_DISABLE_STEPS="${SUBGOAL_DISABLE_STEPS:-60}"
P_SLIP="${P_SLIP:-0.4}"
TIME_COST="${TIME_COST:--0.3}"
DISTANCE_SHAPING_REW="${DISTANCE_SHAPING_REW:-0.3}"
SUBGOAL_PRESS_REW="${SUBGOAL_PRESS_REW:-2}"
HANDOFF_SHAPING="${HANDOFF_SHAPING:-0}"
COMPETITIVE_ONION_SUPPLY="${COMPETITIVE_ONION_SUPPLY:-}"

# CNN 관측 encoder 구조와 사용할 GPU.
# CUDA_VISIBLE_DEVICES는 각 실행 tmux에서 export해서 덮어쓸 수 있다.
CNN_LAYERS_PARAMS="${CNN_LAYERS_PARAMS:-32,3,1,1 64,3,1,1 32,3,1,1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# 학습/평가 공통 스케일.
# NUM_ENV_STEPS는 각 training script에서 별도 export하면 그 값이 들어간다.
# 기본값은 full Stage 1 budget이다. 짧은 검증은 실행할 때 NUM_ENV_STEPS를 명시적으로 줄인다.
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-4}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-1}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
SEED_MAX="${SEED_MAX:-36}"

# checkpoint/log 주기와 PPO 기본값.
# REWARD_SHAPING_HORIZON=100000000은 HSP Stage 1 설정과 맞춘 값이다.
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
TRAIN_ENV_BATCH="${TRAIN_ENV_BATCH:-1}"
PPO_EPOCH="${PPO_EPOCH:-15}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-100000000}"

# Policy pool 관련 기본값.
# MEP_POPULATION_SIZE=6이면 extract 단계에서 init/mid/final을 뽑아 18개 MEP policy가 된다.
# SEED_MAX=36과 HSP_SELECT_K=18은 HSP 후보 생성/선택 script들이 공유해서 쓴다.
MEP_POPULATION_SIZE="${MEP_POPULATION_SIZE:-12}"
HSP_S2_POPULATION_SIZE="${HSP_S2_POPULATION_SIZE:-}"
HSP_S2_TRAIN_ENV_BATCH="${HSP_S2_TRAIN_ENV_BATCH:-1}"
HSP_SELECT_K="${HSP_SELECT_K:-18}"

# W&B 설정. layout별로 W&B project를 분리한다.
WANDB_ENTITY="${WANDB_ENTITY:-dhwngus41-daegu-gyeongbuk-institute-of-science-technology}"
WANDB_PROJECT="${WANDB_PROJECT:-RiskyOvercooked_${LAYOUT}}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-HSP_${LAYOUT}}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-HSP_${LAYOUT}}"

# 대부분의 train/eval script가 transplant.train_risky_adaptive.py 등에 넘기는 공통 CLI 인자.
# 개별 script는 여기에 algorithm_name, experiment_name, stage, population_yaml_path 등을 추가한다.
COMMON_ARGS=(
  --env_name RiskyOvercooked
  --layout_name "${LAYOUT}"
  --num_agents "${NUM_AGENTS}"
  --episode_length "${EPISODE_LENGTH}"
  --subgoal_disable_steps "${SUBGOAL_DISABLE_STEPS}"
  --p_slip "${P_SLIP}"
  --time_cost "${TIME_COST}"
  --distance_shaping_rew "${DISTANCE_SHAPING_REW}"
  --subgoal_press_rew "${SUBGOAL_PRESS_REW}"
  --reward_shaping_horizon "${REWARD_SHAPING_HORIZON}"
  --n_training_threads 1
  --n_rollout_threads "${N_ROLLOUT_THREADS}"
  --n_eval_rollout_threads "${N_EVAL_ROLLOUT_THREADS}"
  --num_env_steps "${NUM_ENV_STEPS}"
  --eval_episodes "${EVAL_EPISODES}"
  --ppo_epoch "${PPO_EPOCH}"
  --num_mini_batch "${NUM_MINI_BATCH}"
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --cnn_layers_params "${CNN_LAYERS_PARAMS}"
  --user_name risky_hsp
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_ENTITY}"
  --wandb_group_name "${WANDB_GROUP_NAME}"
  --wandb_run_prefix "${WANDB_RUN_PREFIX}"
)

case "${HANDOFF_SHAPING}" in
  1|true|True|TRUE|yes|Yes|YES)
    COMMON_ARGS+=(--handoff_shaping)
    ;;
esac
if [[ -n "${COMPETITIVE_ONION_SUPPLY}" ]]; then
  COMMON_ARGS+=(--competitive_onion_supply "${COMPETITIVE_ONION_SUPPLY}")
fi

# W&B 사용 여부.
# 주의: HSP 계열 parser에서 --use_wandb 플래그는 action='store_false' 의미로 쓰인다.
# 그래서 USE_WANDB=1이면 아무 플래그도 추가하지 않고, USE_WANDB=0일 때만 --use_wandb를 추가한다.
USE_WANDB="${USE_WANDB:-1}"
case "${USE_WANDB}" in
  0|false|False|FALSE|no|No|NO)
    COMMON_ARGS+=(--use_wandb)
    ;;
esac

# policy architecture shortcut.
# MLP_POLICY_ARGS는 feed-forward MAPPO, RNN_POLICY_ARGS는 recurrent MAPPO 계열 script에서 사용한다.
MLP_POLICY_ARGS=(--algorithm_name mappo --use_recurrent_policy)
RNN_POLICY_ARGS=(--algorithm_name rmappo)
