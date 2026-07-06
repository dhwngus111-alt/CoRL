#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMEDI_TRANSPLANT_ROOT="${COMEDI_TRANSPLANT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export TRANSPLANT_OUTPUT_ROOT="${TRANSPLANT_OUTPUT_ROOT:-${COMEDI_TRANSPLANT_ROOT}}"
export TRANSPLANT_ROOT="${TRANSPLANT_ROOT:-${COMEDI_TRANSPLANT_ROOT}}"
export CORL_ROOT="${CORL_ROOT:-/scratch/isllab0213/isl_jhoh/CoRL}"
export HSP_ROOT="${HSP_ROOT:-${CORL_ROOT}/HSP}"
export RISKED_ROOT="${RISKED_ROOT:-${CORL_ROOT}/risked_overcooked}"
export RISKY_ROOT="${RISKED_ROOT}"
export POLICY_POOL="${POLICY_POOL:-${COMEDI_TRANSPLANT_ROOT}/policy_pool}"

export PYTHONPATH="${COMEDI_TRANSPLANT_ROOT}:${CORL_ROOT}:${HSP_ROOT}:${RISKED_ROOT}/src:${PYTHONPATH:-}"

DEFAULT_PYTHON="/home/isllab0213/miniconda3/envs/corl/bin/python"
if [[ -z "${PYTHON:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON="${DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

DEFAULT_LAYOUTS="risky_dualpath_subgoal risky_mixed_coordination_subgoal risky_multipath_subgoal"
if [[ -z "${LAYOUTS:-}" ]]; then
  if [[ -n "${LAYOUT:-}" ]]; then
    LAYOUTS="${LAYOUT}"
  else
    LAYOUTS="${DEFAULT_LAYOUTS}"
  fi
fi
LAYOUT="${LAYOUT:-${LAYOUTS%% *}}"

NUM_AGENTS="${NUM_AGENTS:-2}"
EPISODE_LENGTH="${EPISODE_LENGTH:-200}"
SUBGOAL_DISABLE_STEPS="${SUBGOAL_DISABLE_STEPS:-60}"
P_SLIP="${P_SLIP:-0.4}"
TIME_COST="${TIME_COST:--0.3}"
DISTANCE_SHAPING_REW="${DISTANCE_SHAPING_REW:-0.3}"
SUBGOAL_PRESS_REW="${SUBGOAL_PRESS_REW:-2}"
HANDOFF_SHAPING="${HANDOFF_SHAPING:-1}"
COMPETITIVE_ONION_SUPPLY="${COMPETITIVE_ONION_SUPPLY:-}"

CNN_LAYERS_PARAMS="${CNN_LAYERS_PARAMS:-32,3,1,1 64,3,1,1 32,3,1,1}"
# 이 프로젝트는 GPU 3번만 사용한다(고정). 다른 GPU를 쓰려면 CUDA_VISIBLE_DEVICES를 넘겨 override.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES

N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-50}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-1}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
PPO_EPOCH="${PPO_EPOCH:-10}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-0}"

COMEDI_POPULATION_SIZE="${COMEDI_POPULATION_SIZE:-8}"
COMEDI_ALPHA="${COMEDI_ALPHA:-0.5}"
COMEDI_BETA="${COMEDI_BETA:-1.0}"
COMEDI_SELECT_INTERVAL="${COMEDI_SELECT_INTERVAL:-1}"
COMEDI_EVAL_EPISODES="${COMEDI_EVAL_EPISODES:-1}"
COMEDI_LR="${COMEDI_LR:-0.01}"
COMEDI_CRITIC_LR="${COMEDI_CRITIC_LR:-0.01}"
COMEDI_ENTROPY_COEF="${COMEDI_ENTROPY_COEF:-0.0}"
# MP 전용 env 슬롯 수(원본 envs_mp = episode_length-1). 0이면 자동으로 episode_length-1.
# 스모크/자원 절약 시 낮춰서 switch 커버 밀도만 줄인다(학습 결과 영향 미미).
COMEDI_MP_THREADS="${COMEDI_MP_THREADS:-0}"

COMEDI_ADAPTIVE_AGENT_NAME="${COMEDI_ADAPTIVE_AGENT_NAME:-comedi_adaptive}"
COMEDI_ADAPTIVE_SP_WARMUP_STEPS="${COMEDI_ADAPTIVE_SP_WARMUP_STEPS:-200000}"
COMEDI_ADAPTIVE_SP_PPO_EPOCH="${COMEDI_ADAPTIVE_SP_PPO_EPOCH:-100}"
COMEDI_ADAPTIVE_SP_LR="${COMEDI_ADAPTIVE_SP_LR:-0.01}"
COMEDI_ADAPTIVE_SP_ENTROPY_COEF="${COMEDI_ADAPTIVE_SP_ENTROPY_COEF:-0.001}"
COMEDI_S2_TRAIN_ENV_BATCH="${COMEDI_S2_TRAIN_ENV_BATCH:-1}"

WANDB_ENTITY="${WANDB_ENTITY:-isllab}"
WANDB_PROJECT="${WANDB_PROJECT:-Comedi_risky_${LAYOUT}}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-CoMeDi_${LAYOUT}}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-CoMeDi_${LAYOUT}}"

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
  --hidden_size "${HIDDEN_SIZE:-64}"
  --layer_N "${LAYER_N:-2}"
  --activation_id "${ACTIVATION_ID:-1}"
  --user_name comedi
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

USE_WANDB="${USE_WANDB:-0}"
case "${USE_WANDB}" in
  0|false|False|FALSE|no|No|NO)
    COMMON_ARGS+=(--use_wandb)
    ;;
esac

USE_CUDA="${USE_CUDA:-1}"
case "${USE_CUDA}" in
  0|false|False|FALSE|no|No|NO)
    COMMON_ARGS+=(--cuda)
    ;;
esac

MLP_POLICY_ARGS=(--algorithm_name mappo --use_recurrent_policy)
RNN_POLICY_ARGS=(--algorithm_name rmappo)
