#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-50}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-1000000}"
export PPO_EPOCH="${PPO_EPOCH:-10}"
export NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-0}"

source "${SCRIPT_DIR}/common.sh"

printf '\n\n\n=== [01] CoMeDi convention population start: %s ===\n\n\n' "${LAYOUT}"

"${PYTHON}" -m comedi_transplant.train_comedi_population \
  "${COMMON_ARGS[@]}" \
  "${MLP_POLICY_ARGS[@]}" \
  --experiment_name comedi-S1 \
  --wandb_stage_name comedi_population \
  --wandb_run_name "comedi_population_${WANDB_RUN_PREFIX}_seed${SEED:-1}" \
  --seed "${SEED:-1}" \
  --lr "${COMEDI_LR}" \
  --critic_lr "${COMEDI_CRITIC_LR}" \
  --entropy_coef "${COMEDI_ENTROPY_COEF}" \
  --use_linear_lr_decay \
  --comedi_population_size "${COMEDI_POPULATION_SIZE}" \
  --comedi_alpha "${COMEDI_ALPHA}" \
  --comedi_beta "${COMEDI_BETA}" \
  --comedi_select_interval "${COMEDI_SELECT_INTERVAL}" \
  --comedi_eval_episodes "${COMEDI_EVAL_EPISODES}"

printf '\n\n\n=== [01] CoMeDi convention population done: %s ===\n\n\n' "${LAYOUT}"
