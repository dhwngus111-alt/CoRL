#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# CoMeDi convention-aware agent (논문 Eq.2: self-play PPO + convention BC).
# Table 7 / adap_cbr.sh 기준: 200k env steps, ppo_epoch 100, 50 threads, lr 1e-2, entropy 1e-3, MLP.
export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-50}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-200000}"
export PPO_EPOCH="${PPO_EPOCH:-100}"
export NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"

source "${SCRIPT_DIR}/common.sh"

printf '\n\n\n=== [02] CoMeDi convention-aware agent start: %s ===\n\n\n' "${LAYOUT}"

"${PYTHON}" -m comedi_transplant.train_comedi_cbr \
  "${COMMON_ARGS[@]}" \
  --experiment_name comedi-cbr \
  --wandb_stage_name comedi_cbr \
  --wandb_run_name "${SCRIPT_NAME}_seed${SEED:-1}" \
  --seed "${SEED:-1}" \
  --lr "${COMEDI_ADAPTIVE_SP_LR:-0.01}" \
  --critic_lr "${COMEDI_ADAPTIVE_SP_LR:-0.01}" \
  --entropy_coef "${COMEDI_ADAPTIVE_SP_ENTROPY_COEF:-0.001}" \
  --comedi_population_size "${COMEDI_POPULATION_SIZE}" \
  --comedi_adaptive_agent_name "${COMEDI_ADAPTIVE_AGENT_NAME}" \
  --comedi_bc_weight "${COMEDI_BC_WEIGHT:-1.0}"

printf '\n\n\n=== [02] CoMeDi convention-aware agent done: %s ===\n\n\n' "${LAYOUT}"
