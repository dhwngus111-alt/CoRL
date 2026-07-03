#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USER_NUM_ENV_STEPS="${NUM_ENV_STEPS:-}"
export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-300}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"
export PPO_EPOCH="${PPO_EPOCH:-15}"
export NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-100000000}"
if [[ -z "${COMEDI_ADAPTIVE_SP_WARMUP_STEPS:-}" ]]; then
  if [[ -n "${USER_NUM_ENV_STEPS}" && "${USER_NUM_ENV_STEPS}" =~ ^[0-9]+$ && "${USER_NUM_ENV_STEPS}" -lt 200000 ]]; then
    export COMEDI_ADAPTIVE_SP_WARMUP_STEPS="${USER_NUM_ENV_STEPS}"
  else
    export COMEDI_ADAPTIVE_SP_WARMUP_STEPS=200000
  fi
fi

source "${SCRIPT_DIR}/common.sh"

printf '\n\n\n=== [02] CoMeDi adaptive start: %s ===\n\n\n' "${LAYOUT}"

EXTRA_ARGS=()
case "${COMEDI_SKIP_WARMUP:-0}" in
  1|true|True|TRUE|yes|Yes|YES)
    EXTRA_ARGS+=(--comedi_skip_warmup)
    ;;
esac

"${PYTHON}" -m comedi_transplant.train_comedi_adaptive \
  "${COMMON_ARGS[@]}" \
  --algorithm_name adaptive \
  --experiment_name comedi-S2 \
  --wandb_stage_name comedi_adaptive_s2 \
  --wandb_run_name "comedi_adaptive_s2_${WANDB_RUN_PREFIX}_seed${SEED:-1}" \
  --seed "${SEED:-1}" \
  --stage 2 \
  --population_yaml_path "${POLICY_POOL}/${LAYOUT}/comedi/s2/train.yml" \
  --population_size "${COMEDI_POPULATION_SIZE}" \
  --adaptive_agent_name "${COMEDI_ADAPTIVE_AGENT_NAME}" \
  --comedi_adaptive_agent_name "${COMEDI_ADAPTIVE_AGENT_NAME}" \
  --comedi_population_size "${COMEDI_POPULATION_SIZE}" \
  --train_env_batch "${COMEDI_S2_TRAIN_ENV_BATCH}" \
  --use_agent_policy_id \
  --comedi_adaptive_sp_warmup_steps "${COMEDI_ADAPTIVE_SP_WARMUP_STEPS}" \
  --comedi_adaptive_sp_ppo_epoch "${COMEDI_ADAPTIVE_SP_PPO_EPOCH}" \
  --comedi_adaptive_sp_lr "${COMEDI_ADAPTIVE_SP_LR}" \
  --comedi_adaptive_sp_entropy_coef "${COMEDI_ADAPTIVE_SP_ENTROPY_COEF}" \
  "${EXTRA_ARGS[@]}"

printf '\n\n\n=== [02] CoMeDi adaptive done: %s ===\n\n\n' "${LAYOUT}"
