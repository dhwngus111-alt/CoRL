#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HSP Stage 2 고성능 기본값 설정 (대규모 비대칭 병렬화)
export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-300}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"
export HSP_S2_TRAIN_ENV_BATCH="${HSP_S2_TRAIN_ENV_BATCH:-1}"
export REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-100000000}"
export PPO_EPOCH="${PPO_EPOCH:-15}"
export NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export HSP_S2_POPULATION_SIZE="${HSP_S2_POPULATION_SIZE:-36}"

source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [07] HSP S2 학습 시작 ===\n\n\n'

POP_YAML="${POLICY_POOL}/${LAYOUT}/hsp/s2/train.yml"
ACTUAL_POPULATION_SIZE="$("${PYTHON}" -c "import sys, yaml; d=yaml.safe_load(open(sys.argv[1])) or {}; print(sum(1 for k in d if 'hsp_adaptive' not in k))" "${POP_YAML}")"
if [[ "${ACTUAL_POPULATION_SIZE}" != "${HSP_S2_POPULATION_SIZE}" ]]; then
  echo "invalid HSP S2 population size: expected ${HSP_S2_POPULATION_SIZE}, got ${ACTUAL_POPULATION_SIZE} in ${POP_YAML}" >&2
  exit 1
fi
if ! [[ "${HSP_S2_TRAIN_ENV_BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid HSP_S2_TRAIN_ENV_BATCH=${HSP_S2_TRAIN_ENV_BATCH}: it must be a positive integer" >&2
  exit 1
fi
if (( N_ROLLOUT_THREADS % HSP_S2_TRAIN_ENV_BATCH != 0 )); then
  echo "invalid HSP_S2_TRAIN_ENV_BATCH=${HSP_S2_TRAIN_ENV_BATCH}: it must divide N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS}" >&2
  exit 1
fi

"${PYTHON}" -m transplant.train_risky_adaptive \
  "${COMMON_ARGS[@]}" \
  --algorithm_name adaptive \
  --experiment_name hsp-S2 \
  --wandb_stage_name 07_hsp_s2 \
  --wandb_run_name "07_hsp_s2_${WANDB_RUN_PREFIX}_seed${SEED:-1}" \
  --seed "${SEED:-1}" \
  --stage 2 \
  --population_yaml_path "${POP_YAML}" \
  --population_size "${HSP_S2_POPULATION_SIZE}" \
  --adaptive_agent_name hsp_adaptive \
  --train_env_batch "${HSP_S2_TRAIN_ENV_BATCH}" \
  --use_agent_policy_id \
  --entropy_coef "${ENTROPY_COEF:-0.01}"

printf '\n\n\n=== [07] HSP S2 학습 종료 ===\n\n\n'
