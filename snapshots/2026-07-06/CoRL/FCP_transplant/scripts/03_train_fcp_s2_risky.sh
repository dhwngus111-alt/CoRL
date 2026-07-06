#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── FCP Stage 2: frozen 파트너 풀에 대한 best-response(adaptive) ───────────────
# HSP FCP baseline(train_overcooked_fcp_stage_2.sh)의 값을 따른다:
#   algorithm=adaptive, stage=2, population_size 36, adaptive_agent_name fcp_adaptive,
#   n_rollout_threads 300, num_env_steps 1e8, save_interval 20, train_env_batch 1,
#   ppo_epoch 15, num_mini_batch 1, use_agent_policy_id.
#   샘플링: uniform (prioritized 플래그를 주지 않음 = HSP FCP와 동일).
export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-300}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"
export REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-100000000}"
export PPO_EPOCH="${PPO_EPOCH:-15}"
export NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export FCP_S2_POPULATION_SIZE="${FCP_S2_POPULATION_SIZE:-36}"
export FCP_S2_TRAIN_ENV_BATCH="${FCP_S2_TRAIN_ENV_BATCH:-1}"
export ADAPTIVE_CHECKPOINT_INTERVAL_STEPS="${ADAPTIVE_CHECKPOINT_INTERVAL_STEPS:-5000000}"
export ADAPTIVE_CHECKPOINT_DIR_NAME="${ADAPTIVE_CHECKPOINT_DIR_NAME:-milestone_checkpoints}"
# S2도 다른 방법(HSP-S2)과 동일 환경으로 비교되도록 handoff shaping을 켠다.
export HANDOFF_SHAPING="${FCP_S2_HANDOFF_SHAPING:-1}"

source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [03] FCP S2 (best-response) 학습 시작 ===\n\n\n'

POP_YAML="${POLICY_POOL}/${LAYOUT}/fcp/s2/train.yml"
if [[ ! -f "${POP_YAML}" ]]; then
  echo "missing ${POP_YAML}: run 02_extract_fcp_s1_risky.py first" >&2
  exit 1
fi
ACTUAL_POPULATION_SIZE="$("${PYTHON}" -c "import sys, yaml; d=yaml.safe_load(open(sys.argv[1])) or {}; print(sum(1 for k in d if 'fcp_adaptive' not in k))" "${POP_YAML}")"
if [[ "${ACTUAL_POPULATION_SIZE}" != "${FCP_S2_POPULATION_SIZE}" ]]; then
  echo "invalid FCP S2 population size: expected ${FCP_S2_POPULATION_SIZE}, got ${ACTUAL_POPULATION_SIZE} in ${POP_YAML}" >&2
  exit 1
fi
if ! [[ "${FCP_S2_TRAIN_ENV_BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid FCP_S2_TRAIN_ENV_BATCH=${FCP_S2_TRAIN_ENV_BATCH}: must be a positive integer" >&2
  exit 1
fi
if (( N_ROLLOUT_THREADS % FCP_S2_TRAIN_ENV_BATCH != 0 )); then
  echo "invalid FCP_S2_TRAIN_ENV_BATCH=${FCP_S2_TRAIN_ENV_BATCH}: must divide N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS}" >&2
  exit 1
fi

# NOTE: prioritized-sampling 플래그(--mep_use_prioritized_sampling,
# --use_advantage_prioritized_sampling)를 의도적으로 넣지 않는다 → uniform 샘플링(FCP).
"${PYTHON}" -m transplant.train_risky_adaptive \
  "${COMMON_ARGS[@]}" \
  --algorithm_name adaptive \
  --experiment_name fcp-S2 \
  --wandb_stage_name 03_fcp_s2 \
  --wandb_run_name "03_fcp_s2_${WANDB_RUN_PREFIX}_seed${SEED:-1}" \
  --seed "${SEED:-1}" \
  --stage 2 \
  --population_yaml_path "${POP_YAML}" \
  --population_size "${FCP_S2_POPULATION_SIZE}" \
  --adaptive_agent_name fcp_adaptive \
  --adaptive_checkpoint_interval_steps "${ADAPTIVE_CHECKPOINT_INTERVAL_STEPS}" \
  --adaptive_checkpoint_dir_name "${ADAPTIVE_CHECKPOINT_DIR_NAME}" \
  --train_env_batch "${FCP_S2_TRAIN_ENV_BATCH}" \
  --use_agent_policy_id \
  --entropy_coef "${ENTROPY_COEF:-0.01}"

printf '\n\n\n=== [03] FCP S2 학습 종료 ===\n\n\n'
