#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-50}"
export CNN_LAYERS_PARAMS="${CNN_LAYERS_PARAMS:-25,3,1,1 25,3,1,1 25,3,1,1}"
export MEP_S2_LAYER_N="${MEP_S2_LAYER_N:-3}"
export MEP_S2_HIDDEN_SIZE="${MEP_S2_HIDDEN_SIZE:-64}"
export MEP_SELECTED_POLICY_COUNT="${MEP_SELECTED_POLICY_COUNT:-${MEP_S1_POPULATION_SIZE:-5}}"
export MEP_CHECKPOINTS_PER_POLICY="${MEP_CHECKPOINTS_PER_POLICY:-3}"
export MEP_S2_POPULATION_SIZE="${MEP_S2_POPULATION_SIZE:-$((MEP_SELECTED_POLICY_COUNT * MEP_CHECKPOINTS_PER_POLICY))}"
export MEP_S2_TOTAL_STEPS_PER_AGENT="${MEP_S2_TOTAL_STEPS_PER_AGENT:-11000000}"
export MEP_S2_PPO_RUN_TOT_TIMESTEPS="${MEP_S2_PPO_RUN_TOT_TIMESTEPS:-40000}"
if [[ -z "${NUM_ENV_STEPS:-}" ]]; then
  MEP_FORMULA_PYTHON="${PYTHON:-/home/isl_jhoh/miniconda3/envs/corl/bin/python}"
  export NUM_ENV_STEPS="$("${MEP_FORMULA_PYTHON}" - "${MEP_S2_TOTAL_STEPS_PER_AGENT}" "${MEP_S2_POPULATION_SIZE}" "${MEP_S2_PPO_RUN_TOT_TIMESTEPS}" <<'PY'
import math
import sys

total_steps_per_agent = float(sys.argv[1])
population_size = int(sys.argv[2])
ppo_run_tot_timesteps = int(sys.argv[3])
iter_per_selection = population_size
num_pbt_iter = int(total_steps_per_agent * math.sqrt(population_size) // (iter_per_selection * ppo_run_tot_timesteps))
print(num_pbt_iter * iter_per_selection * ppo_run_tot_timesteps)
PY
)"
else
  export NUM_ENV_STEPS
fi
export TRAIN_ENV_BATCH="${TRAIN_ENV_BATCH:-1}"
# Match HSP S2 adaptive: anneal dense reward shaping from 1 to 0 over the full training budget.
export MEP_S2_REWARD_SHAPING_HORIZON="${MEP_S2_REWARD_SHAPING_HORIZON:-${NUM_ENV_STEPS}}"
export REWARD_SHAPING_HORIZON="${MEP_S2_REWARD_SHAPING_HORIZON}"
export MEP_S2_INITIAL_REWARD_SHAPING_FACTOR="${MEP_S2_INITIAL_REWARD_SHAPING_FACTOR:-1.0}"
export MEP_S2_REWARD_SHAPING_FACTOR="${MEP_S2_REWARD_SHAPING_FACTOR:-${MEP_S2_INITIAL_REWARD_SHAPING_FACTOR}}"
export PPO_EPOCH="${PPO_EPOCH:-5}"
export NUM_MINI_BATCH="${NUM_MINI_BATCH:-10}"
export MEP_PPO_LR="${MEP_PPO_LR:-8e-4}"
export MEP_PPO_CRITIC_LR="${MEP_PPO_CRITIC_LR:-8e-4}"
export MEP_PPO_GAMMA="${MEP_PPO_GAMMA:-0.99}"
export MEP_PPO_GAE_LAMBDA="${MEP_PPO_GAE_LAMBDA:-0.98}"
export MEP_PPO_CLIP_PARAM="${MEP_PPO_CLIP_PARAM:-0.05}"
export MEP_PPO_MAX_GRAD_NORM="${MEP_PPO_MAX_GRAD_NORM:-0.1}"
export MEP_PPO_VALUE_LOSS_COEF="${MEP_PPO_VALUE_LOSS_COEF:-0.1}"
export MEP_PPO_POLICY_VALUE_LOSS_COEF="${MEP_PPO_POLICY_VALUE_LOSS_COEF:-0.1}"
export MEP_PPO_ENTROPY_COEF="${MEP_PPO_ENTROPY_COEF:-0.5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ADAPTIVE_CHECKPOINT_INTERVAL_STEPS="${ADAPTIVE_CHECKPOINT_INTERVAL_STEPS:-5000000}"
export ADAPTIVE_CHECKPOINT_DIR_NAME="${ADAPTIVE_CHECKPOINT_DIR_NAME:-milestone_checkpoints}"

source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [02] MEP S2 adaptive training start ===\n\n\n'
echo "MEP S2 reward shaping: initial=${MEP_S2_INITIAL_REWARD_SHAPING_FACTOR}, start=${MEP_S2_REWARD_SHAPING_FACTOR}, horizon=${REWARD_SHAPING_HORIZON}"

POP_YAML="${POLICY_POOL}/${LAYOUT}/mep/s2/train.yml"
ACTUAL_POPULATION_SIZE="$("${PYTHON}" -c "import sys, yaml; d=yaml.safe_load(open(sys.argv[1])) or {}; print(sum(1 for k in d if k != 'mep_adaptive'))" "${POP_YAML}")"
if [[ "${ACTUAL_POPULATION_SIZE}" != "${MEP_S2_POPULATION_SIZE}" ]]; then
  echo "invalid MEP S2 population size: expected ${MEP_S2_POPULATION_SIZE}, got ${ACTUAL_POPULATION_SIZE} in ${POP_YAML}" >&2
  exit 1
fi
if (( N_ROLLOUT_THREADS % TRAIN_ENV_BATCH != 0 )); then
  echo "invalid TRAIN_ENV_BATCH=${TRAIN_ENV_BATCH}: it must divide N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS}" >&2
  exit 1
fi

"${PYTHON}" -m transplant.train_risky_adaptive \
  "${COMMON_ARGS[@]}" \
  --algorithm_name mep \
  --experiment_name mep-S2 \
  --wandb_stage_name 02_mep_s2 \
  --wandb_run_name "02_mep_s2_${WANDB_RUN_PREFIX}_seed${SEED:-1}" \
  --seed "${SEED:-1}" \
  --stage 2 \
  --initial_reward_shaping_factor "${MEP_S2_INITIAL_REWARD_SHAPING_FACTOR}" \
  --reward_shaping_factor "${MEP_S2_REWARD_SHAPING_FACTOR}" \
  --population_yaml_path "${POP_YAML}" \
  --population_size "${MEP_S2_POPULATION_SIZE}" \
  --adaptive_agent_name mep_adaptive \
  --adaptive_checkpoint_interval_steps "${ADAPTIVE_CHECKPOINT_INTERVAL_STEPS}" \
  --adaptive_checkpoint_dir_name "${ADAPTIVE_CHECKPOINT_DIR_NAME}" \
  --train_env_batch "${TRAIN_ENV_BATCH}" \
  --use_agent_policy_id \
  --mep_use_prioritized_sampling \
  --mep_prioritized_alpha "${MEP_PRIORITIZED_ALPHA}" \
  --layer_N "${MEP_S2_LAYER_N}" \
  --hidden_size "${MEP_S2_HIDDEN_SIZE}" \
  --lr "${MEP_PPO_LR}" \
  --critic_lr "${MEP_PPO_CRITIC_LR}" \
  --gamma "${MEP_PPO_GAMMA}" \
  --gae_lambda "${MEP_PPO_GAE_LAMBDA}" \
  --clip_param "${MEP_PPO_CLIP_PARAM}" \
  --max_grad_norm "${MEP_PPO_MAX_GRAD_NORM}" \
  --value_loss_coef "${MEP_PPO_VALUE_LOSS_COEF}" \
  --policy_value_loss_coef "${MEP_PPO_POLICY_VALUE_LOSS_COEF}" \
  --entropy_coef "${MEP_PPO_ENTROPY_COEF}"

printf '\n\n\n=== [02] MEP S2 adaptive training end ===\n\n\n'
