#!/usr/bin/env bash
set -euo pipefail

source /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/common.sh
resolve_wandb_entity
cd_hsp_scripts

GPU="${GPU_MEP_S1:-${GPU:-0}}"
LOG_FILE="${LOG_DIR}/01_train_mep_s1_many_orders.log"

echo "Starting MEP S1 on many_orders. Log: ${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train/train_overcooked_mep.py \
  --env_name Overcooked \
  --algorithm_name mep \
  --experiment_name mep-S1 \
  --layout_name many_orders \
  --num_agents 2 \
  --seed 1 \
  --n_training_threads 1 \
  --n_rollout_threads 100 \
  --num_mini_batch 1 \
  --episode_length 400 \
  --num_env_steps 10000000 \
  --reward_shaping_horizon 100000000 \
  --ppo_epoch 15 \
  --save_interval 50 \
  --log_interval 1 \
  --train_env_batch 100 \
  --stage 1 \
  --mep_entropy_alpha 0.01 \
  --population_yaml_path "${POLICY_POOL}/many_orders/mep/s1/train.yml" \
  --population_size 12 \
  --adaptive_agent_name mep_adaptive \
  --entropy_coef 0.01 \
  --overcooked_version new \
  --wandb_name "${WANDB_ENTITY}" \
  --user_name "${USER_NAME}" \
  --wandb_tags hsp_many_orders paper mep_s1 2>&1 | tee "${LOG_FILE}"
