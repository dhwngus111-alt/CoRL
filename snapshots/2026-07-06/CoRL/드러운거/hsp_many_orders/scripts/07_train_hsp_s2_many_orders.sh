#!/usr/bin/env bash
# --use_policy_in_env 를 추가함.  -> env에서 하던 상대 policy action 계산을 trainer/policy pool 쪽에서 하도록 바꿈
# 추가함으로써 WORKER들이 CPU가 아니라 GPU에서 돌아가도록 함
set -euo pipefail

source /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/common.sh
resolve_wandb_entity
cd_hsp_scripts

GPU="${GPU_HSP_S2:-3}"
LOG_FILE="${LOG_DIR}/07_train_hsp_s2_many_orders.log"

echo "Starting HSP S2 adaptive training on many_orders. Log: ${LOG_FILE}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train/train_overcooked_adaptive.py \
  --env_name Overcooked \
  --algorithm_name adaptive \
  --experiment_name hsp-S2 \
  --layout_name many_orders \
  --num_agents 2 \
  --seed 1 \
  --n_training_threads 1 \
  --num_mini_batch 1 \
  --episode_length 400 \
  --num_env_steps 100000000 \
  --ppo_epoch 15 \
  --reward_shaping_horizon 100000000 \
  --n_rollout_threads 300 \
  --train_env_batch 1 \
  --stage 2 \
  --save_interval 20 \
  --log_interval 10 \
  --population_yaml_path "${POLICY_POOL}/many_orders/hsp/s2/train.yml" \
  --population_size 36 \
  --adaptive_agent_name hsp_adaptive \
  --use_agent_policy_id \
  --use_policy_in_env \
  --overcooked_version new \
  --wandb_name "${WANDB_ENTITY}" \
  --user_name "${USER_NAME}" \
  --wandb_tags hsp_many_orders paper hsp_s2 2>&1 | tee "${LOG_FILE}"
