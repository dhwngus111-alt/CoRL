#!/usr/bin/env bash
set -euo pipefail

source /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/common.sh
resolve_wandb_entity
cd_hsp_scripts

GPU="${GPU_HSP_S1:-${GPU:-0}}"
SEED_START="${HSP_S1_SEED_START:-1}"
SEED_END="${HSP_S1_SEED_END:-36}"
LOG_FILE="${LOG_DIR}/03_train_hsp_s1_many_orders.log"

W0="0,0,0,0,0,r[-5:5:3],0,r[-5:5:3],0,r[0:5:2],0,0,r[-5:5:3],0,r[-10:10:3],r[-10:0:2],r[0:10:2],0,r[-3:3:3],r[-3:3:3],r[-10:0:2],r[0:1:2]"
W1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"

echo "Starting HSP S1 seeds ${SEED_START}..${SEED_END} on many_orders. Log: ${LOG_FILE}"
for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  echo "==== HSP S1 seed ${seed} ====" | tee -a "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train/train_overcooked_hsp.py \
    --env_name Overcooked \
    --algorithm_name mappo \
    --experiment_name hsp-S1 \
    --layout_name many_orders \
    --num_agents 2 \
    --seed "${seed}" \
    --n_training_threads 1 \
    --n_rollout_threads 100 \
    --num_mini_batch 1 \
    --episode_length 400 \
    --num_env_steps 10000000 \
    --ppo_epoch 15 \
    --reward_shaping_horizon 0 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
    --save_interval 25 \
    --log_interval 10 \
    --use_recurrent_policy \
    --overcooked_version new \
    --use_hsp \
    --w0 "${W0}" \
    --w1 "${W1}" \
    --random_index \
    --share_policy \
    --wandb_name "${WANDB_ENTITY}" \
    --user_name "${USER_NAME}" \
    --wandb_tags hsp_many_orders paper hsp_s1 "hsp_seed_${seed}" 2>&1 | tee -a "${LOG_FILE}"
done
