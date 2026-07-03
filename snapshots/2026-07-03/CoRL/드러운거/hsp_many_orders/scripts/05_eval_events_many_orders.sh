#!/usr/bin/env bash
set -euo pipefail

source /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/common.sh
resolve_wandb_entity
cd_hsp_scripts

GPU="${GPU_EVAL_EVENTS:-${GPU:-0}}"
SEED_START="${HSP_S1_SEED_START:-1}"
SEED_END="${HSP_S1_SEED_END:-36}"
OUT_DIR="${BIASED_EVAL_ROOT}/many_orders"
LOG_FILE="${LOG_DIR}/05_eval_events_many_orders.log"

mkdir -p "${OUT_DIR}"
echo "Evaluating HSP S1 event counts for seeds ${SEED_START}..${SEED_END}. Log: ${LOG_FILE}"

for i in $(seq "${SEED_START}" "${SEED_END}"); do
  for direction in w0_first w1_first; do
    if [[ "${direction}" == "w0_first" ]]; then
      agent0_policy_name="hsp${i}_w0"
      agent1_policy_name="hsp${i}_w1"
    else
      agent0_policy_name="hsp${i}_w1"
      agent1_policy_name="hsp${i}_w0"
    fi
    exp="eval-${agent0_policy_name}-${agent1_policy_name}"
    out_file="${OUT_DIR}/${exp}.txt"
    echo "==== ${exp} ====" | tee -a "${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" eval/eval_overcooked.py \
      --env_name Overcooked \
      --algorithm_name population \
      --experiment_name "${exp}" \
      --layout_name many_orders \
      --user_name "${USER_NAME}" \
      --num_agents 2 \
      --seed 1 \
      --episode_length 400 \
      --n_eval_rollout_threads 100 \
      --eval_episodes 100 \
      --eval_stochastic \
      --wandb_name "${WANDB_ENTITY}" \
      --use_wandb \
      --population_yaml_path "${POLICY_POOL}/many_orders/hsp/s1/eval.yml" \
      --agent0_policy_name "${agent0_policy_name}" \
      --agent1_policy_name "${agent1_policy_name}" \
      --overcooked_version new > "${out_file}" 2>&1
  done
done
