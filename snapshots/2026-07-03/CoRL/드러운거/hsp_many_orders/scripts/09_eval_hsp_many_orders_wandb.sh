#!/usr/bin/env bash
set -euo pipefail

source /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/common.sh
resolve_wandb_entity

WANDB_PROJECT="${WANDB_PROJECT:-Overcooked}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-hsp_many_orders_final_eval_seed1}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

WANDB_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

"${PYTHON_BIN}" /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/eval_hsp_many_orders_wandb.py \
  "${WANDB_ARGS[@]}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-mode "${WANDB_MODE}" \
  --run-root "${FINAL_EVAL_ROOT}" \
  --policy-pool "${POLICY_POOL}" \
  --population-yaml "${POLICY_POOL}/many_orders/hsp/s2/eval.yml" \
  --gpu "${GPU}" \
  --seed 1 \
  --eval-episodes 100 \
  --eval-threads 100 \
  --gifs-per-partner 3
