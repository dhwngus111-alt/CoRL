#!/usr/bin/env bash

set -euo pipefail

TEST_ROOT="/home/isl_jhoh/CoRL/transplant/tests/mep_test"
CORL_ROOT="/home/isl_jhoh/CoRL"
PYTHON="${PYTHON:-/home/isl_jhoh/miniconda3/envs/corl/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${TEST_ROOT}/outputs/run_${RUN_STAMP}}"

source "${TEST_ROOT}/config.sh"

mkdir -p "${RUN_DIR}" "${TEST_ROOT}/policy_pool"

export POLICY_POOL="${TEST_ROOT}/policy_pool"
export PYTHONPATH="${CORL_ROOT}:${CORL_ROOT}/HSP:${CORL_ROOT}/risked_overcooked/src:${PYTHONPATH:-}"

cd "${CORL_ROOT}"

"${PYTHON}" "${TEST_ROOT}/eval_mep_vs_random.py" \
  --output-root "${RUN_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_ENTITY}" \
  --wandb_group_name "${WANDB_GROUP_NAME}" \
  --wandb-mode "${WANDB_MODE}" \
  "$@" 2>&1 | tee "${RUN_DIR}/eval.log"
