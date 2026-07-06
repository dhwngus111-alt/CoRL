#!/usr/bin/env bash
set -euo pipefail

source /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/common.sh
cd_hsp_scripts

LOG_FILE="${LOG_DIR}/06_greedy_select_many_orders.log"
echo "Running HSP greedy policy selection. Log: ${LOG_FILE}"
"${PYTHON_BIN}" hsp/greedy_select.py \
  --layout many_orders \
  --k 18 \
  --eval_result_dir "${BIASED_EVAL_ROOT}" \
  --policy_pool_path "${POLICY_POOL}" 2>&1 | tee "${LOG_FILE}"
