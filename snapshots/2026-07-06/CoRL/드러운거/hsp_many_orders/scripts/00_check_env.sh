#!/usr/bin/env bash
set -euo pipefail

source /home/isl_jhoh/CoRL/test/hsp_many_orders/scripts/common.sh

resolve_wandb_entity

echo "HSP_ROOT=${HSP_ROOT}"
echo "ORIG_HSP_SCRIPT_DIR=${ORIG_HSP_SCRIPT_DIR}"
echo "HSP_SCRIPT_DIR=${HSP_SCRIPT_DIR}"
echo "POLICY_POOL=${POLICY_POOL}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "BIASED_EVAL_ROOT=${BIASED_EVAL_ROOT}"
echo "FINAL_EVAL_ROOT=${FINAL_EVAL_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "GPU=${GPU}"
echo "DEFAULT_WANDB_ENTITY=${DEFAULT_WANDB_ENTITY}"
echo "WANDB_ENTITY=${WANDB_ENTITY}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT}"
echo "WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}"

test -d "${HSP_ROOT}"
test -d "${ORIG_HSP_SCRIPT_DIR}"
test -d "${HSP_SCRIPT_DIR}"
test -f "${HSP_SCRIPT_DIR}/train/train_overcooked_mep.py"
test -f "${HSP_SCRIPT_DIR}/train/train_overcooked_hsp.py"
test -f "${HSP_SCRIPT_DIR}/train/train_overcooked_adaptive.py"
test -f "${HSP_SCRIPT_DIR}/eval/eval_overcooked.py"
test -f "${HSP_SCRIPT_DIR}/hsp/greedy_select.py"
test -d "${POLICY_POOL}/many_orders"
test -f "${POLICY_POOL}/many_orders/mep/s1/train.yml"
test -f "${POLICY_POOL}/many_orders/hsp/s1/eval.yml"
test -f "${POLICY_POOL}/many_orders/hsp/s2/train.yml"
test -f "${POLICY_POOL}/many_orders/policy_config/mlp_policy_config.pkl"
test -f "${POLICY_POOL}/many_orders/policy_config/rnn_policy_config.pkl"

"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["numpy", "torch", "wandb", "yaml", "imageio"]
for mod in mods:
    importlib.import_module(mod)
print("python dependencies ok")
PY

"${PYTHON_BIN}" - <<'PY'
from hsp.config import get_config
from hsp.envs.overcooked_new.Overcooked_Env import Overcooked
print("hsp imports ok")
PY

echo "Environment check passed."
