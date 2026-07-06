#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_WANDB_PROJECT="${WANDB_PROJECT:-}"
source "${SCRIPT_DIR}/../common.sh"

LAYOUTS=(
  risky_dualpath_subgoal
  risky_mixed_coordination_subgoal
  risky_multipath_subgoal
)

for layout in "${LAYOUTS[@]}"; do
  export LAYOUT="${layout}"
  export WANDB_PROJECT="${USER_WANDB_PROJECT:-MEP_${layout}}"
  export WANDB_GROUP_NAME="MEP_${layout}"
  export WANDB_RUN_PREFIX="MEP_${layout}"
  export FINAL_EVAL_DIR="${MEP_TRANSPLANT_ROOT}/final_eval/${layout}"

  printf '\n\n\n=== Risky MEP baseline pipeline start: %s ===\n\n\n' "${layout}"

  "${PYTHON}" "${SCRIPT_DIR}/01_prepare_mep_s1_pool.py" --layout "${layout}"
  bash "${SCRIPT_DIR}/02_train_mep_s2_risky.sh"
  "${PYTHON}" "${SCRIPT_DIR}/03_extract_mep_s2_risky.py" \
    --layout "${layout}"
  bash "${SCRIPT_DIR}/04_eval_mep_risky.sh"

  printf '\n\n\n=== Risky MEP baseline pipeline end: %s ===\n\n\n' "${layout}"
done
