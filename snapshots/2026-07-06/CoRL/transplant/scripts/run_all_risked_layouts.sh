#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

LAYOUTS=(
  risky_dualpath_subgoal
  risky_mixed_coordination_subgoal
  risky_multipath_subgoal
)

wandb_enabled() {
  case "${USE_WANDB:-1}" in
    0|false|False|FALSE|no|No|NO)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

for layout in "${LAYOUTS[@]}"; do
  export LAYOUT="${layout}"
  export WANDB_GROUP_NAME="HSP_${layout}"
  export WANDB_RUN_PREFIX="HSP_${layout}"
  export FINAL_EVAL_DIR="${TRANSPLANT_ROOT}/final_eval/${layout}"

  printf '\n\n\n=== Risked HSP pipeline 시작: %s ===\n\n\n' "${layout}"

  bash "${SCRIPT_DIR}/00_check_env.sh"
  bash "${SCRIPT_DIR}/01_train_mep_s1_risky.sh"
  "${PYTHON}" "${SCRIPT_DIR}/02_extract_mep_s1_risky.py" \
    --layout "${layout}" \
    --max-policies "${MEP_POPULATION_SIZE}" \
    --require-p-slip "${P_SLIP}" \
    --require-episode-length "${EPISODE_LENGTH}" \
    --require-time-cost "${TIME_COST}"
  bash "${SCRIPT_DIR}/03_train_hsp_s1_risky.sh"
  "${PYTHON}" "${SCRIPT_DIR}/04_extract_hsp_s1_risky.py" \
    --layout "${layout}" \
    --episode-length "${EPISODE_LENGTH}" \
    --subgoal-disable-steps "${SUBGOAL_DISABLE_STEPS}" \
    --require-seeds "${SEED_MAX}"
  bash "${SCRIPT_DIR}/05_eval_events_risky.sh"
  "${PYTHON}" "${SCRIPT_DIR}/06_greedy_select_risky.py" \
    --layout "${layout}" \
    --k "${HSP_SELECT_K}" \
    --seed "${SEED:-1}"
  bash "${SCRIPT_DIR}/07_train_hsp_s2_risky.sh"

  extract_wandb_args=()
  if wandb_enabled; then
    extract_wandb_args=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
      --wandb-run-name "08_extract_hsp_s2_${WANDB_RUN_PREFIX}"
      --wandb-group "${WANDB_GROUP_NAME}"
      --wandb-mode "${WANDB_MODE:-online}"
    )
  fi
  "${PYTHON}" "${SCRIPT_DIR}/08_extract_hsp_s2_risky.py" \
    --layout "${layout}" \
    "${extract_wandb_args[@]}"
  bash "${SCRIPT_DIR}/09_eval_hsp_risky.sh"

  printf '\n\n\n=== Risked HSP pipeline 종료: %s ===\n\n\n' "${layout}"
done
