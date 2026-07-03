#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/common.sh"

for layout in ${LAYOUTS}; do
  export LAYOUT="${layout}"
  export WANDB_PROJECT="${COMEDI_WANDB_PROJECT:-RiskyOvercooked_CoMeDi_${layout}}"
  export WANDB_GROUP_NAME="CoMeDi_${layout}"
  export WANDB_RUN_PREFIX="CoMeDi_${layout}"
  export FINAL_EVAL_DIR="${COMEDI_TRANSPLANT_ROOT}/final_eval/${layout}"

  printf '\n\n\n=== CoMeDi pipeline start: %s ===\n\n\n' "${layout}"

  bash "${SCRIPT_DIR}/01_train_comedi_population.sh"
  bash "${SCRIPT_DIR}/02_train_comedi_adaptive.sh"
  bash "${SCRIPT_DIR}/03_eval_comedi_risky.sh"

  printf '\n\n\n=== CoMeDi pipeline done: %s ===\n\n\n' "${layout}"
done
