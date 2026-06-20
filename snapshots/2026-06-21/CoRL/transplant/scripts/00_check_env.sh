#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [00] Risky 환경 확인 시작 ===\n\n\n'
echo "layout=${LAYOUT} episode_length=${EPISODE_LENGTH} subgoal_disable_steps=${SUBGOAL_DISABLE_STEPS} p_slip=${P_SLIP} time_cost=${TIME_COST}"

"${PYTHON}" -m transplant.build_policy_configs --layout "${LAYOUT}" --episode-length "${EPISODE_LENGTH}" --mep-population-size "${MEP_POPULATION_SIZE}"
"${PYTHON}" -m transplant.smoke_check --layout "${LAYOUT}"

printf '\n\n\n=== [00] Risky 환경 확인 종료 ===\n\n\n'
