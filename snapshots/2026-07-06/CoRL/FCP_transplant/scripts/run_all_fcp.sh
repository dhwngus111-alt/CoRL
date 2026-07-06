#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 한 layout에 대한 FCP 전체 파이프라인.  LAYOUT을 export해서 바꿀 수 있다.
#
# 짧은 검증 예:
#   USE_WANDB=0 NUM_ENV_STEPS=120000 N_ROLLOUT_THREADS=4 SEED_MAX=2 \
#     bash scripts/run_all_fcp.sh
#
# 엄격한 HSP-FCP 재현(정확한 half-final-reward 'mid' 선택):
#   USE_WANDB=0 REQUIRE_REWARD_HISTORY=1 bash scripts/run_all_fcp.sh
#   (USE_WANDB=0 이어야 seed별 로컬 tensorboard ep_sparse_r 이 기록된다.)

export LAYOUT="${LAYOUT:-risky_dualpath_subgoal}"

# common.sh를 한 번 source해서 PYTHON/경로/env를 sub-script들과 동일하게 맞춘다.
# (이렇게 안 하면 아래 인라인 extraction 호출이 system python으로 떨어질 수 있다.)
source "${SCRIPT_DIR}/../common.sh"
# 전체 파이프라인이 동일한 인터프리터를 쓰도록 sub-script에 상속시킨다.
export PYTHON

# 엄격 재현 시 extraction에서 half-reward mid를 강제한다 (없으면 hard error).
REQUIRE_REWARD_HISTORY="${REQUIRE_REWARD_HISTORY:-0}"
EXTRACT_STRICT_ARGS=()
case "${REQUIRE_REWARD_HISTORY}" in
  1|true|True|TRUE|yes|Yes|YES)
    EXTRACT_STRICT_ARGS+=(--require-reward-history)
    ;;
esac

echo "########## FCP pipeline: layout=${LAYOUT}  python=${PYTHON} ##########"
echo "seed_max=${SEED_MAX}  require_reward_history=${REQUIRE_REWARD_HISTORY}  use_wandb=${USE_WANDB:-1}"
echo "fcp_device=${FCP_DEVICE}  selected_device=${FCP_SELECTED_DEVICE}  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "cuda_probe=${FCP_CUDA_PROBE_RESULT}"

bash "${SCRIPT_DIR}/00_check_env.sh"
bash "${SCRIPT_DIR}/01_train_fcp_s1_risky.sh"
"${PYTHON}" "${SCRIPT_DIR}/02_extract_fcp_s1_risky.py" \
  --layout "${LAYOUT}" --require-seeds "${SEED_MAX}" "${EXTRACT_STRICT_ARGS[@]}"
bash "${SCRIPT_DIR}/03_train_fcp_s2_risky.sh"
"${PYTHON}" "${SCRIPT_DIR}/04_extract_fcp_s2_risky.py" --layout "${LAYOUT}"
bash "${SCRIPT_DIR}/05_eval_fcp_risky.sh"

echo "########## FCP pipeline done: layout=${LAYOUT} ##########"
