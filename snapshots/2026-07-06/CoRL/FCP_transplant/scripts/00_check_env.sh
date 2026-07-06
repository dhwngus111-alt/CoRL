#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [00] FCP: Risky 환경/설정 확인 시작 ===\n\n\n'
echo "layout=${LAYOUT} episode_length=${EPISODE_LENGTH} subgoal_disable_steps=${SUBGOAL_DISABLE_STEPS} p_slip=${P_SLIP} time_cost=${TIME_COST}"
echo "output_root=${TRANSPLANT_OUTPUT_ROOT}  policy_pool=${POLICY_POOL}"
echo "seed_max=${SEED_MAX}  fcp_s2_population_size=${FCP_S2_POPULATION_SIZE}"
echo "python=${PYTHON}"
echo "fcp_device=${FCP_DEVICE} selected_device=${FCP_SELECTED_DEVICE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "cuda_probe=${FCP_CUDA_PROBE_RESULT}"

# policy_config.pkl (mlp/rnn) 생성.  transplant.build_policy_configs를 재사용하되
# POLICY_POOL 이 FCP_transplant/policy_pool 을 가리키므로 FCP pool 아래에 쓰인다.
"${PYTHON}" -m transplant.build_policy_configs \
  --layout "${LAYOUT}" \
  --episode-length "${EPISODE_LENGTH}" \
  --mep-population-size "${SEED_MAX}"

# 어댑터/런너 import + 환경 reset/step/encoding shape 점검.
"${PYTHON}" -m transplant.smoke_check --layout "${LAYOUT}"

# FCP 오버레이 모듈 import 점검.
"${PYTHON}" - <<'PY'
import FCP_transplant.bootstrap  # noqa: F401
import FCP_transplant.train_fcp_s1_bundle  # noqa: F401
import FCP_transplant.extract_fcp_s1  # noqa: F401
import FCP_transplant.extract_fcp_s2  # noqa: F401
print("FCP_transplant modules import OK")
PY

printf '\n\n\n=== [00] FCP: 확인 종료 ===\n\n\n'
