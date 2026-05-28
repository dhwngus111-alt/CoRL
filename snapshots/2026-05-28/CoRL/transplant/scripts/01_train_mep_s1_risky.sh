#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# MEP Stage 1 기본 학습값.
# 외부에서 같은 이름의 환경변수를 export하면 그 값이 우선 적용된다.
export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-100}"
export REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-100000000}"
export PPO_EPOCH="${PPO_EPOCH:-15}"
export MEP_FINAL_GIF_EPISODES="${MEP_FINAL_GIF_EPISODES:-10}"

# 실행 경로, Python 경로, 결과 저장 경로, risky 환경 공통 인자를 불러온다.
# 실제 파일: transplant/common.sh
source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [01] MEP S1 학습 시작 ===\n\n\n'

# MEP Stage 1에서 학습할 population yaml을 만든다.
# 실행 모듈: transplant/build_policy_configs.py
# 출력 예시: transplant/policy_pool/risky_multipath/mep/s1/train.yml
"${PYTHON}" -m transplant.build_policy_configs \
  --layout "${LAYOUT}" \
  --episode-length "${EPISODE_LENGTH}" \
  --mep-population-size "${MEP_POPULATION_SIZE}"

# 방금 생성한 train.yml에 요청한 수만큼의 MEP policy가 들어갔는지 확인한다.
# 여기의 inline Python은 학습을 돌리는 코드가 아니라 yaml 검증용이다.
POP_YAML="${POLICY_POOL}/${LAYOUT}/mep/s1/train.yml"
ACTUAL_POPULATION_SIZE="$("${PYTHON}" -c "import sys, yaml; d=yaml.safe_load(open(sys.argv[1])) or {}; print(sum(1 for k, v in d.items() if k != 'mep_adaptive' and v.get('train', False)))" "${POP_YAML}")"
if [[ "${ACTUAL_POPULATION_SIZE}" != "${MEP_POPULATION_SIZE}" ]]; then
  echo "invalid MEP S1 population size: expected ${MEP_POPULATION_SIZE}, got ${ACTUAL_POPULATION_SIZE} in ${POP_YAML}" >&2
  exit 1
fi

# 실제 MEP Stage 1 학습을 시작한다.
# 실행 모듈: transplant/train_risky_adaptive.py
# 주요 출력 위치: transplant/results/RiskyOvercooked/${LAYOUT}/mep/mep-S1
# 이후 02_extract_mep_s1_risky.py가 이 결과에서 init/mid/final policy를 뽑아 policy_pool에 저장한다.
"${PYTHON}" -m transplant.train_risky_adaptive \
  "${COMMON_ARGS[@]}" \
  --algorithm_name mep \
  --experiment_name mep-S1 \
  --wandb_stage_name 01_mep-S1 \
  --seed "${SEED:-1}" \
  --stage 1 \
  --mep_final_gif_episodes "${MEP_FINAL_GIF_EPISODES}" \
  --mep_entropy_alpha "${MEP_ENTROPY_ALPHA:-0.01}" \
  --population_yaml_path "${POP_YAML}" \
  --population_size "${MEP_POPULATION_SIZE}" \
  --adaptive_agent_name mep_adaptive \
  --train_env_batch "${N_ROLLOUT_THREADS}" \
  --entropy_coef "${ENTROPY_COEF:-0.01}"

printf '\n\n\n=== [01] MEP S1 학습 종료 ===\n\n\n'
