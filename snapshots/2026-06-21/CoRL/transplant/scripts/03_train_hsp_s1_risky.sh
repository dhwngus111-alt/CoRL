#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HSP Stage 1 고성능 기본값 설정 (HSP 원본처럼 rollout threads=100)
export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-100}"
export REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-0}"
export PPO_EPOCH="${PPO_EPOCH:-15}"
export HSP_FINAL_GIF_EPISODES="${HSP_FINAL_GIF_EPISODES:-3}"
# HSP 원본 S1 설정과 동일하게 과도한 checkpoint/W&B I/O를 피한다.
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"

source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [03] HSP S1 학습 시작 ===\n\n\n'

"${PYTHON}" -m transplant.build_policy_configs --layout "${LAYOUT}" --episode-length "${EPISODE_LENGTH}" --mep-population-size "${MEP_POPULATION_SIZE}"

if [[ -z "${HSP_W0:-}" ]]; then
  HSP_W0="$("${PYTHON}" -m transplant.hsp_hidden_utility --kind w0)"
fi
if [[ -z "${HSP_W1:-}" ]]; then
  HSP_W1="$("${PYTHON}" -m transplant.hsp_hidden_utility --kind w1)"
fi

"${PYTHON}" -m transplant.train_risky_hsp_s1_bundle \
  "${COMMON_ARGS[@]}" \
  "${MLP_POLICY_ARGS[@]}" \
  --experiment_name hsp-S1 \
  --wandb_stage_name 03_hsp_s1 \
  --wandb_bundle_run_name "03_hsp_s1_${WANDB_RUN_PREFIX}_seeds1-${SEED_MAX}" \
  --seed_max "${SEED_MAX}" \
  --use_hsp \
  --random_index \
  --share_policy \
  --hsp_final_gif_episodes "${HSP_FINAL_GIF_EPISODES}" \
  --w0 "${HSP_W0}" \
  --w1 "${HSP_W1}"

printf '\n\n\n=== [03] HSP S1 학습 종료 ===\n\n\n'
