#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── FCP Stage 1: self-play 파트너 풀 ─────────────────────────────────────────
# HSP FCP baseline(train_overcooked_sp.sh)의 값을 따른다:
#   seed 1..12, n_rollout_threads 100, num_env_steps 1e7, save_interval 25,
#   ppo_epoch 15, reward_shaping_horizon 1e8.
# (risked 고유 인자 p_slip/time_cost/subgoal 등은 common.sh의 공유 설정을 상속)
export N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-100}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
export REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-100000000}"
export PPO_EPOCH="${PPO_EPOCH:-15}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export SEED_MAX="${SEED_MAX:-12}"
# 다른 방법(MEP-S1)과 동일 환경으로 비교되도록 handoff shaping을 켠다.
export HANDOFF_SHAPING="${FCP_S1_HANDOFF_SHAPING:-1}"
export FCP_S1_FINAL_GIF_EPISODES="${FCP_S1_FINAL_GIF_EPISODES:-0}"

source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [01] FCP S1 (self-play) 학습 시작: seeds 1..%s ===\n\n\n' "${SEED_MAX}"

# policy_config.pkl 보장.
"${PYTHON}" -m transplant.build_policy_configs \
  --layout "${LAYOUT}" \
  --episode-length "${EPISODE_LENGTH}" \
  --mep-population-size "${SEED_MAX}"

"${PYTHON}" -m FCP_transplant.train_fcp_s1_bundle \
  "${COMMON_ARGS[@]}" \
  "${MLP_POLICY_ARGS[@]}" \
  --experiment_name fcp-S1 \
  --wandb_stage_name 01_fcp_s1 \
  --wandb_bundle_run_name "01_fcp_s1_${WANDB_RUN_PREFIX}_seeds1-${SEED_MAX}" \
  --seed_max "${SEED_MAX}" \
  --hsp_final_gif_episodes "${FCP_S1_FINAL_GIF_EPISODES}"

# NOTE: --share_policy 는 의도적으로 넘기지 않는다.  HSP config에서 이 플래그는
# store_false(기본 True)이므로, 넘기면 shared self-play가 꺼져 separated가 된다.
# HSP FCP SP(train_overcooked_sp.sh)는 이 플래그를 넘기지 않아 shared로 학습한다.

printf '\n\n\n=== [01] FCP S1 학습 종료 ===\n\n\n'
