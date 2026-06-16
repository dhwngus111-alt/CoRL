#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_EPISODES="${EVAL_EPISODES:-100}"
export N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-100}"
export EVENT_EVAL_GIF_EPISODES="${EVENT_EVAL_GIF_EPISODES:-3}"

# 평가 및 테스트 단계에서는 순수한 sparse reward와 behavior event 분석을 위해 time_cost를 0.0으로 강제 고정한다.
export TIME_COST=0.0
source "${SCRIPT_DIR}/../common.sh"

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

replace_common_arg() {
  local key="$1"
  local value="$2"
  local index

  for index in "${!COMMON_ARGS[@]}"; do
    if [[ "${COMMON_ARGS[$index]}" == "${key}" ]]; then
      COMMON_ARGS[$((index + 1))]="${value}"
      return
    fi
  done
  COMMON_ARGS+=("${key}" "${value}")
}

if ! is_positive_int "${EVAL_EPISODES}"; then
  echo "error: EVAL_EPISODES must be a positive integer, got '${EVAL_EPISODES}'." >&2
  exit 2
fi
if ! is_positive_int "${N_EVAL_ROLLOUT_THREADS}"; then
  echo "error: N_EVAL_ROLLOUT_THREADS must be a positive integer, got '${N_EVAL_ROLLOUT_THREADS}'." >&2
  exit 2
fi
if [[ ! "${EVENT_EVAL_GIF_EPISODES}" =~ ^[0-9]+$ ]]; then
  echo "error: EVENT_EVAL_GIF_EPISODES must be a non-negative integer, got '${EVENT_EVAL_GIF_EPISODES}'." >&2
  exit 2
fi

if (( EVAL_EPISODES < N_EVAL_ROLLOUT_THREADS )); then
  echo "warning: EVAL_EPISODES (${EVAL_EPISODES}) is less than N_EVAL_ROLLOUT_THREADS (${N_EVAL_ROLLOUT_THREADS})."
  echo "warning: adjusting N_EVAL_ROLLOUT_THREADS to ${EVAL_EPISODES} to prevent empty event evaluation logs."
  N_EVAL_ROLLOUT_THREADS="${EVAL_EPISODES}"
  export N_EVAL_ROLLOUT_THREADS
fi

if (( EVAL_EPISODES % N_EVAL_ROLLOUT_THREADS != 0 )); then
  echo "warning: EVAL_EPISODES (${EVAL_EPISODES}) is not divisible by N_EVAL_ROLLOUT_THREADS (${N_EVAL_ROLLOUT_THREADS})."
  echo "warning: the runner may execute fewer episodes than requested because it uses integer division."
fi

replace_common_arg --eval_episodes "${EVAL_EPISODES}"
replace_common_arg --n_eval_rollout_threads "${N_EVAL_ROLLOUT_THREADS}"

printf '\n\n\n=== [05] HSP S1 이벤트 평가 시작 ===\n\n\n'
echo "event eval settings: EVAL_EPISODES=${EVAL_EPISODES}, N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS}"
echo "event gif settings: EVENT_EVAL_GIF_EPISODES=${EVENT_EVAL_GIF_EPISODES} (W&B only; 0 disables)"

mkdir -p "${TRANSPLANT_ROOT}/biased_eval/${LAYOUT}"
POP_YAML="${POLICY_POOL}/${LAYOUT}/hsp/s1/eval.yml"

"${PYTHON}" -m transplant.eval_risky_hsp_events \
  "${COMMON_ARGS[@]}" \
  --algorithm_name population \
  --experiment_name hsp-S1-event-eval \
  --wandb_stage_name 05_hsp-S1-event-eval \
  --seed_max "${SEED_MAX}" \
  --event_gif_episodes "${EVENT_EVAL_GIF_EPISODES}" \
  --population_yaml_path "${POP_YAML}" \
  --metrics_output_dir "${TRANSPLANT_ROOT}/biased_eval/${LAYOUT}" \
  --eval_stochastic

printf '\n\n\n=== [05] HSP S1 이벤트 평가 종료 ===\n\n\n'
