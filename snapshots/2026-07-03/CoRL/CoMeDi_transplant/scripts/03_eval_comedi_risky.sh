#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TIME_COST="${TIME_COST:-0}"
source "${SCRIPT_DIR}/common.sh"

printf '\n\n\n=== [03] CoMeDi final eval start: %s ===\n\n\n' "${LAYOUT}"

FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-${COMEDI_TRANSPLANT_ROOT}/final_eval/${LAYOUT}}"
RENDER_EVAL_GIF_EPISODES="${RENDER_EVAL_GIF_EPISODES:-1}"
PAIR_COMMON_ARGS=("${COMMON_ARGS[@]}")
case "${EVAL_PAIR_USE_WANDB:-0}" in
  1|true|True|TRUE|yes|Yes|YES)
    ;;
  *)
    PAIR_COMMON_ARGS+=(--use_wandb)
    ;;
esac

EXTRA_ARGS=()
if [[ -n "${PARTNER_POLICY:-}" ]]; then
  EXTRA_ARGS+=(--partner_policy "${PARTNER_POLICY}")
fi
if [[ -n "${PARTNER_GROUP:-}" ]]; then
  EXTRA_ARGS+=(--partner_group "${PARTNER_GROUP}")
fi

"${PYTHON}" -m comedi_transplant.eval_comedi_risky \
  "${PAIR_COMMON_ARGS[@]}" \
  --algorithm_name population \
  --experiment_name comedi-final-eval \
  --wandb_stage_name comedi_final_eval \
  --seed "${SEED:-1}" \
  --population_yaml_path "${POLICY_POOL}/${LAYOUT}/comedi/s2/eval.yml" \
  --comedi_adaptive_agent_name "${COMEDI_ADAPTIVE_AGENT_NAME}" \
  --adaptive_policy_name "${COMEDI_ADAPTIVE_AGENT_NAME}" \
  --final_eval_dir "${FINAL_EVAL_DIR}" \
  --render_eval_gif_episodes "${RENDER_EVAL_GIF_EPISODES}" \
  --eval_stochastic \
  "${EXTRA_ARGS[@]}"

printf '\n\n\n=== [03] CoMeDi final eval done: %s ===\n\n\n' "${LAYOUT}"
