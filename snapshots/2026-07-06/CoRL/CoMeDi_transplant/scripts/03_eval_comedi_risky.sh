#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

export TIME_COST="${TIME_COST:-0}"
source "${SCRIPT_DIR}/common.sh"

printf '\n\n\n=== [03] CoMeDi final eval start: %s ===\n\n\n' "${LAYOUT}"

FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-${COMEDI_TRANSPLANT_ROOT}/final_eval/${LAYOUT}}"
mkdir -p "${FINAL_EVAL_DIR}"
RENDER_EVAL_GIF_EPISODES="${RENDER_EVAL_GIF_EPISODES:-1}"
GIF_PANEL_SECTION="${GIF_PANEL_SECTION:-comedi_final_eval_gifs}"

# eval_comedi_risky가 쓰는 CSV 경로(_eval_paths와 동일). 이후 집계 업로드가 이걸 읽는다.
EVENT_TABLE_CSV="${FINAL_EVAL_DIR}/comedi_adaptive_event_counts_long.csv"
SCORE_TABLE_CSV="${FINAL_EVAL_DIR}/comedi_adaptive_sparse_scores.csv"
GIF_MANIFEST_CSV="${FINAL_EVAL_DIR}/comedi_adaptive_pair_gifs.csv"

# HSP 09와 동일: per-pair 개별 wandb run은 만들지 않고, GIF 스트리밍 + 집계를 하나의 final-eval run에.
FINAL_EVAL_WANDB_RUN_NAME="${FINAL_EVAL_WANDB_RUN_NAME:-${SCRIPT_NAME}_seed${SEED:-1}}"
FINAL_EVAL_WANDB_RUN_ID="${FINAL_EVAL_WANDB_RUN_ID:-${SCRIPT_NAME}_seed${SEED:-1}}"
STREAM_EVAL_GIFS_TO_WANDB="${STREAM_EVAL_GIFS_TO_WANDB:-${USE_WANDB:-1}}"

# per-pair eval의 자체 wandb run은 끔(HSP 09와 동일). --use_wandb는 store_false → wandb OFF.
PAIR_COMMON_ARGS=("${COMMON_ARGS[@]}")
case "${EVAL_PAIR_USE_WANDB:-0}" in
  1|true|True|TRUE|yes|Yes|YES)
    ;;
  *)
    PAIR_COMMON_ARGS+=(--use_wandb)
    ;;
esac

# GIF를 하나의 final-eval run으로 스트리밍(HSP 09의 STREAM_WANDB_ARGS와 동일).
STREAM_WANDB_ARGS=()
case "${USE_WANDB:-1}:${STREAM_EVAL_GIFS_TO_WANDB}" in
  0:*|false:*|False:*|FALSE:*|no:*|No:*|NO:*|*:0|*:false|*:False|*:FALSE|*:no|*:No|*:NO)
    ;;
  *)
    STREAM_WANDB_ARGS=(
      --stream_eval_gifs_to_wandb
      --stream_gif_wandb_project "${WANDB_PROJECT}"
      --stream_gif_wandb_entity "${WANDB_ENTITY}"
      --stream_gif_wandb_mode "${WANDB_MODE:-online}"
      --stream_gif_wandb_run_name "${FINAL_EVAL_WANDB_RUN_NAME}"
      --stream_gif_wandb_run_id "${FINAL_EVAL_WANDB_RUN_ID}"
      --gif_panel_section "${GIF_PANEL_SECTION}"
    )
    echo "stream eval GIFs to W&B section: ${GIF_PANEL_SECTION} (run ${FINAL_EVAL_WANDB_RUN_ID})"
    ;;
esac

EXTRA_ARGS=()
if [[ -n "${PARTNER_POLICY:-}" ]]; then
  EXTRA_ARGS+=(--partner_policy "${PARTNER_POLICY}")
fi
if [[ -n "${PARTNER_GROUP:-}" ]]; then
  EXTRA_ARGS+=(--partner_group "${PARTNER_GROUP}")
fi

# pool 파트너(양방향) + uniform-random 파트너(양방향)를 한 번에 평가(내부 루프).
"${PYTHON}" -m comedi_transplant.eval_comedi_risky \
  "${PAIR_COMMON_ARGS[@]}" \
  "${STREAM_WANDB_ARGS[@]}" \
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

# 집계 이벤트/점수 테이블을 같은 final-eval run에 업로드(HSP 09와 동일).
UPLOAD_SCRIPT="${CORL_ROOT}/transplant/scripts/upload_hsp_eval_event_table_risky.py"
UPLOAD_WANDB_ARGS=()
case "${USE_WANDB:-1}" in
  0|false|False|FALSE|no|No|NO)
    ;;
  *)
    UPLOAD_WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
      --wandb-mode "${WANDB_MODE:-online}"
      --wandb-run-name "${FINAL_EVAL_WANDB_RUN_NAME}"
      --wandb-run-id "${FINAL_EVAL_WANDB_RUN_ID}"
      --wandb-group "${WANDB_GROUP_NAME}"
    )
    ;;
esac

if [[ -f "${EVENT_TABLE_CSV}" && -f "${UPLOAD_SCRIPT}" ]]; then
  "${PYTHON}" "${UPLOAD_SCRIPT}" \
    --input "${EVENT_TABLE_CSV}" \
    --score-input "${SCORE_TABLE_CSV}" \
    --output-dir "${FINAL_EVAL_DIR}" \
    --gif-manifest "${GIF_MANIFEST_CSV}" \
    --gif-panel-section "${GIF_PANEL_SECTION}" \
    --adaptive-policy-name "${COMEDI_ADAPTIVE_AGENT_NAME}" \
    --layout "${LAYOUT}" \
    "${UPLOAD_WANDB_ARGS[@]}"
fi

printf '\n\n\n=== [03] CoMeDi final eval done: %s ===\n\n\n' "${LAYOUT}"
