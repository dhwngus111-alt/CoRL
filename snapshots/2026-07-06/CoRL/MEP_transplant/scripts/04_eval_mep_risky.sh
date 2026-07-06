#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TIME_COST=0
source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [04] MEP final evaluation start ===\n\n\n'
echo "final eval settings: EVAL_EPISODES=${EVAL_EPISODES}, N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS}, TIME_COST=${TIME_COST}"

FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-${MEP_TRANSPLANT_ROOT}/final_eval/${LAYOUT}}"
mkdir -p "${FINAL_EVAL_DIR}"
POP_YAML="${POLICY_POOL}/${LAYOUT}/mep/s2/eval.yml"
EVENT_TABLE_CSV="${EVENT_TABLE_CSV:-${FINAL_EVAL_DIR}/mep_adaptive_event_counts_long.csv}"
SCORE_TABLE_CSV="${SCORE_TABLE_CSV:-${FINAL_EVAL_DIR}/mep_adaptive_sparse_scores.csv}"
GIF_MANIFEST_CSV="${GIF_MANIFEST_CSV:-${FINAL_EVAL_DIR}/mep_adaptive_pair_gifs.csv}"
RENDER_EVAL_GIF_EPISODES="${RENDER_EVAL_GIF_EPISODES:-1}"
GIF_PANEL_SECTION="${GIF_PANEL_SECTION:-mep_final_eval_gifs}"
MAX_GIF_UPLOAD="${MAX_GIF_UPLOAD:-5}"
RANDOM_GIF_UPLOAD="${RANDOM_GIF_UPLOAD:-1}"
: > "${EVENT_TABLE_CSV}"
: > "${SCORE_TABLE_CSV}"
: > "${GIF_MANIFEST_CSV}"

if [[ -n "${PARTNER_POLICY:-}" ]]; then
  PARTNER_POLICIES=("${PARTNER_POLICY}")
  PARTNER_SET_NAME="${PARTNER_POLICY}"
else
  PARTNER_SET_NAME="mep_pool"
  mapfile -t PARTNER_POLICIES < <("${PYTHON}" - "${POP_YAML}" <<'PY'
import sys
import yaml

data = yaml.safe_load(open(sys.argv[1])) or {}
partners = [name for name in data if name != "mep_adaptive"]
if not partners:
    raise SystemExit(f"error: no partner policies found in {sys.argv[1]}")
print("\n".join(partners))
PY
)
fi

echo "final eval partners (${#PARTNER_POLICIES[@]}): ${PARTNER_POLICIES[*]}"
echo "final eval GIFs per direction: ${RENDER_EVAL_GIF_EPISODES}"
echo "final eval GIF W&B section: ${GIF_PANEL_SECTION}"
echo "final eval GIF W&B upload limit: ${MAX_GIF_UPLOAD} (random=${RANDOM_GIF_UPLOAD})"

FINAL_EVAL_WANDB_RUN_STAMP="${FINAL_EVAL_WANDB_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
FINAL_EVAL_WANDB_RUN_NAME="${FINAL_EVAL_WANDB_RUN_NAME:-04_final_eval_${WANDB_RUN_PREFIX}_${PARTNER_SET_NAME}_seed${SEED:-1}_${FINAL_EVAL_WANDB_RUN_STAMP}}"
FINAL_EVAL_WANDB_RUN_ID="${FINAL_EVAL_WANDB_RUN_ID:-}"
STREAM_EVAL_GIFS_TO_WANDB="${STREAM_EVAL_GIFS_TO_WANDB:-0}"

PAIR_COMMON_ARGS=("${COMMON_ARGS[@]}")
case "${EVAL_PAIR_USE_WANDB:-0}" in
  1|true|True|TRUE|yes|Yes|YES)
    ;;
  *)
    PAIR_COMMON_ARGS+=(--use_wandb)
    ;;
esac

STREAM_WANDB_ARGS=()
case "${USE_WANDB:-1}:${STREAM_EVAL_GIFS_TO_WANDB}" in
  0:*|false:*|False:*|FALSE:*|no:*|No:*|NO:*|*:0|*:false|*:False|*:FALSE|*:no|*:No|*:NO)
    ;;
  *)
    if [[ -z "${FINAL_EVAL_WANDB_RUN_ID}" ]]; then
      echo "warning: stream eval GIFs need FINAL_EVAL_WANDB_RUN_ID; final uploader will still create a new W&B run"
    else
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
    fi
    ;;
esac

for partner_policy in "${PARTNER_POLICIES[@]}"; do
  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    "${STREAM_WANDB_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-mep_adaptive-${partner_policy}" \
    --wandb_stage_name 04_final_eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --adaptive_policy_name mep_adaptive \
    --agent0_policy_name mep_adaptive \
    --agent1_policy_name "${partner_policy}" \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --score_table_output_path "${SCORE_TABLE_CSV}" \
    --gif_manifest_output_path "${GIF_MANIFEST_CSV}" \
    --render_eval_gif_episodes "${RENDER_EVAL_GIF_EPISODES}" \
    --render_gif_subdir "mep_adaptive_vs_${partner_policy}" \
    --eval_stochastic | tee "${FINAL_EVAL_DIR}/mep_adaptive_vs_${partner_policy}.txt"

  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    "${STREAM_WANDB_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-${partner_policy}-mep_adaptive" \
    --wandb_stage_name 04_final_eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --adaptive_policy_name mep_adaptive \
    --agent0_policy_name "${partner_policy}" \
    --agent1_policy_name mep_adaptive \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --score_table_output_path "${SCORE_TABLE_CSV}" \
    --gif_manifest_output_path "${GIF_MANIFEST_CSV}" \
    --render_eval_gif_episodes "${RENDER_EVAL_GIF_EPISODES}" \
    --render_gif_subdir "${partner_policy}_vs_mep_adaptive" \
    --eval_stochastic | tee "${FINAL_EVAL_DIR}/${partner_policy}_vs_mep_adaptive.txt"
done

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
      --wandb-group "${WANDB_GROUP_NAME}"
    )
    if [[ -n "${FINAL_EVAL_WANDB_RUN_ID}" ]]; then
      UPLOAD_WANDB_ARGS+=(--wandb-run-id "${FINAL_EVAL_WANDB_RUN_ID}")
    fi
    ;;
esac

GIF_UPLOAD_ARGS=()
if [[ "${MAX_GIF_UPLOAD}" != "0" ]]; then
  GIF_UPLOAD_ARGS+=(--max-gif-upload "${MAX_GIF_UPLOAD}")
fi
case "${RANDOM_GIF_UPLOAD}" in
  1|true|True|TRUE|yes|Yes|YES)
    GIF_UPLOAD_ARGS+=(--random-gif-upload)
    ;;
esac
if [[ -n "${GIF_UPLOAD_SEED:-}" ]]; then
  GIF_UPLOAD_ARGS+=(--gif-upload-seed "${GIF_UPLOAD_SEED}")
fi

"${PYTHON}" "${CORL_ROOT}/transplant/scripts/upload_hsp_eval_event_table_risky.py" \
  --input "${EVENT_TABLE_CSV}" \
  --score-input "${SCORE_TABLE_CSV}" \
  --output-dir "${FINAL_EVAL_DIR}" \
  --gif-manifest "${GIF_MANIFEST_CSV}" \
  --gif-panel-section "${GIF_PANEL_SECTION}" \
  --layout "${LAYOUT}" \
  --adaptive-policy-name mep_adaptive \
  "${GIF_UPLOAD_ARGS[@]}" \
  "${UPLOAD_WANDB_ARGS[@]}"

printf '\n\n\n=== [04] MEP final evaluation end ===\n\n\n'
