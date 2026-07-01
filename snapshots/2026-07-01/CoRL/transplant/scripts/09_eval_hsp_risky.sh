#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TIME_COST=0
source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [09] HSP 최종 평가 시작 ===\n\n\n'
echo "final eval settings: EVAL_EPISODES=${EVAL_EPISODES}, N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS}, TIME_COST=${TIME_COST}"

FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-${TRANSPLANT_ROOT}/final_eval/${LAYOUT}}"
mkdir -p "${FINAL_EVAL_DIR}"
POP_YAML="${POLICY_POOL}/${LAYOUT}/hsp/s2/eval.yml"
EVENT_TABLE_CSV="${EVENT_TABLE_CSV:-${FINAL_EVAL_DIR}/hsp_adaptive_event_counts_long.csv}"
SCORE_TABLE_CSV="${SCORE_TABLE_CSV:-${FINAL_EVAL_DIR}/hsp_adaptive_sparse_scores.csv}"
GIF_MANIFEST_CSV="${GIF_MANIFEST_CSV:-${FINAL_EVAL_DIR}/hsp_adaptive_pair_gifs.csv}"
RENDER_EVAL_GIF_EPISODES="${RENDER_EVAL_GIF_EPISODES:-1}"
GIF_PANEL_SECTION="${GIF_PANEL_SECTION:-hsp_final_eval_gifs}"
: > "${EVENT_TABLE_CSV}"
: > "${SCORE_TABLE_CSV}"
: > "${GIF_MANIFEST_CSV}"

if [[ -n "${PARTNER_POLICY:-}" ]]; then
  PARTNER_POLICIES=("${PARTNER_POLICY}")
  PARTNER_SET_NAME="${PARTNER_POLICY}"
else
  PARTNER_GROUP="${PARTNER_GROUP:-all}"
  PARTNER_SET_NAME="${PARTNER_GROUP}"
  mapfile -t PARTNER_POLICIES < <("${PYTHON}" - "${POP_YAML}" "${PARTNER_GROUP}" <<'PY'
import sys
import yaml

path, group = sys.argv[1], sys.argv[2]
data = yaml.safe_load(open(path)) or {}
partners = [name for name in data if name != "hsp_adaptive"]
if group == "mep":
    partners = [name for name in partners if name.startswith("mep")]
elif group == "hsp":
    partners = [name for name in partners if name.startswith("hsp")]
elif group != "all":
    raise SystemExit(f"invalid PARTNER_GROUP={group}: expected all, mep, or hsp")
if not partners:
    raise SystemExit(f"error: no partner policies found in {path}")
print("\n".join(partners))
PY
)
fi

echo "final eval partners (${#PARTNER_POLICIES[@]}): ${PARTNER_POLICIES[*]}"
echo "final eval GIFs per direction: ${RENDER_EVAL_GIF_EPISODES}"
echo "final eval GIF W&B section: ${GIF_PANEL_SECTION}"

FINAL_EVAL_WANDB_RUN_NAME="${FINAL_EVAL_WANDB_RUN_NAME:-09_final_eval_${WANDB_RUN_PREFIX}_${PARTNER_SET_NAME}_seed${SEED:-1}}"
FINAL_EVAL_WANDB_RUN_ID="${FINAL_EVAL_WANDB_RUN_ID:-09_final_eval_${WANDB_RUN_PREFIX}_${PARTNER_SET_NAME}_seed${SEED:-1}}"
STREAM_EVAL_GIFS_TO_WANDB="${STREAM_EVAL_GIFS_TO_WANDB:-${USE_WANDB:-1}}"

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

for partner_policy in "${PARTNER_POLICIES[@]}"; do
  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    "${STREAM_WANDB_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-hsp_adaptive-${partner_policy}" \
    --wandb_stage_name 09_final_eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --agent0_policy_name hsp_adaptive \
    --agent1_policy_name "${partner_policy}" \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --score_table_output_path "${SCORE_TABLE_CSV}" \
    --gif_manifest_output_path "${GIF_MANIFEST_CSV}" \
    --render_eval_gif_episodes "${RENDER_EVAL_GIF_EPISODES}" \
    --render_gif_subdir "hsp_adaptive_vs_${partner_policy}" \
    --eval_stochastic | tee "${FINAL_EVAL_DIR}/hsp_adaptive_vs_${partner_policy}.txt"

  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    "${STREAM_WANDB_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-${partner_policy}-hsp_adaptive" \
    --wandb_stage_name 09_final_eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --agent0_policy_name "${partner_policy}" \
    --agent1_policy_name hsp_adaptive \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --score_table_output_path "${SCORE_TABLE_CSV}" \
    --gif_manifest_output_path "${GIF_MANIFEST_CSV}" \
    --render_eval_gif_episodes "${RENDER_EVAL_GIF_EPISODES}" \
    --render_gif_subdir "${partner_policy}_vs_hsp_adaptive" \
    --eval_stochastic | tee "${FINAL_EVAL_DIR}/${partner_policy}_vs_hsp_adaptive.txt"
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
      --wandb-run-id "${FINAL_EVAL_WANDB_RUN_ID}"
      --wandb-group "${WANDB_GROUP_NAME}"
    )
    ;;
esac

"${PYTHON}" "${SCRIPT_DIR}/upload_hsp_eval_event_table_risky.py" \
  --input "${EVENT_TABLE_CSV}" \
  --score-input "${SCORE_TABLE_CSV}" \
  --output-dir "${FINAL_EVAL_DIR}" \
  --gif-manifest "${GIF_MANIFEST_CSV}" \
  --gif-panel-section "${GIF_PANEL_SECTION}" \
  --layout "${LAYOUT}" \
  "${UPLOAD_WANDB_ARGS[@]}"

printf '\n\n\n=== [09] HSP 최종 평가 종료 ===\n\n\n'
