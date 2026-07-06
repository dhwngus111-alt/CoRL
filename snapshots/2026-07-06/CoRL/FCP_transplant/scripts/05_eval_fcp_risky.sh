#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# 평가는 time_cost=0으로 고정한다 (transplant eval 관례).
export TIME_COST=0
source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [05] FCP 최종 평가 시작 ===\n\n\n'
echo "settings: EVAL_EPISODES=${EVAL_EPISODES}, N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS}, TIME_COST=${TIME_COST}"

POP_YAML="${POLICY_POOL}/${LAYOUT}/fcp/s2/eval.yml"
if [[ ! -f "${POP_YAML}" ]]; then
  echo "missing ${POP_YAML}: run 04_extract_fcp_s2_risky.py first" >&2
  exit 1
fi

ADAPTIVE_NAME="${FCP_ADAPTIVE_AGENT_NAME:-fcp_adaptive}"
RANDOM_POLICY_NAME="${RANDOM_POLICY_NAME:-random}"
EVAL_RANDOM_PARTNER="${EVAL_RANDOM_PARTNER:-1}"
RENDER_EVAL_GIF_EPISODES="${RENDER_EVAL_GIF_EPISODES:-1}"
GIF_PANEL_SECTION="${GIF_PANEL_SECTION:-fcp_final_eval_gifs}"

FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-${TRANSPLANT_OUTPUT_ROOT}/final_eval/${LAYOUT}}"
mkdir -p "${FINAL_EVAL_DIR}"
EVENT_TABLE_CSV="${FINAL_EVAL_DIR}/fcp_adaptive_event_counts_long.csv"
SCORE_TABLE_CSV="${FINAL_EVAL_DIR}/fcp_adaptive_sparse_scores.csv"
GIF_MANIFEST_CSV="${FINAL_EVAL_DIR}/fcp_adaptive_pair_gifs.csv"
: > "${EVENT_TABLE_CSV}"
: > "${SCORE_TABLE_CSV}"
: > "${GIF_MANIFEST_CSV}"

# 파트너 = policy pool(yaml에서 adaptive/random 제외) [+ uniform-random].
mapfile -t PARTNER_POLICIES < <("${PYTHON}" - "${POP_YAML}" "${ADAPTIVE_NAME}" "${RANDOM_POLICY_NAME}" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1])) or {}
adaptive, rand = sys.argv[2], sys.argv[3]
print("\n".join(n for n in data if n not in (adaptive, rand)))
PY
)
if [[ "${#PARTNER_POLICIES[@]}" -eq 0 ]]; then
  echo "error: no partner policies in ${POP_YAML}" >&2
  exit 1
fi
case "${EVAL_RANDOM_PARTNER}" in
  1|true|True|TRUE|yes|Yes|YES)
    PARTNER_POLICIES+=("${RANDOM_POLICY_NAME}")
    ;;
esac
echo "eval partners (${#PARTNER_POLICIES[@]}): ${PARTNER_POLICIES[*]}"

# HSP 09 / CoMeDi 03과 동일: per-pair 개별 W&B run은 만들지 않고, GIF/집계를 하나의 final-eval run에 누적.
FINAL_EVAL_WANDB_RUN_NAME="${FINAL_EVAL_WANDB_RUN_NAME:-${SCRIPT_NAME}_seed${SEED:-1}}"
FINAL_EVAL_WANDB_RUN_ID="${FINAL_EVAL_WANDB_RUN_ID:-${SCRIPT_NAME}_seed${SEED:-1}}"
STREAM_EVAL_GIFS_TO_WANDB="${STREAM_EVAL_GIFS_TO_WANDB:-${USE_WANDB:-1}}"

# per-pair eval의 자체 W&B run은 끔(--use_wandb는 store_false → off).
PAIR_COMMON_ARGS=("${COMMON_ARGS[@]}")
case " ${PAIR_COMMON_ARGS[*]} " in
  *" --use_wandb "*) ;;
  *) PAIR_COMMON_ARGS+=(--use_wandb) ;;
esac

# GIF를 하나의 final-eval run으로 스트리밍(partner_group별 policy_pool_gif / random_gif 섹션).
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
    echo "stream eval GIFs to one final run: ${FINAL_EVAL_WANDB_RUN_ID}"
    ;;
esac

for partner in "${PARTNER_POLICIES[@]}"; do
  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    "${STREAM_WANDB_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-${ADAPTIVE_NAME}-${partner}" \
    --wandb_stage_name 05_final_eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --adaptive_policy_name "${ADAPTIVE_NAME}" \
    --agent0_policy_name "${ADAPTIVE_NAME}" \
    --agent1_policy_name "${partner}" \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --score_table_output_path "${SCORE_TABLE_CSV}" \
    --gif_manifest_output_path "${GIF_MANIFEST_CSV}" \
    --render_eval_gif_episodes "${RENDER_EVAL_GIF_EPISODES}" \
    --render_gif_subdir "${ADAPTIVE_NAME}_vs_${partner}" \
    --eval_stochastic | tee "${FINAL_EVAL_DIR}/${ADAPTIVE_NAME}_vs_${partner}.txt"

  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    "${STREAM_WANDB_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-${partner}-${ADAPTIVE_NAME}" \
    --wandb_stage_name 05_final_eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --adaptive_policy_name "${ADAPTIVE_NAME}" \
    --agent0_policy_name "${partner}" \
    --agent1_policy_name "${ADAPTIVE_NAME}" \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --score_table_output_path "${SCORE_TABLE_CSV}" \
    --gif_manifest_output_path "${GIF_MANIFEST_CSV}" \
    --render_eval_gif_episodes "${RENDER_EVAL_GIF_EPISODES}" \
    --render_gif_subdir "${partner}_vs_${ADAPTIVE_NAME}" \
    --eval_stochastic | tee "${FINAL_EVAL_DIR}/${partner}_vs_${ADAPTIVE_NAME}.txt"
done

# 집계 이벤트/점수 테이블 + GIF를 같은 final-eval run에 업로드 (HSP 09 / CoMeDi 03과 동일).
# partner_group(policy_pool/random)별로 policy_pool_eval / random_eval, policy_pool_gif / random_gif 섹션 생성.
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
    --adaptive-policy-name "${ADAPTIVE_NAME}" \
    --layout "${LAYOUT}" \
    "${UPLOAD_WANDB_ARGS[@]}" || echo "warning: eval table upload failed (non-fatal)"
fi

echo "score table:  ${SCORE_TABLE_CSV}"
echo "event table:  ${EVENT_TABLE_CSV}"
printf '\n\n\n=== [05] FCP 최종 평가 종료 ===\n\n\n'
