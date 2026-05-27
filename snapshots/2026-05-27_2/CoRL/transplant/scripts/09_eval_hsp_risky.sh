#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 평가 및 테스트 단계에서는 순수한 sparse reward와 behavior event 분석을 위해 time_cost를 0.0으로 강제 고정한다.
export TIME_COST=0.0
source "${SCRIPT_DIR}/../common.sh"

printf '\n\n\n=== [09] HSP 최종 평가 시작 ===\n\n\n'
echo "final eval settings: EVAL_EPISODES=${EVAL_EPISODES}, N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS}"

mkdir -p "${TRANSPLANT_ROOT}/final_eval"
POP_YAML="${POLICY_POOL}/${LAYOUT}/hsp/s2/eval.yml"
EVENT_TABLE_CSV="${EVENT_TABLE_CSV:-${TRANSPLANT_ROOT}/final_eval/hsp_adaptive_event_counts_long.csv}"
: > "${EVENT_TABLE_CSV}"

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

PAIR_COMMON_ARGS=("${COMMON_ARGS[@]}")
case "${EVAL_PAIR_USE_WANDB:-0}" in
  1|true|True|TRUE|yes|Yes|YES)
    ;;
  *)
    PAIR_COMMON_ARGS+=(--use_wandb)
    ;;
esac

for partner_policy in "${PARTNER_POLICIES[@]}"; do
  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-hsp_adaptive-${partner_policy}" \
    --wandb_stage_name 09_final-eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --agent0_policy_name hsp_adaptive \
    --agent1_policy_name "${partner_policy}" \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --eval_stochastic | tee "${TRANSPLANT_ROOT}/final_eval/hsp_adaptive_vs_${partner_policy}.txt"

  "${PYTHON}" -m transplant.eval_risky_hsp \
    "${PAIR_COMMON_ARGS[@]}" \
    --algorithm_name population \
    --experiment_name "final-${partner_policy}-hsp_adaptive" \
    --wandb_stage_name 09_final-eval \
    --seed "${SEED:-1}" \
    --population_yaml_path "${POP_YAML}" \
    --agent0_policy_name "${partner_policy}" \
    --agent1_policy_name hsp_adaptive \
    --event_table_output_path "${EVENT_TABLE_CSV}" \
    --eval_stochastic | tee "${TRANSPLANT_ROOT}/final_eval/${partner_policy}_vs_hsp_adaptive.txt"
done

UPLOAD_WANDB_ARGS=()
case "${USE_WANDB:-1}" in
  0|false|False|FALSE|no|No|NO)
    ;;
  *)
    UPLOAD_WANDB_ARGS=(
      --use-wandb
      --wandb-project "RiskyOvercooked"
      --wandb-entity "${WANDB_ENTITY}"
      --wandb-mode "${WANDB_MODE:-online}"
      --wandb-run-name "09_final-eval-event-table-${LAYOUT}-${PARTNER_SET_NAME}-seed${SEED:-1}"
    )
    ;;
esac

"${PYTHON}" "${SCRIPT_DIR}/upload_hsp_eval_event_table_risky.py" \
  --input "${EVENT_TABLE_CSV}" \
  --output-dir "${TRANSPLANT_ROOT}/final_eval" \
  --layout "${LAYOUT}" \
  "${UPLOAD_WANDB_ARGS[@]}"

printf '\n\n\n=== [09] HSP 최종 평가 종료 ===\n\n\n'
