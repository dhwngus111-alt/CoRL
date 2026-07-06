#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# FCP_transplant 공통 설정 (transplant/common.sh 미러 + FCP 하이퍼파라미터).
#
# 이 파일은 FCP_transplant/scripts/*.sh에서 공통으로 source된다.  경로/PYTHONPATH,
# risked 환경 인자, PPO 기본값을 한곳에서 관리한다.  transplant 드라이버(train_risky_*)
# 를 그대로 재사용하되, 결과/policy_pool 만 FCP_transplant/ 아래로 분리한다.
#
# 하이퍼파라미터는 HSP repo의 FCP baseline을 따른다:
#   - S1(self-play): train_overcooked_sp.sh  (seed 1..12, threads 100, 1e7 steps)
#   - Extract:       extract_sp_S1_models.py (seed당 init/mid/final = 3*12=36)
#   - S2(adaptive):  train_overcooked_fcp_stage_2.sh (threads 300, 1e8 steps, pop 36, uniform)
# risked 환경 고유 인자(p_slip/time_cost/subgoal 등)는 transplant 연구의 공유 설정을
# 그대로 상속하여 다른 방법(MEP/HSP)과 공정 비교가 되도록 한다.
# ============================================================================

# --- 경로: common.sh 위치에서 유도 (머신 독립) ---------------------------------
FCP_TRANSPLANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FCP_TRANSPLANT_ROOT
export CORL_ROOT="${CORL_ROOT:-$(cd "${FCP_TRANSPLANT_ROOT}/.." && pwd)}"
export TRANSPLANT_CODE_ROOT="${TRANSPLANT_CODE_ROOT:-${CORL_ROOT}/transplant}"
export HSP_ROOT="${HSP_ROOT:-${CORL_ROOT}/HSP}"
export RISKED_ROOT="${RISKED_ROOT:-${CORL_ROOT}/risked_overcooked}"
export RISKY_ROOT="${RISKED_ROOT}"

# transplant 드라이버가 결과/pool을 FCP_transplant/ 아래에 쓰도록 강제한다.
export TRANSPLANT_OUTPUT_ROOT="${TRANSPLANT_OUTPUT_ROOT:-${FCP_TRANSPLANT_ROOT}}"
export POLICY_POOL="${POLICY_POOL:-${TRANSPLANT_OUTPUT_ROOT}/policy_pool}"

# CoRL(=transplant, FCP_transplant), HSP, risked_overcooked/src, transplant/compat 를 import 가능하게.
export PYTHONPATH="${CORL_ROOT}:${HSP_ROOT}:${RISKED_ROOT}/src:${TRANSPLANT_CODE_ROOT}/compat:${PYTHONPATH:-}"

# Python 실행기.  외부에서 PYTHON=... 을 export하면 우선한다.
# 이 서버의 프로젝트 conda env (torch 2.4.1+cu124).
DEFAULT_PYTHON="/home/isllab0213/miniconda3/envs/risky_overcooked/bin/python"
if [[ -z "${PYTHON:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON="${DEFAULT_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi

# --- Risked Overcooked 환경 기본값 (transplant 연구와 동일하게 공유) ------------
LAYOUT="${LAYOUT:-risky_dualpath_subgoal}"
NUM_AGENTS="${NUM_AGENTS:-2}"
EPISODE_LENGTH="${EPISODE_LENGTH:-200}"
SUBGOAL_DISABLE_STEPS="${SUBGOAL_DISABLE_STEPS:-60}"
P_SLIP="${P_SLIP:-0.4}"
TIME_COST="${TIME_COST:--0.3}"
DISTANCE_SHAPING_REW="${DISTANCE_SHAPING_REW:-0.3}"
SUBGOAL_PRESS_REW="${SUBGOAL_PRESS_REW:-2}"
HANDOFF_SHAPING="${HANDOFF_SHAPING:-0}"
COMPETITIVE_ONION_SUPPLY="${COMPETITIVE_ONION_SUPPLY:-}"

# CNN 관측 encoder 구조 (HSP FCP와 동일) + GPU.
CNN_LAYERS_PARAMS="${CNN_LAYERS_PARAMS:-32,3,1,1 64,3,1,1 32,3,1,1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

_fcp_probe_cuda() {
  "${PYTHON}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"python={sys.version.split()[0]} torch_import_error={type(exc).__name__}: {exc}")
    raise SystemExit(1)

parts = [
    f"python={sys.version.split()[0]}",
    f"torch={torch.__version__}",
    f"torch_cuda={torch.version.cuda}",
]
try:
    available = torch.cuda.is_available()
except Exception as exc:
    parts.append(f"is_available_error={type(exc).__name__}: {exc}")
    print(" ".join(parts))
    raise SystemExit(1)

parts.append(f"available={available}")
parts.append(f"device_count={torch.cuda.device_count()}")
if not available:
    print(" ".join(parts))
    raise SystemExit(1)

try:
    torch.empty(1, device="cuda:0")
except Exception as exc:
    parts.append(f"alloc_error={type(exc).__name__}: {exc}")
    print(" ".join(parts))
    raise SystemExit(1)

try:
    parts.append(f"device0={torch.cuda.get_device_name(0)}")
except Exception:
    pass
parts.append("alloc=ok")
print(" ".join(parts))
PY
}

FCP_DEVICE="${FCP_DEVICE:-cuda}"
FCP_SELECTED_DEVICE=""
FCP_CUDA_PROBE_RESULT=""
FCP_DEVICE_ARGS=()
case "${FCP_DEVICE}" in
  cuda)
    if ! FCP_CUDA_PROBE_RESULT="$(_fcp_probe_cuda 2>&1)"; then
      echo "error: FCP_DEVICE=cuda but CUDA preflight failed." >&2
      echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
      echo "${FCP_CUDA_PROBE_RESULT}" >&2
      echo "Set FCP_DEVICE=cpu for intentional CPU smoke/debug runs." >&2
      exit 1
    fi
    FCP_SELECTED_DEVICE="cuda"
    ;;
  cpu)
    FCP_SELECTED_DEVICE="cpu"
    FCP_CUDA_PROBE_RESULT="skipped (FCP_DEVICE=cpu)"
    FCP_DEVICE_ARGS=(--cuda)
    ;;
  auto)
    if FCP_CUDA_PROBE_RESULT="$(_fcp_probe_cuda 2>&1)"; then
      FCP_SELECTED_DEVICE="cuda"
    else
      echo "warning: FCP_DEVICE=auto could not use CUDA; falling back to CPU." >&2
      echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
      echo "${FCP_CUDA_PROBE_RESULT}" >&2
      FCP_SELECTED_DEVICE="cpu"
      FCP_DEVICE_ARGS=(--cuda)
    fi
    ;;
  *)
    echo "error: invalid FCP_DEVICE=${FCP_DEVICE}; expected cuda, cpu, or auto" >&2
    exit 1
    ;;
esac
export FCP_DEVICE FCP_SELECTED_DEVICE FCP_CUDA_PROBE_RESULT

# --- 학습/평가 스케일 (stage 스크립트가 export로 덮어씀) -------------------------
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-100}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-1}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"

# FCP population: SP seed 12개 -> extract 시 init/mid/final = 36 partners.
SEED_MAX="${SEED_MAX:-12}"
FCP_S2_POPULATION_SIZE="${FCP_S2_POPULATION_SIZE:-36}"

# checkpoint/log 주기 + PPO 기본값 (HSP config/FCP script 정합).
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
TRAIN_ENV_BATCH="${TRAIN_ENV_BATCH:-1}"
PPO_EPOCH="${PPO_EPOCH:-15}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
REWARD_SHAPING_HORIZON="${REWARD_SHAPING_HORIZON:-100000000}"

# --- W&B --------------------------------------------------------------------
# entity 는 transplant(같은 사용자/프로젝트)와 동일하게 둔다.  다른 계정이면
# WANDB_ENTITY=... 로 override.  USE_WANDB=0 이면 W&B 없이 로컬 tensorboard로 기록.
# 별도 프로젝트(RiskyOvercooked_FCP_*)로 분리하며, HSP/MEP와 한 프로젝트에서
# 비교하려면 WANDB_PROJECT=RiskyOvercooked_${LAYOUT} 로 override 하면 된다.
WANDB_ENTITY="${WANDB_ENTITY:-isllab}"
WANDB_PROJECT="${WANDB_PROJECT:-RiskyOvercooked_FCP_${LAYOUT}}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-FCP_${LAYOUT}}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-FCP_${LAYOUT}}"

# --- 공통 CLI 인자 -----------------------------------------------------------
# 개별 script는 여기에 algorithm_name/experiment_name/stage/population_yaml_path 등을 추가한다.
COMMON_ARGS=(
  --env_name RiskyOvercooked
  --layout_name "${LAYOUT}"
  --num_agents "${NUM_AGENTS}"
  --episode_length "${EPISODE_LENGTH}"
  --subgoal_disable_steps "${SUBGOAL_DISABLE_STEPS}"
  --p_slip "${P_SLIP}"
  --time_cost "${TIME_COST}"
  --distance_shaping_rew "${DISTANCE_SHAPING_REW}"
  --subgoal_press_rew "${SUBGOAL_PRESS_REW}"
  --reward_shaping_horizon "${REWARD_SHAPING_HORIZON}"
  --n_training_threads 1
  --n_rollout_threads "${N_ROLLOUT_THREADS}"
  --n_eval_rollout_threads "${N_EVAL_ROLLOUT_THREADS}"
  --num_env_steps "${NUM_ENV_STEPS}"
  --eval_episodes "${EVAL_EPISODES}"
  --ppo_epoch "${PPO_EPOCH}"
  --num_mini_batch "${NUM_MINI_BATCH}"
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --cnn_layers_params "${CNN_LAYERS_PARAMS}"
  --user_name risky_fcp
  --wandb_project "${WANDB_PROJECT}"
  --wandb_group_name "${WANDB_GROUP_NAME}"
  --wandb_run_prefix "${WANDB_RUN_PREFIX}"
)
if [[ "${#FCP_DEVICE_ARGS[@]}" -gt 0 ]]; then
  COMMON_ARGS+=("${FCP_DEVICE_ARGS[@]}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb_name "${WANDB_ENTITY}")
fi

case "${HANDOFF_SHAPING}" in
  1|true|True|TRUE|yes|Yes|YES)
    COMMON_ARGS+=(--handoff_shaping)
    ;;
esac
if [[ -n "${COMPETITIVE_ONION_SUPPLY}" ]]; then
  COMMON_ARGS+=(--competitive_onion_supply "${COMPETITIVE_ONION_SUPPLY}")
fi

# W&B 사용 여부.  주의: HSP parser에서 --use_wandb 플래그는 store_false 의미다.
# USE_WANDB=1이면 아무것도 추가하지 않고, USE_WANDB=0일 때만 --use_wandb 를 추가한다.
USE_WANDB="${USE_WANDB:-1}"
case "${USE_WANDB}" in
  0|false|False|FALSE|no|No|NO)
    COMMON_ARGS+=(--use_wandb)
    ;;
esac

# policy architecture shortcut.
# HSP config에서 use_recurrent_policy 는 store_false(기본 True)라, 이 플래그를 주면
# recurrent가 꺼져 MLP가 된다.  HSP FCP-S1(train_overcooked_sp.sh)의 `mappo +
# --use_recurrent_policy` 조합과 정확히 일치한다.
MLP_POLICY_ARGS=(--algorithm_name mappo --use_recurrent_policy)
RNN_POLICY_ARGS=(--algorithm_name rmappo)
