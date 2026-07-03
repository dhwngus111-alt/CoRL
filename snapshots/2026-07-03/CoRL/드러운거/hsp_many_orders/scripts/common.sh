#!/usr/bin/env bash
set -euo pipefail

export HSP_ROOT="${HSP_ROOT:-/home/isl_jhoh/CoRL/HSP}"
export TEST_ROOT="${TEST_ROOT:-/home/isl_jhoh/CoRL/test/hsp_many_orders}"
export ORIG_HSP_SCRIPT_DIR="${ORIG_HSP_SCRIPT_DIR:-${HSP_ROOT}/hsp/scripts}"
export HSP_SCRIPT_DIR="${HSP_SCRIPT_DIR:-${TEST_ROOT}}"
export POLICY_POOL="${POLICY_POOL:-${TEST_ROOT}/policy_pool}"
export STUB_ROOT="${STUB_ROOT:-${TEST_ROOT}/stubs}"
export LOG_DIR="${LOG_DIR:-${TEST_ROOT}/logs}"
export RESULTS_ROOT="${RESULTS_ROOT:-${HSP_SCRIPT_DIR}/results}"
export BIASED_EVAL_ROOT="${BIASED_EVAL_ROOT:-${TEST_ROOT}/biased_eval}"
export FINAL_EVAL_ROOT="${FINAL_EVAL_ROOT:-${TEST_ROOT}/final_eval}"
export PYTHON_BIN="${PYTHON_BIN:-/home/isl_jhoh/miniconda3/envs/corl/bin/python}"
export GPU="${GPU:-1}"
export USER_NAME="${USER_NAME:-${USER:-isl_jhoh}}"
export DEFAULT_WANDB_ENTITY="${DEFAULT_WANDB_ENTITY:-dhwngus41-daegu-gyeongbuk-institute-of-science-technology}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
if [[ -z "${WANDB__SERVICE_WAIT:-}" || "${WANDB__SERVICE_WAIT}" -lt 300 ]]; then
  export WANDB__SERVICE_WAIT=300
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TEST_ROOT}/.matplotlib}"
export PYTHONPATH="${STUB_ROOT}:${HSP_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}" "${RESULTS_ROOT}" "${BIASED_EVAL_ROOT}" "${FINAL_EVAL_ROOT}" "${POLICY_POOL}"

ensure_link() {
  local link_path="$1"
  local target_path="$2"

  if [[ ! -e "${target_path}" ]]; then
    echo "Missing runtime target: ${target_path}" >&2
    return 2
  fi

  if [[ -L "${link_path}" ]]; then
    local current_target
    current_target="$(readlink "${link_path}")"
    if [[ "${current_target}" != "${target_path}" ]]; then
      echo "${link_path} points to ${current_target}, expected ${target_path}" >&2
      return 2
    fi
    return 0
  fi

  if [[ -e "${link_path}" ]]; then
    echo "${link_path} already exists and is not a symlink." >&2
    return 2
  fi

  ln -s "${target_path}" "${link_path}"
}

sync_policy_pool_templates() {
  local src="${HSP_ROOT}/hsp/policy_pool/many_orders"
  local dst="${POLICY_POOL}/many_orders"

  if [[ ! -d "${src}" ]]; then
    echo "Missing source policy templates: ${src}" >&2
    return 2
  fi

  mkdir -p "${dst}"

  while IFS= read -r rel_dir; do
    mkdir -p "${dst}/${rel_dir#./}"
  done < <(cd "${src}" && find . -type d -print)

  while IFS= read -r rel_file; do
    local clean="${rel_file#./}"
    cp -n "${src}/${clean}" "${dst}/${clean}"
  done < <(cd "${src}" && find . -type f \( -name '*.yml' -o -name '*.pkl' \) -print)
}

prepare_runtime_layout() {
  mkdir -p "${HSP_SCRIPT_DIR}" "${HSP_SCRIPT_DIR}/hsp"
  ensure_link "${HSP_SCRIPT_DIR}/train" "${ORIG_HSP_SCRIPT_DIR}/train"
  ensure_link "${HSP_SCRIPT_DIR}/eval" "${ORIG_HSP_SCRIPT_DIR}/eval"
  ensure_link "${HSP_SCRIPT_DIR}/hsp/greedy_select.py" "${ORIG_HSP_SCRIPT_DIR}/hsp/greedy_select.py"
  sync_policy_pool_templates
}

prepare_runtime_layout

resolve_wandb_entity() {
  if [[ "${WANDB_ENTITY:-}" == "dhwngus41" ]]; then
    export WANDB_ENTITY="${DEFAULT_WANDB_ENTITY}"
  fi

  if [[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "WANDB_NAME" && "${WANDB_ENTITY}" != "wandb_name" && "${WANDB_ENTITY}" != "user" ]]; then
    if [[ "${WANDB_ENTITY}" == "${USER_NAME}" ]]; then
      local logged_in_guess=""
      logged_in_guess="$(
        find "${LOG_DIR}" "${RESULTS_ROOT}" -type f 2>/dev/null \
          | xargs -r grep -h "Currently logged in as:" 2>/dev/null \
          | tail -n 1 \
          | sed -E 's/.*Currently logged in as: ([^ ]+) .*/\1/'
      )"
      logged_in_guess="$(echo "${logged_in_guess}" | tr -d '[:space:]')"
      if [[ -n "${logged_in_guess}" && "${logged_in_guess}" != "${WANDB_ENTITY}" ]]; then
        export WANDB_ENTITY="${logged_in_guess}"
      fi
    fi
    return 0
  fi

  local guessed=""
  guessed="$(timeout 12 "${PYTHON_BIN}" - <<'PY' || true
import contextlib
import io

try:
    import wandb

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        viewer = wandb.Api(timeout=6).viewer
    print(getattr(viewer, "entity", None) or getattr(viewer, "username", None) or "")
except Exception:
    pass
PY
)"
  guessed="$(echo "${guessed}" | tr -d '[:space:]')"

  if [[ -n "${guessed}" && "${guessed}" != "user" && "${guessed}" != "wandb_name" && "${guessed}" != "WANDB_NAME" ]]; then
    export WANDB_ENTITY="${guessed}"
    return 0
  fi

  guessed="$(
    find "${LOG_DIR}" "${RESULTS_ROOT}" -type f 2>/dev/null \
      | xargs -r grep -h "Currently logged in as:" 2>/dev/null \
      | tail -n 1 \
      | sed -E 's/.*Currently logged in as: ([^ ]+) .*/\1/'
  )"
  guessed="$(echo "${guessed}" | tr -d '[:space:]')"

  if [[ -n "${guessed}" && "${guessed}" != "user" && "${guessed}" != "wandb_name" && "${guessed}" != "WANDB_NAME" ]]; then
    export WANDB_ENTITY="${guessed}"
    return 0
  fi

  if [[ "${WANDB_MODE}" == "offline" || "${WANDB_MODE}" == "disabled" ]]; then
    export WANDB_ENTITY="${USER_NAME}"
    return 0
  fi

  echo "Could not resolve a valid W&B entity. Run 'wandb login --relogin' or export WANDB_ENTITY=<your_wandb_username_or_team>." >&2
  return 2
}

require_wandb_entity() {
  resolve_wandb_entity
}

cd_hsp_scripts() {
  cd "${HSP_SCRIPT_DIR}"
}
