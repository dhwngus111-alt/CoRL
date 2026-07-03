#!/usr/bin/env bash

set -euo pipefail

MEP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MEP_TRANSPLANT_ROOT="${MEP_TRANSPLANT_ROOT:-${MEP_SCRIPT_DIR}}"
export CORL_ROOT="${CORL_ROOT:-/home/isl_jhoh/CoRL}"
export TRANSPLANT_ROOT="${MEP_TRANSPLANT_ROOT}"
export TRANSPLANT_OUTPUT_ROOT="${MEP_TRANSPLANT_ROOT}"
export POLICY_POOL="${POLICY_POOL:-${MEP_TRANSPLANT_ROOT}/policy_pool}"
export LAYOUT="${LAYOUT:-risky_dualpath_subgoal}"
export WANDB_PROJECT="${WANDB_PROJECT:-MEP_${LAYOUT}}"
export WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-MEP_${LAYOUT}}"
export WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-MEP_${LAYOUT}}"

export MEP_SELECTED_POLICY_COUNT="${MEP_SELECTED_POLICY_COUNT:-${MEP_S1_POPULATION_SIZE:-5}}"
export MEP_CHECKPOINTS_PER_POLICY="${MEP_CHECKPOINTS_PER_POLICY:-3}"
export MEP_S2_POPULATION_SIZE="${MEP_S2_POPULATION_SIZE:-$((MEP_SELECTED_POLICY_COUNT * MEP_CHECKPOINTS_PER_POLICY))}"
export MEP_S2_STEPS_PER_PARTNER="${MEP_S2_STEPS_PER_PARTNER:-10000000}"
export HANDOFF_SHAPING="${HANDOFF_SHAPING:-1}"

source "${CORL_ROOT}/transplant/common.sh"

export MEP_PRIORITIZED_ALPHA="${MEP_PRIORITIZED_ALPHA:-3.0}"
# Source roots are read-only inputs from the original transplant pipeline.
# All new policy-pool, training, extraction, and eval outputs stay under
# MEP_TRANSPLANT_ROOT via TRANSPLANT_OUTPUT_ROOT above.
export MEP_SOURCE_POLICY_POOL="${MEP_SOURCE_POLICY_POOL:-${CORL_ROOT}/transplant/policy_pool}"
