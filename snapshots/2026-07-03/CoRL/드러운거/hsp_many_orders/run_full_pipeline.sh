#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/home/isl_jhoh/CoRL/test/hsp_many_orders/scripts"

"${SCRIPT_DIR}/00_check_env.sh"
"${SCRIPT_DIR}/01_train_mep_s1_many_orders.sh"
"${SCRIPT_DIR}/02_extract_mep_s1_many_orders.py"
"${SCRIPT_DIR}/03_train_hsp_s1_many_orders.sh"
"${SCRIPT_DIR}/04_extract_hsp_s1_many_orders.py"
"${SCRIPT_DIR}/05_eval_events_many_orders.sh"
"${SCRIPT_DIR}/06_greedy_select_many_orders.sh"
"${SCRIPT_DIR}/07_train_hsp_s2_many_orders.sh"
"${SCRIPT_DIR}/08_extract_hsp_s2_many_orders.py"
"${SCRIPT_DIR}/09_eval_hsp_many_orders_wandb.sh"

