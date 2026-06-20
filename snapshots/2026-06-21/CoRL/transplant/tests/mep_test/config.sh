#!/usr/bin/env bash

# MEP-vs-random 평가 전용 W&B 설정.
export WANDB_PROJECT="${WANDB_PROJECT:-mep_random_test}"
export WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-MEP_eval_risky_multipath_subgoal}"
export WANDB_ENTITY="${WANDB_ENTITY:-dhwngus41-daegu-gyeongbuk-institute-of-science-technology}"
export WANDB_MODE="${WANDB_MODE:-online}"

# 물리 GPU 3만 평가 프로세스에 노출한다. 프로세스 내부에서는 cuda:0으로 보인다.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
