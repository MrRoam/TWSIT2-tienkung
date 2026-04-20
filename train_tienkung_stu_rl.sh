#!/bin/bash
set -euo pipefail

# Usage:
#   bash train_tienkung_stu_rl.sh <exptid> <teacher_exptid> <teacher_checkpoint> <device> <num_envs> <max_iterations> <motion_yaml> [extra args...]

EXPTID=${1:-tk_sturl_$(date +%m%d_%H%M)}
TEACHER_EXPTID=${2:-tienkung_priv_placeholder}
TEACHER_CHECKPOINT=${3:--1}
DEVICE=${4:-cuda:0}
NUM_ENVS=${5:-256}
MAX_ITER=${6:-10000}
MOTION_YAML=${7:-/home/qsh/workspace_twist2/TWIST2/legged_gym/motion_data_configs/tienkung_ei_train_fullbody_no_object.yaml}

if [ "$#" -gt 7 ]; then
  shift 7
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[train_tienkung_stu_rl] WANDB_API_KEY is not set, append --no_wandb."
  EXTRA_ARGS+=("--no_wandb")
fi

cd /home/qsh/workspace_twist2/TWIST2/legged_gym/legged_gym/scripts

LD_LIBRARY_PATH=/home/qsh/miniconda3/envs/twist2/lib:${LD_LIBRARY_PATH:-} \
/home/qsh/miniconda3/envs/twist2/bin/python train.py \
  --task tienkung_stu_rl \
  --proj_name tienkung_stu_rl \
  --exptid "${EXPTID}" \
  --teacher_exptid "${TEACHER_EXPTID}" \
  --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
  --device "${DEVICE}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITER}" \
  --env.motion.motion_file "${MOTION_YAML}" \
  "${EXTRA_ARGS[@]}"
