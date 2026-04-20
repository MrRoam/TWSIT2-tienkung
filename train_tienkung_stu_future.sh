#!/bin/bash
set -euo pipefail

# Usage:
#   bash train_tienkung_stu_future.sh <exptid> <device> <num_envs> <max_iterations> <motion_yaml> [extra args...]
# Legacy teacher args can still be passed through extra args, e.g.
#   --teacher_exptid some_teacher --teacher_checkpoint 1000

EXPTID=${1:-tk_stufut_$(date +%m%d_%H%M)}
DEVICE=${2:-cuda:0}
NUM_ENVS=${3:-256}
MAX_ITER=${4:-10000}
MOTION_YAML=${5:-/home/qsh/workspace_twist2/TWIST2/legged_gym/motion_data_configs/tienkung_ei_train_fullbody_no_object.yaml}

if [ "$#" -gt 5 ]; then
  shift 5
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[train_tienkung_stu_future] WANDB_API_KEY is not set, append --no_wandb."
  EXTRA_ARGS+=("--no_wandb")
fi

cd /home/qsh/workspace_twist2/TWIST2/legged_gym/legged_gym/scripts

LD_LIBRARY_PATH=/home/qsh/miniconda3/envs/twist2/lib:${LD_LIBRARY_PATH:-} \
/home/qsh/miniconda3/envs/twist2/bin/python train.py \
  --task tienkung_stu_future \
  --proj_name tienkung_stu_future \
  --exptid "${EXPTID}" \
  --teacher_exptid "None" \
  --teacher_checkpoint -1 \
  --device "${DEVICE}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITER}" \
  --env.motion.motion_file "${MOTION_YAML}" \
  "${EXTRA_ARGS[@]}"
