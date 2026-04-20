#!/bin/bash
set -euo pipefail

# Usage:
#   bash train_tienkung_mimic_real.sh <exptid> <device> <num_envs> <max_iterations> <motion_yaml> [extra args...]
# Example:
#   bash train_tienkung_mimic_real.sh tk_real_0329 cuda:0 256 10000

EXPTID=${1:-tk_real_$(date +%m%d_%H%M)}
DEVICE=${2:-cuda:0}
NUM_ENVS=${3:-256}
MAX_ITER=${4:-10000}
MOTION_YAML=${5:-/home/qsh/workspace_twist2/TWIST2/legged_gym/motion_data_configs/tienkung_ei_train30.yaml}

if [ "$#" -gt 5 ]; then
  shift 5
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[train_tienkung_mimic_real] WANDB_API_KEY is not set, append --no_wandb."
  EXTRA_ARGS+=("--no_wandb")
fi

cd /home/qsh/workspace_twist2/TWIST2/legged_gym/legged_gym/scripts

LD_LIBRARY_PATH=/home/qsh/miniconda3/envs/twist2/lib:${LD_LIBRARY_PATH:-} \
/home/qsh/miniconda3/envs/twist2/bin/python train.py \
  --task tienkung_2_0_mimic \
  --proj_name tienkung_2_0_mimic \
  --exptid "${EXPTID}" \
  --device "${DEVICE}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITER}" \
  --env.motion.motion_file "${MOTION_YAML}" \
  "${EXTRA_ARGS[@]}"
