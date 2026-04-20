#!/bin/bash
set -euo pipefail

# Usage:
#   bash train.sh <experiment_id> <device> [num_envs] [max_iterations] [extra args...]
#
# Examples:
#   bash train.sh g1_debug_$(date +%m%d_%H%M) cuda:0 4 20 --debug
#   bash train.sh g1_train_$(date +%m%d_%H%M) cuda:0 512 10000 --no_wandb

cd /home/qsh/workspace_twist2/TWIST2/legged_gym/legged_gym/scripts

robot_name="g1"
exptid=${1:-g1_stu_future_$(date +%m%d_%H%M)}
device=${2:-cuda:0}
num_envs=${3:-}
max_iterations=${4:-}

extra_args=()
if [ $# -ge 5 ]; then
  shift 4
  extra_args=("$@")
fi

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

cmd=(
  python train.py
  --task "${task_name}"
  --proj_name "${proj_name}"
  --exptid "${exptid}"
  --device "${device}"
  --teacher_exptid "None"
)

if [ -n "${num_envs}" ]; then
  cmd+=(--num_envs "${num_envs}")
fi

if [ -n "${max_iterations}" ]; then
  cmd+=(--max_iterations "${max_iterations}")
fi

cmd+=("${extra_args[@]}")

"${cmd[@]}"
