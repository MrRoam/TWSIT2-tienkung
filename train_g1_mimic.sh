#!/bin/bash
set -euo pipefail

# Usage:
#   bash train_g1_mimic.sh [exptid] [device] [num_envs] [max_iterations] [extra_args...]
# Example:
#   bash train_g1_mimic.sh g1_smoke_$(date +%m%d_%H%M) cuda:0 128 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/legged_gym/legged_gym/scripts"

EXPTID="${1:-g1_mimic_$(date +%m%d_%H%M)}"
DEVICE="${2:-cuda:0}"
NUM_ENVS="${3:-128}"
MAX_ITERS="${4:-2}"

python train.py \
  --task g1_mimic \
  --proj_name g1_mimic \
  --exptid "${EXPTID}" \
  --device "${DEVICE}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  "${@:5}"
