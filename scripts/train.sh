#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0


cd legged_gym/legged_gym/scripts

robot_name="tienkung"
exptid=$1
device=$2

task_name="${robot_name}_mimic"
proj_name="${robot_name}_mimic"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                # --resume \
                # --debug \
