#!/bin/bash

# --- Configuration ---
GPU_ID=0
DOUBLE_CHECK=true # Set to true to enable double-checking, false to disable
DOUBLE_CHECK_WAIT_PERIOD="1m" # Wait period for double-checking (e.g., 10s, 5m, 1h)
CHECK_INTERVAL="5m"          # Interval between checks when GPU is busy (e.g., 30s, 2m, 1h)
PYTHON_SCRIPT="train.py --outdir=output_runs_95 --cfg=stylegan2 --data=bffhq_256_95_balanced --gpus=1 --batch=32 --gamma=8 --cond=1 --kimg=3500 --mirror=1 --aug=ada --gpu $GPU_ID" # Path to the Python script to launch
#PYTHON_SCRIPT="pretraining/ddpm/ddpm_main.py -g 0 --resume"
#PYTHON_SCRIPT="distillation/distillation_main.py -g 0"

# --- End Configuration ---

while true; do
    # Check if GPU has any running processes
    PROCESSES_ON_GPU=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader,nounits | grep -v '^$' | wc -l)
    echo "$(date): Number of processes running on GPU $GPU_ID: $PROCESSES_ON_GPU"

    if [ "$PROCESSES_ON_GPU" -eq 0 ]; then
        echo "$(date): GPU $GPU_ID is free."

        if [ "$DOUBLE_CHECK" = true ]; then
            # Double-check logic enabled
            echo "$(date): Double-checking enabled. Waiting ${DOUBLE_CHECK_WAIT_PERIOD} to recheck."
            sleep ${DOUBLE_CHECK_WAIT_PERIOD} # Wait period before rechecking

            # Recheck if GPU is still free
            PROCESSES_ON_GPU_RECHECK=$(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader,nounits | grep -v '^$' | wc -l)

            if [ "$PROCESSES_ON_GPU_RECHECK" -eq 0 ]; then
                echo "$(date): GPU $GPU_ID is still free after waiting. Launching the Python script: ${PYTHON_SCRIPT}"
                python ${PYTHON_SCRIPT}
                exit 0
            else
                echo "$(date): GPU $GPU_ID became busy during the wait period (${DOUBLE_CHECK_WAIT_PERIOD})."
            fi
        else
            # Double-check logic disabled, launch immediately
            echo "$(date): Double-checking disabled. Launching the Python script immediately: ${PYTHON_SCRIPT}"
            python ${PYTHON_SCRIPT}
            exit 0
        fi
    else
        echo "$(date): GPU $GPU_ID is currently in use."
    fi
    echo "$(date): Waiting ${CHECK_INTERVAL} before the next check cycle."
    sleep ${CHECK_INTERVAL} # Wait before the next check cycle
done