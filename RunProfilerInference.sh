#!/bin/bash
#BSUB -n 4
#BSUB -W 1:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -q bme_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
##BSUB -R "select[a100]"
#BSUB -J inference
#BSUB -o ./logs/.%J
#BSUB -e ./logs/.%J

# Load environment
if [[ -n "${DD_MODULES:-}" ]]; then
    read -r -a DD_MODULE_NAMES <<< "$DD_MODULES"
    module load "${DD_MODULE_NAMES[@]}"
fi

# Check GPU status
hostname
nvidia-smi

# Run the inference script using variables passed from the environment
"${DD_PYTHON:-python}" ./RunProfilerInference.py "$TRAINING_FOLDER" "$EPOCH"
