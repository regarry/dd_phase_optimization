#!/bin/bash
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -q short_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -J PhaseNorm
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
#nvidia-smi topo -m

# Run the inference script
"${DD_PYTHON:-python}" -m physics.phase_mask_norm "$@"
