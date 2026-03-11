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
module load conda
module load cuda/12.3

# Check GPU status
hostname
nvidia-smi
#nvidia-smi topo -m

# Run the inference script
conda run -p /rsstu/users/a/agrinba/DeepDesign/deepdesign python ./physics/phase_mask_norm.py