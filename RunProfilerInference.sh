#!/bin/bash
#BSUB -n 2
#BSUB -W 1:00
#BSUB -R "rusage[mem=4GB]"
##BSUB -R "select[hname!='gpu16']"
#BSUB -q short_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -J inference
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
conda run -p /rsstu/users/a/agrinba/DeepDesign/deepdesign python ./RunProfilerInference.py