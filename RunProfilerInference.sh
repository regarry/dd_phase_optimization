#!/bin/bash
#BSUB -n 4
#BSUB -W 1:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[hname!='gpu18']"
#BSUB -q short_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
##BSUB -R "select[a100]"
#BSUB -J inference
#BSUB -o ./logs/.%J
#BSUB -e ./logs/.%J

# Load environment
module load conda
module load cuda/12.3

# Check GPU status
hostname
nvidia-smi

# Run the inference script using variables passed from the environment
/rsstu/users/a/agrinba/DeepDesign/deepdesign/bin/python ./RunProfilerInference.py "$TRAINING_FOLDER" "$EPOCH"