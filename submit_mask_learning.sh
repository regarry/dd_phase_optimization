#!/bin/bash
#BSUB -n 8
#BSUB -W 96:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -q bme_gpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -R "select[hname!=gpu18]"
#BSUB -o ./logs/.%J
#BSUB -e ./logs/.%J
nvidia-smi

module load conda
module load cuda/12.3
#export CUDA_LAUNCH_BLOCKING=1
#conda activate /rsstu/users/a/agrinba/DeepDesign/deepdesign
conda run -p /rsstu/users/a/agrinba/DeepDesign/deepdesign python ./mask_learning.py
