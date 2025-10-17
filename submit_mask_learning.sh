#!/bin/bash
#BSUB -n 1
#BSUB -W 30:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -q bme_gpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o ./logs/.%J
#BSUB -e ./logs/.%J
nvidia-smi

module load conda
module load cuda/12.3

#conda activate /rsstu/users/a/agrinba/DeepDesign/deepdesign
conda run -p /rsstu/users/a/agrinba/DeepDesign/deepdesign python ./mask_learning.py
