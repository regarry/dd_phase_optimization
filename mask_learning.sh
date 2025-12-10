#!/bin/bash
#BSUB -n 16
#BSUB -W 96:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -q bme_gpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
##BSUB -R "select[hname!=gpu18]"
#BSUB -J psf_test
#BSUB -o ./logs/.%J
#BSUB -e ./logs/.%J
nvidia-smi

module load conda
module load cuda/12.3
#export CUDA_LAUNCH_BLOCKING=1
#conda activate /rsstu/users/a/agrinba/DeepDesign/deepdesign

# Get the unique config file path from the first argument
CONFIG_FILE="__CONFIG_FILE__"

echo "Job is running with config file: $CONFIG_FILE"
echo "--- Config Contents ---"
cat $CONFIG_FILE
echo "-----------------------"

conda run -p /rsstu/users/a/agrinba/DeepDesign/deepdesign python ./mask_learning.py --config $CONFIG_FILE

# Optional: Clean up the copied config file
echo "Cleaning up $CONFIG_FILE"
rm $CONFIG_FILE