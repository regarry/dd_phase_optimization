#!/bin/bash
#SBATCH --job-name=train_dd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --qos=short_gpu
#SBATCH --partition=gpu_partners
#SBATCH --gres=gpu:l40s:3
#SBATCH --output=./logs/train_dd.%j.out
#SBATCH --error=./logs/train_dd.%j.err

set -euo pipefail

# Run from the original submission directory
cd "$SLURM_SUBMIT_DIR"

# GPU and node information
hostname
nvidia-smi

# Load environment
module load conda
module load cuda/12.3

# Optional debugging
# export CUDA_LAUNCH_BLOCKING=1

# Get the unique config file from the submission script
CONFIG_FILE="__CONFIG_FILE__"

echo "Job is running with config file: $CONFIG_FILE"
echo "--- Config Contents ---"
cat "$CONFIG_FILE"
echo "-----------------------"

# Clean up config on exit, including failed jobs
trap 'echo "Cleaning up $CONFIG_FILE"; rm -f -- "$CONFIG_FILE"' EXIT

# Run training
conda run -p /rsstu/users/a/agrinba/DeepDesign/deepdesign \
    python ./training/training.py --config "$CONFIG_FILE"
