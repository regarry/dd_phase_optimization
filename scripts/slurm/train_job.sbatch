#!/bin/bash
#SBATCH --job-name=train_dd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --output=./logs/train_dd.%j.out
#SBATCH --error=./logs/train_dd.%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -n "${DD_MODULES:-}" ]]; then
    read -r -a DD_MODULE_NAMES <<< "$DD_MODULES"
    module load "${DD_MODULE_NAMES[@]}"
fi
DD_PYTHON="${DD_PYTHON:-python}"
source ./scripts/slurm/worker_runtime.sh
"$DD_PYTHON" -m utils.worker_preflight
CONFIG_FILE=__CONFIG_FILE__
"$DD_PYTHON" -u -m training.training --config "$CONFIG_FILE" \
    --worker-timeout 60 --progress-every 32 --wall-seconds __WALL_SECONDS__ \
    --stop-margin-seconds 900
