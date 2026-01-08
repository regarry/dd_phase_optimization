#!/bin/bash
#BSUB -n 8
#BSUB -W 4:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -q bme_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -J psf_inference
#BSUB -o ./logs/.%J
#BSUB -e ./logs/.%J

# Load environment
module load conda
module load cuda/12.3

# Check GPU status
nvidia-smi
#nvidia-smi topo -m

# Run the inference script
conda run -p /rsstu/users/a/agrinba/DeepDesign/deepdesign python ./RunProfilerInference.py