#!/bin/bash

# Create logs directory if it doesn't exist to prevent LSF errors
mkdir -p ./logs

# Submit the LSF script to the queue
bsub < RunPhaseMaskNorm.sh

#echo "Job submitted to LSF queue."