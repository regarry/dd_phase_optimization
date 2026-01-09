#!/bin/bash

# 1. Create a unique file name. Using job ID is hard, so use a timestamp.
TIMESTAMP=$(date +%s)
UNIQUE_CONFIG="config.${TIMESTAMP}.yaml"

# 2. Copy the config *now*
echo "Capturing config.yaml into $UNIQUE_CONFIG"
cp config.yaml $UNIQUE_CONFIG

# 3. Submit the job, passing the new config file as an argument
echo "Submitting job..."
# This will find __CONFIG_FILE__ in train_job.bsub,
# replace it with the value of $UNIQUE_CONFIG,
# and then pipe that new script to bsub
sed "s|__CONFIG_FILE__|$UNIQUE_CONFIG|g" train_job.bsub | bsub

echo "Job submitted. It will use $UNIQUE_CONFIG"