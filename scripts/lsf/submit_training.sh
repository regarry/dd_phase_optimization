#!/bin/bash

# 1. Define folder and timestamp
TEMP_DIR="temp_configs"
TIMESTAMP=$(date +%s)
UNIQUE_CONFIG="${TEMP_DIR}/config.${TIMESTAMP}.yaml"

# 2. Create the directory if it doesn't exist
mkdir -p "$TEMP_DIR"
mkdir -p ./logs

# 3. Copy the config *now*
echo "Capturing config.yaml into $UNIQUE_CONFIG"
cp config.yaml "$UNIQUE_CONFIG"

# 4. Submit the job, passing the new config file as an argument
echo "Submitting job..."
# The '|' delimiter in sed handles the slashes in the file path correctly
sed "s|__CONFIG_FILE__|$UNIQUE_CONFIG|g" ./scripts/lsf/train_job.bsub | bsub

echo "Job submitted. It will use $UNIQUE_CONFIG"