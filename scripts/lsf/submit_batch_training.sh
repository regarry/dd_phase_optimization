#!/bin/bash
#Prerequisites
#For this automation to work seamlessly, open your base config.yaml file and replace the hardcoded values for those two parameters with placeholder tokens:
#Set initial_phase_mask to __PHASE_MASK__
#Set num_classes to __NUM_CLASSES__
# 1. Define folder, base configs, and arrays
TEMP_DIR="temp_configs"
BASE_CONFIG="config.yaml"
mkdir -p "$TEMP_DIR"
mkdir -p ./logs

# Define the parameter ranges
PHASE_MASKS=("empty" "axicon" "lens")
NUM_CLASSES_ARR=(1 3)

# 2. Loop through all permutations
for mask in "${PHASE_MASKS[@]}"; do
    for classes in "${NUM_CLASSES_ARR[@]}"; do
        
        # Create a unique timestamp + descriptor for each permutation
        TIMESTAMP=$(date +%s_%N) # Added %N (nanoseconds) to ensure uniqueness in fast loops
        UNIQUE_CONFIG="${TEMP_DIR}/config.${mask}_c${classes}.${TIMESTAMP}.yaml"
        
        echo "Generating config for Mask: $mask, Classes: $classes"
        
        # 3. Copy the base config and inject the specific parameters
        # We chain sed commands to replace both placeholders in one go
        sed -e "s|__PHASE_MASK__|$mask|g" \
            -e "s|__NUM_CLASSES__|$classes|g" "$BASE_CONFIG" > "$UNIQUE_CONFIG"
        
        # 4. Submit the job using your existing LSF template setup
        echo "Submitting job with $UNIQUE_CONFIG..."
        sed "s|__CONFIG_FILE__|$UNIQUE_CONFIG|g" ./scripts/lsf/train_batch_job.bsub | bsub
        
        # Small sleep just to avoid any potential race conditions with fast file writing
        sleep 0.5
        echo "--------------------------------------------"
    done
done

echo "All 6 job permutations have been queued to LSF!"