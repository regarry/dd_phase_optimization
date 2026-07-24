#!/bin/bash
#Prerequisites
#For this automation to work seamlessly, open your base config.yaml file and replace the hardcoded values for these parameters with placeholder tokens:
#Set initial_phase_mask to __PHASE_MASK__
#Set num_classes to __NUM_CLASSES__
#Set bce_weight to __BCE_WEIGHT__
#Set dice_weight to __DICE_WEIGHT__

# 1. Define folder, base configs, and arrays
TEMP_DIR="temp_configs"
BASE_CONFIG="config.yaml"
mkdir -p "$TEMP_DIR"
mkdir -p ./logs

# Define the parameter ranges
PHASE_MASKS=("random" "empty" "axicon" "lens") #("empty" "axicon" "lens" "random")
NUM_CLASSES_ARR=(1 3) #(1 3)

# Define the loss weight ratios to test (e.g., 1:0, 1:1, 0.5:1, 0:1)
CE_WEIGHTS=(1.0 0.0 1.0)  #(1.0 0.1 1.0)
DICE_WEIGHTS=(0.0 1.0 1.0)  #(1.0 1.0 0.1)
MAE_WEIGHTS=(0.0 0.0 0.0)
MSE_WEIGHTS=(0.0 0.0 0.0)


# 2. Loop through all permutations
for mask in "${PHASE_MASKS[@]}"; do
    for classes in "${NUM_CLASSES_ARR[@]}"; do
        # Loop through the indices of the weight arrays
        for i in "${!CE_WEIGHTS[@]}"; do
            ce_w=${CE_WEIGHTS[$i]}
            dice_w=${DICE_WEIGHTS[$i]}
            mae_w=${MAE_WEIGHTS[$i]}
            mse_w=${MSE_WEIGHTS[$i]}
            
            # Create a unique timestamp + descriptor for each permutation
            TIMESTAMP=$(date +%s_%N) 
            UNIQUE_CONFIG="${TEMP_DIR}/config.${mask}_c${classes}_ce${ce_w}_dice${dice_w}_mae${mae_w}_mse${mse_w}.${TIMESTAMP}.yaml"
            
            echo "Generating config for Mask: $mask, Classes: $classes, CE Weight: $ce_w, Dice Weight: $dice_w, MAE Weight: $mae_w, MSE Weight: $mse_w"
            
            # 3. Copy the base config and inject all specific parameters
            sed -e "s|__PHASE_MASK__|$mask|g" \
                -e "s|__NUM_CLASSES__|$classes|g" \
                -e "s|__CE_WEIGHT__|$ce_w|g" \
                -e "s|__MAE_WEIGHT__|$mae_w|g" \
                -e "s|__MSE_WEIGHT__|$mse_w|g" \
                -e "s|__DICE_WEIGHT__|$dice_w|g" "$BASE_CONFIG" > "$UNIQUE_CONFIG"
            
            # 4. Submit the job using your existing LSF template setup
            echo "Submitting job with $UNIQUE_CONFIG..."
            sed "s|__CONFIG_FILE__|$UNIQUE_CONFIG|g" ./scripts/lsf/train_batch_job.bsub | bsub
            
            # Small sleep just to avoid any potential race conditions with fast file writing
            sleep 0.5
            echo "--------------------------------------------"
        done
    done
done

echo "All job permutations have been queued to LSF!"