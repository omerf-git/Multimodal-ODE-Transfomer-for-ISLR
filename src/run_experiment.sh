#!/bin/bash

# Exit script on error
set -e

# Path to configuration file
CONFIG_FILE="config.sh"

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Source variables from configuration file
source "$CONFIG_FILE"

# Manage NORM_FIRST flag
NORM_FLAG=""
if [ "$NORM_FIRST" = "False" ] || [ "$NORM_FIRST" = "false" ]; then
    NORM_FLAG="--no-norm-first"
fi

# Construct command to run train.py
# Concatenating arguments using variables
COMMAND="python -m train \
    --model $MODEL \
    --dataset $DATASET \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --dropout $DROPOUT \
    --cnn $CNN \
    --embed_size $EMBED_SIZE \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --enc_calculate_num $ENC_CALCULATE_NUM \
    --rk_type $RK_TYPE \
    --encoder_history_type $ENCODER_HISTORY_TYPE \
    $NORM_FLAG"

# Print constructed command
echo "Executed Command:"
echo "$COMMAND"
echo "-----------------------------------------------------"

# Run command
eval $COMMAND