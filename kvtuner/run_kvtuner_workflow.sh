#!/bin/bash
# Example workflow script for finding KV config and running inference

# Define variables
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
SCHEME="kivi"  # pertoken or kivi
TARGET_BITS=4.0
DATASET_PATH="path/to/dataset.jsonl"
GPU_IDS="0,1"  # Specify which GPUs to use

# Create directories
mkdir -p ./configs ./results

echo "Step 1: Finding optimal KV configuration for $MODEL_NAME with $SCHEME scheme"
python find_optimal_kv_config.py \
    --model_name "$MODEL_NAME" \
    --scheme "$SCHEME" \
    --target_bits "$TARGET_BITS" \
    --output_dir ./configs \
    --search_method optuna \
    --n_trials 30

# Get the most recent config file
CONFIG_FILE=$(ls -t ./configs/${MODEL_NAME##*/}_${SCHEME}_${TARGET_BITS%.*}_bits_*.yaml 2>/dev/null | head -1)

if [ -z "$CONFIG_FILE" ]; then
    echo "No config file found. Using uniform quantization instead."
    
    echo "Step 2: Running inference with uniform quantization"
    python run_kvtuner_inference.py \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET_PATH" \
        --scheme "$SCHEME" \
        --key_bits 4 \
        --value_bits 4 \
        --gpu_ids "$GPU_IDS" \
        --output_dir ./results
else
    echo "Found config file: $CONFIG_FILE"
    
    echo "Step 2: Running inference with the found configuration"
    python run_kvtuner_inference.py \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET_PATH" \
        --kv_config "$CONFIG_FILE" \
        --scheme "$SCHEME" \
        --gpu_ids "$GPU_IDS" \
        --output_dir ./results
fi

echo "Done! Check the results directory for outputs."
