#!/bin/bash
# KVTuner Inference Runner for Qwen2.5-3B-Instruct
# 
# Edit the variables below to match your setup:

# === CONFIGURATION ===
MODEL_PATH="/path/to/your/qwen2.5-3b-instruct"  # Change this to your model path
DATASET_PATH="/path/to/your/dataset.csv"         # Change this to your dataset path
GPU_IDS="0,1"                                   # Change this to your preferred GPUs
SCHEME="pertoken"                               # Change to "kivi" if preferred
TEXT_COLUMN="text"                              # Change to match your dataset column
MAX_ROWS=""                                     # Leave empty to process all rows, or set a number

# === EXECUTION ===
echo "Running KVTuner inference with Qwen2.5-3B-Instruct..."
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Scheme: $SCHEME"
echo "GPUs: $GPU_IDS"
echo ""

# Build the command
CMD="python kvtuner_inference.py --model_path \"$MODEL_PATH\" --dataset \"$DATASET_PATH\" --scheme $SCHEME --gpu_ids \"$GPU_IDS\" --text_column $TEXT_COLUMN"

# Add max_rows if specified
if [ -n "$MAX_ROWS" ]; then
    CMD="$CMD --max_rows $MAX_ROWS"
fi

# Add debug flag if desired (uncomment the next line for debug output)
# CMD="$CMD --debug"

echo "Executing: $CMD"
echo ""

# Run the command
eval $CMD
