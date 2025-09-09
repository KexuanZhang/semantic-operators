#!/bin/bash
# Example script to run the experiment using specific GPUs (6,7)

# Check if dataset path is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset_path> [max_rows]"
  echo "Example: $0 path/to/dataset.csv 100"
  exit 1
fi

DATASET_PATH=$1
MAX_ROWS=${2:-50}  # Default to 50 rows if not specified

# Use GPUs 6,7 and TinyLlama model
python reorder_inference_experiment.py \
  --dataset $DATASET_PATH \
  --gpu_ids "6,7" \
  --tp_size 2 \
  --max_rows $MAX_ROWS \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --reorder

# For a larger model, uncomment this:
# python reorder_inference_experiment.py \
#   --dataset $DATASET_PATH \
#   --gpu_ids "6,7" \
#   --tp_size 2 \
#   --max_rows $MAX_ROWS \
#   --model "meta-llama/Llama-2-7b-chat-hf" \
#   --reorder
