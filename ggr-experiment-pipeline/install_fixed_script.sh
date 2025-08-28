#!/bin/bash
# Script to install the fixed run_experiment_enhanced.py on the server

# Set the source and destination paths
SRC_DIR="/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline"
DEST_DIR="/home/data/so/semantic-operators/ggr-experiment-pipeline"

# Check if source files exist
if [ ! -f "$SRC_DIR/run_experiment_fixed.py" ]; then
  echo "Error: Fixed script not found at $SRC_DIR/run_experiment_fixed.py"
  exit 1
fi

# Copy files to server
echo "Copying files to server..."
scp "$SRC_DIR/run_experiment_fixed.py" "data@gpu103:$DEST_DIR/"

# Make the script executable on the server
echo "Setting execution permissions..."
ssh data@gpu103 "chmod +x $DEST_DIR/run_experiment_fixed.py"

echo "Installation complete!"
echo "To run the experiment, use the following command on the server:"
echo "cd $DEST_DIR && python run_experiment_fixed.py /home/data/so/semantic-operators/ggr-experiment-pipeline/src/datasets/final_movies_dataset_sample500_random_20250825_193609.csv agg_movies_sentiment --gpus 4,5,6,7 --tensor-parallel-size 4 --model /home/data/so/semantic-operators/ggr-experiment-pipeline/src/model/Qwen/Qwen1.5-7B-Chat --gpu-memory 0.7"
