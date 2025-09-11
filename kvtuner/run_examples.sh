#!/bin/bash

# KVTuner LLM Inference Usage Examples
# This script shows how to use the llm_inference.py script with KVTuner

echo "KVTuner LLM Inference - Usage Examples"
echo "======================================"

# Configuration
MODEL_PATH="/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct"
DATASET_PATH="/Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/datasets/sample_data.csv"
KVTUNER_DIR="/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  KVTuner: $KVTUNER_DIR"
echo ""

# Function to run example
run_example() {
    local name="$1"
    local cmd="$2"
    
    echo "Example: $name"
    echo "Command: $cmd"
    echo "Press Enter to run this example, or Ctrl+C to skip..."
    read
    
    echo "Running..."
    eval "$cmd"
    echo ""
    echo "Example completed!"
    echo "----------------------------------------"
    echo ""
}

# Example 1: Basic pertoken inference
run_example "Basic Pertoken Inference" \
    "python llm_inference.py \\
        --dataset \"$DATASET_PATH\" \\
        --model \"$MODEL_PATH\" \\
        --kvtuner_scheme pertoken \\
        --include_columns text \\
        --max_rows 5 \\
        --max_new_tokens 100"

# Example 2: KiVi scheme inference
run_example "KiVi Scheme Inference" \
    "python llm_inference.py \\
        --dataset \"$DATASET_PATH\" \\
        --model \"$MODEL_PATH\" \\
        --kvtuner_scheme kivi \\
        --include_columns text \\
        --max_rows 3 \\
        --max_new_tokens 80"

# Example 3: Custom prompt template
run_example "Custom Prompt Template" \
    "python llm_inference.py \\
        --dataset \"$DATASET_PATH\" \\
        --model \"$MODEL_PATH\" \\
        --kvtuner_scheme pertoken \\
        --include_columns text \\
        --prompt_template \"Summarize this text in one sentence: {text}\" \\
        --max_rows 3 \\
        --max_new_tokens 50"

# Example 4: Using specific GPU
run_example "Specific GPU Usage" \
    "python llm_inference.py \\
        --dataset \"$DATASET_PATH\" \\
        --model \"$MODEL_PATH\" \\
        --kvtuner_scheme pertoken \\
        --include_columns text \\
        --gpu_ids \"0\" \\
        --max_rows 2 \\
        --max_new_tokens 60"

echo "All examples completed!"
echo ""
echo "Additional options you can use:"
echo "  --kvtuner_dir: Custom path to KVTuner directory"
echo "  --tokenizer: Custom tokenizer path"
echo "  --output_prefix: Custom prefix for output files"
echo "  --max_model_len: Maximum model context length"
echo ""
echo "Results are saved in the inference_results/ directory with timestamps."
