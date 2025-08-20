#!/bin/bash
# Example usage commands for multi-GPU vLLM experiment pipeline

# Multi-GPU Examples for Qwen1.5-7B-Chat model

echo "=== Multi-GPU vLLM Experiment Examples ==="
echo ""

# Example 1: Use 4 GPUs with full context length
echo "1. Using 4 GPUs (4,5,6,7) with high memory utilization:"
echo 'python src/experiment/run_experiment.py dataset.csv query_key \'
echo '    --model "/home/data/so/semantic-operators/ggr-experiment-pipeline/src/model/Qwen/Qwen1.5-7B-Chat" \'
echo '    --gpus "4,5,6,7" \'
echo '    --gpu-memory 0.95 \'
echo '    --batch-size 8'
echo ""

# Example 2: Use 2 GPUs with reduced sequence length
echo "2. Using 2 GPUs (6,7) with reduced max sequence length:"
echo 'python src/experiment/run_experiment.py dataset.csv query_key \'
echo '    --model "/home/data/so/semantic-operators/ggr-experiment-pipeline/src/model/Qwen/Qwen1.5-7B-Chat" \'
echo '    --gpus "6,7" \'
echo '    --max-model-len 8192 \'
echo '    --gpu-memory 0.90 \'
echo '    --batch-size 6'
echo ""

# Example 3: Single GPU with aggressive optimization (backward compatible)
echo "3. Single GPU (7) with memory optimization:"
echo 'python src/experiment/run_experiment.py dataset.csv query_key \'
echo '    --model "/home/data/so/semantic-operators/ggr-experiment-pipeline/src/model/Qwen/Qwen1.5-7B-Chat" \'
echo '    --gpu 7 \'
echo '    --max-model-len 4096 \'
echo '    --gpu-memory 0.95 \'
echo '    --batch-size 4'
echo ""

# Example 4: Check GPU availability
echo "4. Check available GPUs:"
echo 'python src/experiment/run_experiment.py --check-gpu'
echo ""

# Example 5: Validate model before running
echo "5. Validate local model:"
echo 'python src/experiment/run_experiment.py dataset.csv query_key \'
echo '    --model "/home/data/so/semantic-operators/ggr-experiment-pipeline/src/model/Qwen/Qwen1.5-7B-Chat" \'
echo '    --validate-model'
echo ""

# Example 6: List available query templates
echo "6. List all available query templates:"
echo 'python src/experiment/run_experiment.py --list-queries'
echo ""

echo "=== Memory Optimization Guidelines ==="
echo ""
echo "For models requiring >16GB memory (like Qwen1.5-7B-Chat with 32K context):"
echo "- Use --gpus with multiple GPUs (e.g., '4,5,6,7' or '6,7')"
echo "- Increase --gpu-memory to 0.95 for maximum utilization"
echo "- Reduce --max-model-len to 8192 or 4096 to save memory"
echo "- Reduce --batch-size if still running out of memory"
echo ""
echo "Expected memory usage for Qwen1.5-7B-Chat:"
echo "- Model weights: ~7GB (FP16)"
echo "- KV Cache (32K): ~16GB"
echo "- Total: ~23GB → Requires 2-4 GPUs"
echo ""

# Available query types
echo "=== Available Query Types ==="
echo "- Aggregation: agg_movies_sentiment, agg_products_sentiment"
echo "- Multi-invocation: multi_movies_sentiment, multi_products_sentiment"
echo "- Filter: filter_movies_kids, filter_products_sentiment, filter_bird_statistics, etc."
echo "- Projection: proj_movies_summary, proj_products_consistency, etc."
echo "- RAG: rag_fever, rag_squad"
