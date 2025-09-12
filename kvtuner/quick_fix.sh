#!/bin/bash
# Quick fix script for KVTuner multi-GPU issues

echo "KVTuner Multi-GPU Quick Fix"
echo "=========================="

# Check GPU availability
echo "1. Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

# Get script directory
SCRIPT_DIR="/home/data/so2/semantic-operators/kvtuner"

echo ""
echo "2. Available fix strategies:"
echo ""

echo "Strategy A: Single GPU + KVTuner (Recommended)"
echo "-----------------------------------------------"
echo "python ${SCRIPT_DIR}/llm_inference.py \\"
echo "  --dataset your_dataset.csv \\"
echo "  --model /home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct \\"
echo "  --cache_mode kvtuner \\"
echo "  --kvtuner_scheme pertoken \\"
echo "  --gpu_memory_utilization 0.6 \\"
echo "  --max_new_tokens 200"
echo ""

echo "Strategy B: Dual GPU + Basic Cache (High Performance)"
echo "------------------------------------------------------"
echo "python ${SCRIPT_DIR}/llm_inference.py \\"
echo "  --dataset your_dataset.csv \\"
echo "  --model /home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct \\"
echo "  --cache_mode basic \\"
echo "  --gpu_ids 0,1 \\"
echo "  --gpu_memory_utilization 0.6 \\"
echo "  --max_new_tokens 200"
echo ""

echo "Strategy C: Test compatibility first"
echo "------------------------------------"
echo "python ${SCRIPT_DIR}/test_multi_gpu_compatibility.py"
echo ""

echo "3. Root cause of your error:"
echo "   - KVTuner quantization + tensor_parallel_size=2 is unstable"
echo "   - Engine core initialization fails with multi-GPU KVTuner"
echo "   - The fixed script now includes automatic fallbacks"
echo ""

echo "4. Choose a strategy above or run the compatibility test first."
