# Multi-GPU Support Implementation for LLM Experiment Pipeline

## Problem Solved

**Original Error:**
```
ValueError: To serve at least one request with the models's max seq len (32768), 
16.00 GiB KV cache is needed, which is larger than the available KV cache memory (4.85 GiB). 
Based on the available memory, the estimated maximum model length is 9936.
```

The Qwen1.5-7B-Chat model with 32K sequence length requires 16GB+ of KV cache memory, but only ~5GB was available on a single GPU.

## Solution: Multi-GPU Support + Memory Optimization

### Key Changes Made

#### 1. **Multi-GPU Device Management**
- **New function:** `set_gpu_devices(gpu_ids: List[int])` - supports multiple GPUs
- **Enhanced:** `SimpleLLMExperiment.__init__()` - accepts `gpu_ids` parameter
- **Backward compatible:** Still supports single `gpu_id` parameter

#### 2. **Memory Estimation & Optimization**
- **New function:** `calculate_memory_requirements()` - estimates model + KV cache memory needs
- **Auto-tuning:** Automatically adjusts `max_model_len` if memory is insufficient
- **Better error messages:** Provides specific memory optimization suggestions

#### 3. **Enhanced vLLM Configuration**
- **Tensor parallelism:** Automatically enables `tensor_parallel_size` for multi-GPU
- **Memory management:** Better `gpu_memory_utilization` handling
- **Sequence length control:** `max_model_len` parameter to reduce memory usage

#### 4. **Updated Command-Line Interface**
- **New argument:** `--gpus "4,5,6,7"` for multi-GPU specification
- **New argument:** `--max-model-len 8192` for memory optimization  
- **Enhanced:** `--gpu-memory 0.95` for higher memory utilization
- **Backward compatible:** `--gpu 7` still works for single GPU

## Usage Examples

### Multi-GPU Setup (Recommended for Large Models)
```bash
# Use 4 GPUs for Qwen1.5-7B-Chat with full 32K context
python src/experiment/run_experiment.py dataset.csv query_key \
    --model "/path/to/Qwen1.5-7B-Chat" \
    --gpus "4,5,6,7" \
    --gpu-memory 0.95

# Use 2 GPUs with reduced sequence length for memory efficiency
python src/experiment/run_experiment.py dataset.csv query_key \
    --model "/path/to/Qwen1.5-7B-Chat" \
    --gpus "6,7" \
    --max-model-len 8192 \
    --gpu-memory 0.90
```

### Memory-Optimized Single GPU
```bash
# Single GPU with aggressive memory optimization
python src/experiment/run_experiment.py dataset.csv query_key \
    --model "/path/to/Qwen1.5-7B-Chat" \
    --gpu 7 \
    --max-model-len 4096 \
    --gpu-memory 0.95 \
    --batch-size 4
```

### Model Validation
```bash
# Check if local model is valid before running
python src/experiment/run_experiment.py dataset.csv query_key \
    --model "/path/to/Qwen1.5-7B-Chat" \
    --validate-model
```

## Memory Optimization Strategy

The implementation uses a multi-layered approach:

1. **Memory Estimation:** Calculate required memory based on model config
2. **Multi-GPU Distribution:** Spread model across multiple GPUs using tensor parallelism
3. **Sequence Length Adjustment:** Reduce `max_model_len` to fit available memory
4. **Dynamic Configuration:** Auto-adjust parameters based on available resources

### Memory Calculation Formula

```
Total Memory = Model Weights + KV Cache
Model Memory ≈ Parameters × 2 bytes (FP16)
KV Cache = 2 × Layers × Hidden Size × Max Seq Length × 2 bytes
```

For Qwen1.5-7B-Chat:
- Model: ~7GB (FP16 weights)
- KV Cache (32K): ~16GB 
- **Total: ~23GB** → Requires 2-4 GPUs depending on GPU memory

## Error Handling & Suggestions

The enhanced error handling provides specific suggestions:

```
Memory optimization suggestions:
1. Try using more GPUs: --gpus 4,5,6,7
2. Increase gpu_memory_utilization: --gpu-memory 0.95  
3. Reduce max_model_len: --max-model-len 8192
4. Reduce batch size: --batch-size 4
Total GPU memory available: 101.6GB across 4 GPUs
```

## Backward Compatibility

All existing single-GPU commands continue to work:
- `--gpu 7` (single GPU)
- All existing parameters unchanged
- Same output format and monitoring

## Performance Benefits

1. **Memory Distribution:** 16GB KV cache spread across multiple GPUs
2. **Parallel Processing:** Tensor parallelism for faster inference
3. **Higher Utilization:** Can use 90-95% of GPU memory safely
4. **Adaptive Tuning:** Automatically optimizes based on available resources

## Testing

Run the test script to validate functionality:
```bash
python test_multi_gpu.py
```

This implementation should resolve the KV cache memory limitation and enable successful loading of large language models like Qwen1.5-7B-Chat with their full context length.
