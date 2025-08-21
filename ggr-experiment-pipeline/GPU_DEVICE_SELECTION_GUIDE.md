# GPU Device Selection Guide for KV Cache Tests

## Overview
Both test scripts now support GPU device selection to control which GPU vLLM uses for inference and KV cache testing.

## Quick Usage

### For Tiny Dataset Test:
```bash
# Use default GPU (GPU 0)
python test_tiny_dataset_kv_cache.py

# Use specific GPU (e.g., GPU 2)
python test_tiny_dataset_kv_cache.py --gpu 2

# Use GPU 4
python test_tiny_dataset_kv_cache.py --gpu 4
```

### For Final Comprehensive Test:
```bash
# Use default GPU (GPU 0) 
python final_kv_cache_test.py

# Use specific GPU (e.g., GPU 1)
python final_kv_cache_test.py --gpu 1

# Use GPU 3
python final_kv_cache_test.py --gpu 3
```

## GPU Device Setting Methods

### Method 1: Command Line Arguments (Recommended)
- **Easy**: Just add `--gpu X` where X is your GPU ID
- **Flexible**: Different GPU per test run
- **Clear**: Shows which GPU is being used in output

### Method 2: Environment Variable
```bash
# Set before running test
export CUDA_VISIBLE_DEVICES=2
python test_tiny_dataset_kv_cache.py

# Or inline
CUDA_VISIBLE_DEVICES=1 python final_kv_cache_test.py
```

### Method 3: Direct Code Modification
In the test file, modify the vLLM initialization:
```python
self.llm = LLM(
    model=self.model_name,
    # ... other parameters ...
    device="cuda:2",  # Force GPU 2
)
```

## GPU Selection Examples

### Single GPU System:
```bash
python test_tiny_dataset_kv_cache.py --gpu 0
```

### Multi-GPU System (choose best GPU):
```bash
# Check available GPUs first
nvidia-smi

# Use GPU with most free memory (e.g., GPU 3)
python test_tiny_dataset_kv_cache.py --gpu 3
```

### For Multi-GPU Models:
```python
# In code, modify tensor_parallel_size
self.llm = LLM(
    model="large_model",
    tensor_parallel_size=2,  # Use 2 GPUs
    # vLLM will automatically use GPUs 0,1
)
```

## GPU Verification

Both test scripts will show GPU information:
```
🎯 GPU device set to: GPU 2
   GPU 2: NVIDIA RTX 4090 (24.0 GB)
```

## Troubleshooting

### Issue: "GPU X not available"
**Solution**: Check available GPUs with `nvidia-smi` and use valid GPU ID

### Issue: "CUDA not available" 
**Solution**: Install CUDA-enabled PyTorch and ensure GPU drivers are working

### Issue: Out of memory
**Solutions**:
- Use GPU with more memory: `--gpu X`
- Reduce memory usage: modify `gpu_memory_utilization=0.3`
- Use smaller model: change `model_name`

## Best Practices

1. **Check GPU availability first**:
   ```bash
   nvidia-smi
   ```

2. **Use GPU with most free memory** for testing

3. **Start with conservative memory settings**:
   - `gpu_memory_utilization=0.5` or lower
   - Small `max_model_len=1024`

4. **Monitor GPU usage during test**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Test Output Example
```
🚀 Tiny Dataset KV Cache Stats Validation
Testing real KV cache values with shared prefixes
============================================================
🎯 Using GPU: 2
🎯 GPU device set to: GPU 2
   GPU 2: NVIDIA RTX 4090 (24.0 GB)
🚀 Setting up vLLM for KV cache testing...
✅ vLLM initialized successfully
```

The GPU device setting is now integrated into both test scripts and provides clear feedback about which GPU is being used for the KV cache testing.
