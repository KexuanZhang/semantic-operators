# LLM Inference with Dual Cache Support

This directory contains scripts for running LLM inference with configurable cache modes, supporting both KVTuner's quantized cache and vLLM's default cache.

## Features

### 🚀 Dual Cache Modes
- **KVTuner Cache**: Mixed precision quantized cache (2-8 bits per layer) for ~4.6x memory reduction
- **Basic Cache**: Standard vLLM cache implementation for maximum compatibility

### 🔄 Automatic Fallback
- Automatically falls back to basic cache if KVTuner configs are not available
- Ensures inference always works regardless of setup

### ⚙️ Configurable Memory Management
- Adjustable GPU memory utilization (10% to 100%)
- Multi-GPU support with tensor parallelism
- Conservative default settings to avoid OOM errors

## Files

- `llm_inference.py` - Main inference script with dual cache support
- `dual_cache_example.py` - Example script demonstrating both cache modes
- `test_dataset.csv` - Sample dataset for testing
- `complete_integration_test.py` - Comprehensive integration validation

## Quick Start

### Basic Usage

```bash
# Using basic vLLM cache (always works)
python llm_inference.py \
    --dataset test_dataset.csv \
    --model microsoft/DialoGPT-small \
    --cache_mode basic

# Using KVTuner quantized cache (requires preset configs)
python llm_inference.py \
    --dataset test_dataset.csv \
    --model meta-llama/Llama-2-7b-chat-hf \
    --cache_mode kvtuner \
    --kvtuner_scheme pertoken
```

### Advanced Configuration

```bash
# Custom memory settings and GPU selection
python llm_inference.py \
    --dataset test_dataset.csv \
    --model meta-llama/Llama-2-7b-chat-hf \
    --cache_mode kvtuner \
    --kvtuner_scheme kivi \
    --gpu_memory_utilization 0.6 \
    --gpu_ids "0,1" \
    --max_rows 100
```

## Command Line Options

### Required Parameters
- `--dataset PATH` - Path to CSV dataset file
- `--model NAME` - HuggingFace model name or local path

### Cache Configuration
- `--cache_mode {kvtuner,basic}` - Cache mode (default: kvtuner)
- `--kvtuner_scheme {pertoken,kivi}` - KVTuner quantization scheme (default: pertoken)
- `--kvtuner_dir PATH` - Path to KVTuner directory (default: /home/data/so2/KVTuner)

### Memory and GPU Settings
- `--gpu_memory_utilization FLOAT` - GPU memory fraction (0.1-1.0, default: 0.75)
- `--gpu_ids "0,1,..."` - Specific GPU IDs to use
- `--max_model_len INT` - Maximum model context length

### Inference Settings
- `--prompt_template STR` - Prompt template with {column_name} placeholders
- `--include_columns COL1 COL2` - Dataset columns to include in prompts
- `--max_new_tokens INT` - Maximum tokens to generate (default: 200)
- `--max_rows INT` - Limit dataset size for testing

### Output Settings
- `--output_prefix STR` - Prefix for result files (default: dataset name)

## Cache Mode Details

### KVTuner Cache (`--cache_mode kvtuner`)

**Benefits:**
- ~4.6x memory reduction compared to default cache
- Mixed precision quantization (2-8 bits per layer)
- Minimal quality degradation
- Supports both PerToken and KIVI schemes

**Requirements:**
- KVTuner installation at `/home/data/so2/KVTuner`
- Calibration preset files for the specific model
- Available presets: Meta-Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, etc.

**Automatic Fallback:**
If KVTuner configs are not found, automatically switches to basic cache mode.

### Basic Cache (`--cache_mode basic`)

**Benefits:**
- Maximum compatibility with all models
- No additional dependencies
- Standard vLLM performance characteristics

**Use Cases:**
- Models without KVTuner presets
- Development and testing
- Baseline performance comparison

## Memory Management

### Conservative Defaults
- GPU memory utilization: 75% (instead of vLLM's default 90%)
- CUDA graphs disabled (`enforce_eager=True`) to save memory
- Automatic GPU memory clearing before model loading

### Customization
```bash
# Use only 60% of GPU memory
--gpu_memory_utilization 0.6

# Use specific GPUs
--gpu_ids "6,7"

# Multi-GPU with tensor parallelism
--gpu_ids "0,1,2,3"  # Automatically enables tensor_parallel_size=4
```

## Output Files

Results are saved to `inference_results/YYYYMMDD_HHMMSS/` with cache-specific naming:

- `{prefix}_basic_*.json/csv` - Basic cache results
- `{prefix}_kvtuner_*.json/csv` - KVTuner cache results
- `{prefix}_{cache_mode}_summary.txt` - Human-readable summary
- `{prefix}_{cache_mode}_stats.json` - Detailed statistics

## Example Workflows

### 1. Model Compatibility Testing
```bash
# Test with basic cache first
python llm_inference.py --dataset test_dataset.csv --model new-model --cache_mode basic --max_rows 5

# Try with KVTuner if supported
python llm_inference.py --dataset test_dataset.csv --model new-model --cache_mode kvtuner --max_rows 5
```

### 2. Memory Optimization
```bash
# Start conservative
python llm_inference.py --dataset large_dataset.csv --model large-model --gpu_memory_utilization 0.6

# Gradually increase if no OOM errors
python llm_inference.py --dataset large_dataset.csv --model large-model --gpu_memory_utilization 0.8
```

### 3. Performance Comparison
```bash
# Baseline with basic cache
python llm_inference.py --dataset benchmark.csv --model test-model --cache_mode basic --output_prefix baseline

# KVTuner comparison
python llm_inference.py --dataset benchmark.csv --model test-model --cache_mode kvtuner --output_prefix kvtuner
```

## Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce `--gpu_memory_utilization` to 0.6 or lower
2. **KVTuner Not Found**: Use `--cache_mode basic` or install KVTuner
3. **Model Not Supported**: Check if calibration presets exist for your model
4. **Slow Inference**: Try different GPU memory settings or fewer GPUs

### Debug Mode
```bash
# Add verbose output and limit dataset for debugging
python llm_inference.py --dataset debug.csv --model test-model --max_rows 1 --gpu_memory_utilization 0.5
```

## Integration with vLLM

This script integrates seamlessly with the modified vLLM that includes KVTuner support:

- KVTuner quantization method registered in vLLM
- Automatic configuration loading from YAML presets
- Fallback mechanisms for unsupported models
- Unified API for both cache modes

## Development

For development and testing:

```bash
# Run example demonstrations
python dual_cache_example.py

# Show usage examples
python dual_cache_example.py --examples

# Run integration tests
python complete_integration_test.py
```

## Performance Notes

- KVTuner cache typically provides 4.6x memory reduction
- Inference speed may vary depending on quantization scheme
- PerToken scheme generally faster than KIVI
- Multi-GPU setups benefit from tensor parallelism
- Conservative memory settings prevent OOM at cost of memory efficiency

## Model Support

### Tested Models
- Meta-Llama-3.1-8B-Instruct (KVTuner presets available)
- Mistral-7B-Instruct-v0.3 (KVTuner presets available)
- microsoft/DialoGPT-small (Basic cache only)

### Adding New Models
To add KVTuner support for new models:
1. Generate calibration presets using KVTuner tools
2. Place YAML files in `/home/data/so2/KVTuner/calibration_presets/`
3. Follow naming convention: `{model_name}_{scheme}_KVTuner{version}_{variant}.yaml`

## Recent Fixes

### KVTuner Integration Fix (September 2025)
- **Issue**: `TypeError: KVTunerConfig.get_config_filenames() missing 1 required positional argument: 'self'`
- **Fix**: Changed `get_config_filenames()` from instance method to class method in `kvtuner.py`
- **Status**: Fixed in commit `0a0b97d32` and pushed to `kvt` branch
