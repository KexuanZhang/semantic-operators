# vLLM Server Experiment with GPU Support

This document explains how to use the vLLM server experiment script with GPU acceleration for sentiment analysis tasks.

## Overview

The `server_exp.py` script hosts a vLLM server with a language model and processes datasets for sentiment analysis experiments. It collects comprehensive metrics including KV cache usage, inference times, and throughput statistics with full GPU support.

## Features

- ✅ **GPU Acceleration**: Full support for single and multi-GPU setups
- ✅ **Tensor Parallelism**: Distribute model across multiple GPUs
- ✅ **Memory Management**: Configurable GPU memory utilization
- ✅ **Quantization Support**: Memory-efficient quantization (AWQ, GPTQ, etc.)
- ✅ **Comprehensive Metrics**: Real-time monitoring of GPU cache, latency, and throughput
- ✅ **Multiple Data Formats**: Support for CSV, JSON, and TXT datasets
- ✅ **Automatic Cleanup**: Proper server shutdown and resource cleanup

## Quick Start

### 1. Basic GPU Usage (Single GPU)

```bash
python src/experiment/server_exp.py \
    --model microsoft/DialoGPT-medium \
    --dataset sample_dataset.csv
```

### 2. Multi-GPU Setup with Tensor Parallelism

```bash
python src/experiment/server_exp.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset sample_dataset.csv \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85
```

### 3. Memory-Efficient Setup for Large Models

```bash
python src/experiment/server_exp.py \
    --model meta-llama/Llama-2-13b-chat-hf \
    --dataset sample_dataset.csv \
    --quantization awq \
    --gpu-memory-utilization 0.95 \
    --dtype float16
```

### 4. Specific GPU Selection

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/experiment/server_exp.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset sample_dataset.csv \
    --tensor-parallel-size 2
```

## Command Line Arguments

### Basic Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | **required** | Name or path of the model to load |
| `--dataset` | str | **required** | Path to dataset file (CSV, JSON, TXT) |
| `--host` | str | localhost | Host to bind server to |
| `--port` | int | 8000 | Port to bind server to |
| `--result-dir` | str | results | Directory to save results |
| `--max-tokens` | int | 10 | Maximum tokens per response |
| `--temperature` | float | 0.0 | Sampling temperature |
| `--timeout` | int | 30 | Request timeout in seconds |

### GPU Configuration Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tensor-parallel-size` | int | 1 | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | float | 0.9 | GPU memory utilization ratio (0.1-1.0) |
| `--dtype` | str | auto | Data type (auto, half, float16, bfloat16, float32) |
| `--max-model-len` | int | None | Maximum sequence length |
| `--quantization` | str | None | Quantization method (awq, gptq, squeezellm, fp8) |
| `--max-num-seqs` | int | 256 | Maximum sequences in batch |
| `--enforce-eager` | flag | False | Disable CUDA graph (for debugging) |
| `--disable-chunked-prefill` | flag | False | Disable chunked prefill |
| `--cuda-visible-devices` | str | None | Comma-separated GPU IDs (e.g., 0,1,2) |

## GPU Configuration Examples

### Single GPU Optimized

```bash
python src/experiment/server_exp.py \
    --model microsoft/DialoGPT-medium \
    --dataset sample_dataset.csv \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --dtype float16 \
    --max-num-seqs 128
```

### Multi-GPU High Throughput

```bash
python src/experiment/server_exp.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset sample_dataset.csv \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --max-num-seqs 512
```

### Large Model with Quantization

```bash
python src/experiment/server_exp.py \
    --model meta-llama/Llama-2-70b-chat-hf \
    --dataset sample_dataset.csv \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.95 \
    --dtype float16 \
    --quantization awq \
    --max-model-len 2048
```

### Debug Mode (CPU Fallback)

```bash
CUDA_VISIBLE_DEVICES="" python src/experiment/server_exp.py \
    --model microsoft/DialoGPT-small \
    --dataset sample_dataset.csv \
    --enforce-eager \
    --dtype float32
```

## Dataset Formats

### CSV Format
```csv
text,label
"This product is amazing!",positive
"Worst purchase ever",negative
```
Supported column names: `text`, `review`, `comment`, `content`, `message`

### JSON Format
```json
[
  {"text": "This product is amazing!", "label": "positive"},
  {"text": "Worst purchase ever", "label": "negative"}
]
```

### TXT Format
```
This product is amazing!
Worst purchase ever
Great service and fast delivery
```
One text per line.

## Output Files

The experiment generates several output files in the results directory:

### 1. Experiment Summary (`experiment_summary_YYYYMMDD_HHMMSS.json`)
```json
{
  "status": "completed",
  "experiment_config": {...},
  "timing_stats": {
    "total_experiment_time": 45.32,
    "total_inference_time": 23.14,
    "average_inference_time": 0.231,
    "throughput_requests_per_second": 2.21
  },
  "vllm_metrics_summary": {
    "cache_metrics": {
      "max_kv_cache_usage_percent": 45.2,
      "cache_hit_rate_percent": 78.5
    },
    "token_metrics": {...},
    "latency_histograms": {...}
  }
}
```

### 2. Detailed Results (`inference_results_YYYYMMDD_HHMMSS.json`)
Individual inference results with timing and response data.

### 3. Metrics Data (`metrics_data_YYYYMMDD_HHMMSS.json`)
Raw vLLM metrics collected during the experiment.

### 4. CSV Export (`inference_results_YYYYMMDD_HHMMSS.csv`)
Results in CSV format for easy analysis.

### 5. Log File (`experiment.log`)
Detailed logging of the experiment process.

## vLLM Metrics Collected

The script collects comprehensive vLLM metrics including:

### Cache Metrics
- **KV Cache Usage**: GPU memory used by key-value cache
- **Prefix Cache Hit Rate**: Efficiency of prefix caching
- **Cache Queries/Hits**: Total cache operations

### Performance Metrics
- **Time to First Token (TTFT)**: Latency for first token generation
- **Time Per Output Token (TPOT)**: Token generation speed
- **End-to-End Latency**: Complete request processing time
- **Throughput**: Requests processed per second

### Resource Metrics
- **GPU Memory Utilization**: Current GPU memory usage
- **Running/Waiting Requests**: Queue status
- **Token Generation**: Prompt and output token counts

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Out of Memory Errors
- Reduce `--gpu-memory-utilization` (try 0.7 or 0.8)
- Use `--quantization awq` or `--quantization gptq`
- Reduce `--max-model-len`
- Use `--dtype float16` or `--dtype bfloat16`

### Server Startup Issues
- Increase `--server-timeout` for large models
- Check if port is available: `lsof -i :8000`
- Verify model name/path is correct
- Check vLLM installation: `pip show vllm`

### Performance Issues
- Enable chunked prefill (default)
- Use appropriate `--tensor-parallel-size`
- Optimize `--max-num-seqs` for your GPU memory
- Consider `--dtype bfloat16` for newer GPUs

## Performance Optimization Tips

### Memory Optimization
1. **Use quantization**: `--quantization awq` can reduce memory by 2-4x
2. **Adjust memory utilization**: Start with 0.8, increase gradually
3. **Choose appropriate dtype**: `float16` or `bfloat16` for memory efficiency
4. **Limit sequence length**: Set `--max-model-len` appropriately

### Throughput Optimization
1. **Tensor parallelism**: Use multiple GPUs with `--tensor-parallel-size`
2. **Batch size**: Optimize `--max-num-seqs` for your hardware
3. **Chunked prefill**: Keep enabled for better memory usage
4. **CUDA graphs**: Disable `--enforce-eager` only for debugging

### Latency Optimization
1. **Single GPU**: Avoid tensor parallelism for small models
2. **Shorter sequences**: Reduce `--max-tokens` if possible
3. **Warmed cache**: Run a few requests before measuring performance
4. **Appropriate model size**: Balance model quality vs. latency

## Example Workflow

1. **Start with basic setup**:
   ```bash
   python src/experiment/server_exp.py --model microsoft/DialoGPT-small --dataset sample_dataset.csv
   ```

2. **Scale up model size**:
   ```bash
   python src/experiment/server_exp.py --model microsoft/DialoGPT-medium --dataset sample_dataset.csv --dtype float16
   ```

3. **Add quantization for large models**:
   ```bash
   python src/experiment/server_exp.py --model meta-llama/Llama-2-7b-chat-hf --dataset sample_dataset.csv --quantization awq
   ```

4. **Use multiple GPUs**:
   ```bash
   python src/experiment/server_exp.py --model meta-llama/Llama-2-7b-chat-hf --dataset sample_dataset.csv --tensor-parallel-size 2
   ```

5. **Analyze results** in the `results/` directory

## Integration with Analysis Tools

The generated metrics can be analyzed using:

- **Pandas**: Load CSV files for statistical analysis
- **Matplotlib/Seaborn**: Visualize performance metrics
- **Jupyter Notebooks**: Interactive analysis and plotting
- **Prometheus/Grafana**: Real-time monitoring (advanced setup)

## Requirements

- Python 3.8+
- vLLM 0.10.0+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support
- Required Python packages (see requirements-vllm.txt)

For installation instructions, see the main README.md file.
