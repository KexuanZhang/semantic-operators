# Enhanced vLLM with Stats Logging for Experiments

This setup provides a modified version of vLLM with detailed statistics logging for offline inference experiments, specifically designed for KV cache usage, prefix cache hit rates, and request queue monitoring.

## 🚀 Quick Start

### 1. Install the Modified vLLM

```bash
cd /Users/zhang/Desktop/huawei/so1/vllm
./setup_vllm_with_stats.sh
```

### 2. Test the Installation

```bash
cd /Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline
python test_enhanced_vllm_integration.py
```

### 3. Run Enhanced Experiments

```bash
# Run with your dataset
python src/experiment/run_experiment_enhanced.py your_dataset.csv query_key --model your/model/path

# Example with sample data
python src/experiment/run_experiment_enhanced.py sample_dataset.csv filter_movies_kids --model Qwen/Qwen1.5-14B
```

## 📋 Features

### Enhanced Stats Logging
- **KV Cache Usage**: Real-time GPU KV cache utilization monitoring
- **Prefix Cache Hit Rate**: Detailed prefix cache hit/miss statistics  
- **Request Queue Stats**: Running, waiting, and corrupted request counts
- **Performance Metrics**: Throughput, latency, and token statistics
- **Speculative Decoding**: Draft token acceptance rates (if enabled)

### Comprehensive Output
- **Detailed Results**: CSV files with generation results and metadata
- **Stats Export**: JSON files with comprehensive statistics for analysis
- **Batch-level Monitoring**: Stats collected at each batch for trend analysis
- **Error Handling**: Robust error handling and logging

## 🔧 Configuration

### LLM Parameters
```python
llm = OfflineLLMWithStats(
    model="your/model/path",
    tensor_parallel_size=1,          # Number of GPUs
    max_model_len=2048,              # Maximum sequence length
    gpu_memory_utilization=0.9,      # GPU memory usage
    enable_prefix_caching=True,      # Enable prefix caching
    log_stats_interval=1,            # Stats logging frequency
    log_stats=True                   # Enable stats logging
)
```

### Query Templates
Available query types:
- **Aggregation**: `agg_movies_sentiment`, `agg_products_sentiment`
- **Multi-invocation**: `multi_movies_sentiment`, `multi_products_sentiment`
- **Filter**: `filter_movies_kids`, `filter_products_sentiment`, `filter_bird_statistics`, `filter_pdmx_individual`, `filter_beer_european`
- **Projection**: `proj_movies_summary`, `proj_products_consistency`, `proj_bird_comment`, `proj_pdmx_music`, `proj_beer_overview`
- **RAG**: `rag_fever`, `rag_squad`

## 📊 Output Files

### Results CSV
Contains generation results with metadata:
- `row_id`: Row identifier
- `query_type`: Query template used
- `prompt`: Input prompt (truncated)
- `generated_text`: Generated response
- `finish_reason`: Completion reason
- `prompt_tokens`: Number of prompt tokens
- `generated_tokens`: Number of generated tokens
- `batch_idx`: Batch number
- `inference_time`: Inference time for the batch

### Stats JSON
Contains comprehensive statistics:
```json
{
  "final_stats": {
    "total_requests": 100,
    "total_inference_time": 45.23,
    "avg_time_per_request": 0.452,
    "engine_kv_cache_usage": 0.75,
    "engine_prefix_cache_stats_hits": 42,
    "engine_prefix_cache_stats_queries": 100,
    "engine_num_running_reqs": 0,
    "engine_num_waiting_reqs": 0
  },
  "batch_stats": [...],  // Per-batch statistics
  "experiment_config": {...}  // Experiment configuration
}
```

## 🛠 Architecture

### Modified Components
1. **vLLM V1 EngineCore** (`/vllm/vllm/v1/engine/core.py`)
   - Added `get_stats()` method for stats access
   - Added `log_inference_stats()` for detailed logging
   - Modified `step()` method to include stats logging

2. **Enhanced LLM Wrapper** (`/vllm/vllm/offline_llm_with_stats.py`)
   - Wraps vLLM's LLM class with stats monitoring
   - Auto-detects V0 vs V1 engine formats
   - Provides performance metrics collection

3. **Integration Layer** (`use_local_vllm.py`)
   - Manages local vLLM path setup
   - Provides convenience functions for experiments
   - Handles initialization and verification

## 📈 Usage Examples

### Basic Usage
```python
from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm
from vllm import SamplingParams

# Initialize
initialize_experiment_with_local_vllm()

# Create enhanced LLM
llm = create_enhanced_llm("your/model/path")

# Run inference with stats
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Your prompt here"], sampling_params, log_detailed_stats=True)

# Get stats
stats = llm.get_current_stats()
print(f"KV Cache Usage: {stats.get('engine_kv_cache_usage', 0)*100:.1f}%")
```

### Experiment Pipeline
```python
from src.experiment.run_experiment_enhanced import run_enhanced_experiment

# Run comprehensive experiment
results_df, detailed_stats = run_enhanced_experiment(
    csv_file="dataset.csv",
    query_key="filter_movies_kids",
    model_path="Qwen/Qwen1.5-14B",
    batch_size=10,
    output_dir="results"
)
```

## 🔍 Monitoring and Analysis

### Key Metrics to Track
- **KV Cache Usage** (`engine_kv_cache_usage`): Percentage of KV cache utilized
- **Prefix Hit Rate** (`hits/queries`): Efficiency of prefix caching
- **Request Queue** (`num_running_reqs`, `num_waiting_reqs`): System load
- **Throughput** (`total_requests/total_inference_time`): Processing speed
- **Token Statistics** (`prompt_tokens`, `generated_tokens`): Workload characteristics

### Performance Optimization Tips
1. **Batch Size**: Larger batches improve throughput but increase memory usage
2. **Prefix Caching**: Enable for repeated prompt patterns to improve hit rates
3. **Model Length**: Reduce `max_model_len` if not using long contexts
4. **GPU Memory**: Increase `gpu_memory_utilization` for better cache utilization
5. **Tensor Parallelism**: Use multiple GPUs for large models

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure local vLLM is properly installed
   cd /Users/zhang/Desktop/huawei/so1/vllm
   pip install -e . --verbose
   ```

2. **CUDA Memory Issues**
   ```bash
   # Reduce memory usage
   --gpu-memory 0.7 --max-model-len 1024
   ```

3. **Stats Not Available**
   ```python
   # Ensure log_stats=True is set
   llm = OfflineLLMWithStats(model="path", log_stats=True)
   ```

### Debug Mode
Enable debug logging for detailed information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 File Structure

```
/Users/zhang/Desktop/huawei/so1/
├── vllm/                                    # Modified vLLM repository
│   ├── vllm/v1/engine/core.py              # Enhanced EngineCore with stats
│   ├── vllm/offline_llm_with_stats.py      # Enhanced LLM wrapper
│   ├── setup_vllm_with_stats.sh            # Installation script
│   └── test_vllm_stats_logging.py          # Basic test script
├── semantic-operators/ggr-experiment-pipeline/
│   ├── use_local_vllm.py                   # Integration layer
│   ├── src/experiment/run_experiment_enhanced.py  # Enhanced experiment runner
│   ├── test_enhanced_vllm_integration.py   # Integration test script
│   └── README_ENHANCED.md                  # This file
```

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the test script to verify installation
3. Check logs for detailed error information
4. Ensure CUDA setup is correct for GPU usage

## 📊 Statistics Reference

### V1 Engine Stats (SchedulerStats)
- `kv_cache_usage`: GPU KV cache utilization (0.0-1.0)
- `prefix_cache_stats.hits`: Number of prefix cache hits
- `prefix_cache_stats.queries`: Number of prefix cache queries  
- `num_running_reqs`: Currently running requests
- `num_waiting_reqs`: Requests waiting in queue
- `num_corrupted_reqs`: Number of corrupted requests
- `step_counter`: Engine step counter
- `spec_decoding_stats`: Speculative decoding statistics (if enabled)

### V0 Engine Stats (Legacy)
- `gpu_cache_usage_sys`: GPU KV cache usage
- `gpu_prefix_cache_hit_rate`: GPU prefix cache hit rate
- `cpu_prefix_cache_hit_rate`: CPU prefix cache hit rate
- `num_running_sys`: Running requests
- `num_waiting_sys`: Waiting requests  
- `num_swapped_sys`: Swapped requests

The system automatically detects and handles both V0 and V1 engine formats.
