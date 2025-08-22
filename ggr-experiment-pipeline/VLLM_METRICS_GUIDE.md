# vLLM Metrics Collection Guide

This guide explains the comprehensive metrics collection system implemented in the server experiment script (`server_exp.py`).

## Overview

The script collects detailed vLLM metrics through the Prometheus-compatible `/metrics` endpoint, providing insights into model performance, KV cache efficiency, and resource utilization during sentiment analysis experiments.

## Metrics Categories

### 1. Frontend Stats Collection

The vLLM frontend collects various statistics for each engine core iteration:

#### Token Metrics
- **Prompt Tokens** (`vllm:prompt_tokens_total`): Total number of prompt tokens processed
- **Generation Tokens** (`vllm:generation_tokens_total`): Total number of new tokens generated
- **Total Tokens**: Combined prompt and generation tokens

#### Request Flow Metrics
- **Running Requests** (`vllm:num_requests_running`): Current number of requests in execution
- **Waiting Requests** (`vllm:num_requests_waiting`): Current number of queued requests
- **Success Rate**: Ratio of successful to total requests

### 2. KV Cache Metrics

#### Cache Usage
- **GPU Cache Usage** (`vllm:gpu_cache_usage_perc`): Percentage of GPU memory used by KV cache
- **Peak Usage**: Maximum KV cache utilization during experiment

#### Prefix Caching (v0.10.0+)
- **Cache Queries** (`vllm:cache_query_total`): Total number of cache queries (Counter)
- **Cache Hits** (`vllm:cache_query_hit_total`): Number of successful cache hits (Counter)
- **Hit Rate**: Calculated as `hits / queries * 100`

> **Note**: Modern vLLM versions use counters for cache queries/hits instead of a gauge for hit rate, enabling time-series analysis in Prometheus with queries like:
> ```promql
> rate(vllm:cache_query_hit_total[5m]) / rate(vllm:cache_query_total[5m])
> ```

### 3. Latency Histograms

The script captures detailed latency distributions:

#### Time to First Token (TTFT)
- **Metric**: `vllm:time_to_first_token_seconds`
- **Description**: Distribution of time from request arrival to first token generation
- **Buckets**: 0.001s, 0.005s, 0.01s, 0.02s, 0.04s, 0.06s, 0.08s, 0.1s, +Inf

#### Time Per Output Token (TPOT)
- **Metric**: `vllm:time_per_output_token_seconds`
- **Description**: Inter-token intervals for ongoing generation
- **Usage**: Measures generation efficiency after prefill

#### End-to-End Latency
- **Metric**: `vllm:e2e_request_latency_seconds`
- **Description**: Total time from frontend arrival to final token
- **Includes**: Input processing, queuing, prefill, and generation time

### 4. Request Success Tracking

#### Success Counters
- **Success by Reason** (`vllm:request_success_total`):
  - `finished_reason="stop"`: Normal completion
  - `finished_reason="length"`: Max tokens reached
  - `finished_reason="abort"`: Request aborted

#### Request Flow Intervals
- **Queue Time**: Time spent waiting in queue before scheduling
- **Prefill Time**: Time to process input prompt
- **Inference Time**: Total generation time
- **Decode Time**: Time for output token generation

## Metrics Collection Implementation

### Collection Process
1. **Parallel Monitoring**: Metrics collected in separate thread every 1 second (configurable)
2. **Prometheus Parsing**: Uses `prometheus_client.parser` to parse `/metrics` endpoint
3. **Type Classification**: Categorizes metrics as Counters, Gauges, or Histograms
4. **Delta Calculations**: Computes deltas for counters to show experiment-specific changes

### Data Storage
- **Raw Metrics**: All metrics with timestamps stored in JSON format
- **Summary Statistics**: Key metrics summarized for easy analysis
- **CSV Export**: Inference results exported as CSV for data analysis tools

## Usage Example

```bash
# Basic experiment
python server_exp.py --model microsoft/DialoGPT-medium --dataset data/reviews.csv

# With custom metrics collection interval
python server_exp.py --model microsoft/DialoGPT-medium --dataset data/reviews.csv --metrics-interval 0.5

# Enable prefix caching for better cache metrics
python server_exp.py --model microsoft/DialoGPT-medium --dataset data/reviews.csv --enable-prefix-caching
```

## Output Files

The script generates several output files in the results directory:

1. **experiment_summary_{timestamp}.json**: High-level experiment results and metrics summary
2. **inference_results_{timestamp}.json**: Detailed per-request results
3. **metrics_data_{timestamp}.json**: Raw time-series metrics data
4. **inference_results_{timestamp}.csv**: CSV format for analysis tools
5. **experiment.log**: Detailed execution log

## Key Metrics Summary

The experiment summary includes:

```json
{
  "vllm_metrics_summary": {
    "cache_metrics": {
      "max_kv_cache_usage_percent": 85.2,
      "total_cache_queries": 1500,
      "total_cache_hits": 1200,
      "cache_hit_rate_percent": 80.0
    },
    "token_metrics": {
      "total_prompt_tokens": 15000,
      "total_generation_tokens": 2000,
      "total_tokens": 17000
    },
    "request_metrics": {
      "max_running_requests": 8,
      "max_waiting_requests": 12
    },
    "latency_histograms": {
      "time_to_first_token": {"total_count": 100, "buckets": {...}},
      "time_per_output_token": {"total_count": 2000, "buckets": {...}},
      "end_to_end_latency": {"total_count": 100, "buckets": {...}}
    }
  }
}
```

## Deprecated/Legacy Metrics

Some metrics are deprecated in newer vLLM versions:

- `vllm:gpu_prefix_cache_hit_rate`: Replaced by separate counters
- `vllm:num_requests_swapped`: V0 preemption mode (no longer relevant)
- `vllm:cpu_cache_usage_perc`: CPU swapping deprecated in favor of prefix caching

## Best Practices

1. **Collection Interval**: 1-second intervals provide good resolution without overhead
2. **Prefix Caching**: Enable for better cache efficiency metrics
3. **Long-Running Experiments**: Monitor trends over time for cache warmup effects
4. **Resource Monitoring**: Watch KV cache usage to avoid OOM conditions
5. **Histogram Analysis**: Use bucket distributions to understand latency patterns

## Troubleshooting

### Common Issues

1. **Missing Metrics**: Ensure vLLM version supports the metrics (v0.10.0+ recommended)
2. **Empty Cache Metrics**: Enable prefix caching with `--enable-prefix-caching`
3. **High Memory Usage**: Monitor `vllm:gpu_cache_usage_perc` and adjust batch size
4. **Connection Errors**: Verify server startup and `/metrics` endpoint availability

### Debug Tips

- Use `--metrics-interval 0.1` for higher resolution debugging
- Check `experiment.log` for detailed execution information
- Verify Prometheus metrics format with direct `/metrics` endpoint access
