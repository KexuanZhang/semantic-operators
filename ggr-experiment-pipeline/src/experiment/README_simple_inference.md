# Simple LLM Inference Script

This is a minimal experiment script that takes only dataset and query arguments, feeds them to an LLM, and outputs only the inference time in JSON format.

## Features

- **Minimal Dependencies**: Only requires vLLM and standard Python libraries
- **Simple Usage**: Just dataset and query arguments
- **JSON Output**: Clean JSON format with inference timing
- **Multiple Formats**: Supports CSV, JSON, JSONL, and Parquet datasets
- **Flexible Queries**: Pre-defined templates or custom queries

## Installation

```bash
pip install vllm torch pandas
```

## Usage

### Basic Usage

```bash
# Using pre-defined query templates
python simple_inference.py dataset.csv movie_analysis
python simple_inference.py dataset.csv summarize
python simple_inference.py dataset.csv extract
```

### Custom Queries

```bash
# Using custom query
python simple_inference.py dataset.csv custom --custom-query "Analyze this data and provide insights:"
```

### Additional Options

```bash
# Limit number of rows
python simple_inference.py dataset.csv movie_analysis --max-rows 50

# Save output to file
python simple_inference.py dataset.csv summarize --output results.json

# Use different model
python simple_inference.py dataset.csv extract --model "meta-llama/Llama-3-8b-hf"

# Adjust generation parameters
python simple_inference.py dataset.csv custom --custom-query "Summarize:" --max-tokens 512 --temperature 0.2
```

## Output Format

The script outputs a JSON object with the following structure:

```json
{
  "inference_time_seconds": 15.42,
  "dataset_path": "dataset.csv",
  "query_key": "movie_analysis",
  "custom_query": null,
  "total_rows_processed": 100,
  "timestamp": "2024-01-15T14:30:45.123456",
  "model_name": "meta-llama/Llama-2-7b-hf",
  "avg_time_per_row": 0.154
}
```

## Pre-defined Query Templates

- **`movie_analysis`**: Analyzes movie/entertainment data
- **`summarize`**: Creates brief summaries of data entries
- **`extract`**: Extracts key information from data
- **`custom`**: Use your own query text

## Examples

### Movie Data Analysis
```bash
python simple_inference.py movies.csv movie_analysis --max-rows 25 --output movie_results.json
```

### Custom Data Summarization
```bash
python simple_inference.py customer_data.csv custom --custom-query "Provide a customer profile summary:" --max-rows 50
```

### Quick Testing
```bash
python simple_inference.py sample.csv summarize --max-rows 5
```

## Command Line Arguments

- `dataset` - Path to dataset file (required)
- `query_key` - Query template key: movie_analysis, summarize, extract, custom (required)
- `--custom-query` - Custom query text (required when using 'custom' key)
- `--max-rows` - Maximum number of rows to process
- `--model` - Model name or path (default: meta-llama/Llama-2-7b-hf)
- `--output` - Output JSON file path (default: stdout)
- `--max-tokens` - Maximum tokens to generate (default: 256)
- `--temperature` - Sampling temperature (default: 0.1)
- `--top-p` - Top-p sampling parameter (default: 0.9)

## Differences from Full Experiment Script

This simplified script:
- ❌ **Removes**: Comprehensive reporting, visualizations, monitoring, batch analysis
- ❌ **Removes**: KV cache analysis, resource utilization tracking, multi-GPU support
- ❌ **Removes**: Complex metrics collection and performance analysis
- ✅ **Keeps**: Basic LLM inference and timing measurement
- ✅ **Keeps**: Dataset loading and prompt creation
- ✅ **Keeps**: Simple JSON output format

## Performance

The script focuses purely on inference timing and provides:
- Total inference time in seconds
- Average time per row
- Row count processed
- Timestamp of execution

Perfect for quick benchmarking and timing measurements without the overhead of comprehensive analysis features.
