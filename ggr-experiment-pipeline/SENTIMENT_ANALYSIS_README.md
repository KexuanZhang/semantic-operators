# Sentiment Analysis Experiment Runner

A Python tool for running sentiment analysis experiments using local LLM models with vLLM for fast GPU inference. The tool outputs raw LLM responses without post-processing, allowing you to analyze the model's natural language output.

## Features

- Load and run local LLM models using vLLM for optimized GPU inference
- Process CSV datasets for batch sentiment analysis
- GPU device selection for multi-GPU systems
- Save detailed results including raw LLM responses and timing information
- Export results in both JSON and CSV formats
- Track inference times and experiment metadata

## Installation

1. Clone the repository and navigate to the project directory
2. Install the required dependencies:

```bash
pip install -r requirements-vllm.txt
```

Required packages:
- vllm
- torch
- pandas
- tqdm

## Usage

### Basic Command

```bash
python src/experiment/run_exp.py \
    --model-path /path/to/your/local/llm/model \
    --dataset-path your_dataset.csv \
    --text-column text_column_name \
    --results-dir results \
    --experiment-name my_experiment
```

### With GPU Device Selection

```bash
python src/experiment/run_exp.py \
    --model-path /path/to/your/local/llm/model \
    --dataset-path your_dataset.csv \
    --text-column text_column_name \
    --results-dir results \
    --gpu-device cuda:0 \
    --experiment-name my_experiment_gpu0
```

### Arguments

- `--model-path` (required): Path to your local LLM model directory
- `--dataset-path` (required): Path to your CSV dataset file
- `--text-column` (optional): Name of the column containing text to analyze (default: "text")
- `--results-dir` (optional): Directory to save results (default: "results")
- `--gpu-device` (optional): GPU device to use (e.g., "cuda:0", "cuda:1"). If not specified, uses default GPU
- `--experiment-name` (optional): Custom name for experiment files (default: auto-generated timestamp)
- `--batch-size` (optional): Batch size for processing (default: 1)

## Dataset Format

Your CSV dataset should contain at least one text column for analysis. Example:

```csv
id,text,category
1,"I love this product! It's amazing.",electronics
2,"This is terrible. I hate it.",electronics
3,"It's okay, not great but not bad.",books
```

## Output

The experiment generates three output files in the results directory:

1. **`{experiment_name}_results.json`**: Detailed results with raw LLM responses
2. **`{experiment_name}_results.csv`**: Results in CSV format for easy analysis
3. **`{experiment_name}_metadata.json`**: Experiment metadata and timing information

### Output Format

Each result entry contains:
- `index`: Row index from original dataset
- `text`: Original text that was analyzed
- `predicted_sentiment`: Raw LLM response (unprocessed)
- `inference_time`: Time taken for this specific inference
- `timestamp`: When the inference was performed
- `original_*`: All other columns from the original dataset

## Example

Run the example script to see usage demonstrations:

```bash
python example_usage.py
```

This will create a sample dataset and show various command examples.

## GPU Requirements

- CUDA-compatible GPU
- Sufficient GPU memory for your model (typically 8GB+ for 7B models)
- CUDA toolkit installed
- PyTorch with CUDA support

## Multi-GPU Usage

To use specific GPU devices in multi-GPU systems:

```bash
# Use first GPU
python src/experiment/run_exp.py --gpu-device cuda:0 ...

# Use second GPU
python src/experiment/run_exp.py --gpu-device cuda:1 ...

# Use third GPU
python src/experiment/run_exp.py --gpu-device cuda:2 ...
```

## Performance Notes

- vLLM provides optimized inference with features like:
  - PagedAttention for memory efficiency
  - Continuous batching for higher throughput
  - Optimized CUDA kernels
- GPU memory utilization is set to 80% by default
- Inference times are tracked per sample for performance analysis

## Troubleshooting

1. **Out of GPU memory**: Reduce model size or increase GPU memory limit
2. **Model loading errors**: Verify model path and format compatibility
3. **CUDA errors**: Ensure CUDA toolkit and PyTorch GPU support are properly installed
4. **CSV parsing errors**: Check dataset format and column names

## Example Commands

### Using Llama 2 7B model on first GPU
```bash
python src/experiment/run_exp.py \
    --model-path /data/models/llama2-7b-chat \
    --dataset-path movie_reviews.csv \
    --text-column review_text \
    --gpu-device cuda:0 \
    --experiment-name llama2_movie_sentiment
```

### Using different GPU for parallel experiments
```bash
# Terminal 1 - GPU 0
python src/experiment/run_exp.py \
    --model-path /data/models/llama2-7b-chat \
    --dataset-path dataset_part1.csv \
    --gpu-device cuda:0 \
    --experiment-name experiment_part1_gpu0

# Terminal 2 - GPU 1
python src/experiment/run_exp.py \
    --model-path /data/models/llama2-7b-chat \
    --dataset-path dataset_part2.csv \
    --gpu-device cuda:1 \
    --experiment-name experiment_part2_gpu1
```

## Sample Output

The raw LLM responses are preserved exactly as generated, for example:

```json
{
  "index": 0,
  "text": "I love this product! It's amazing and works perfectly.",
  "predicted_sentiment": "POSITIVE\n\nThis text expresses clear satisfaction and enthusiasm about the product, using positive language like 'love', 'amazing', and 'perfectly'.",
  "inference_time": 0.234,
  "timestamp": "2025-08-22T10:30:15.123456"
}
```

This allows for detailed analysis of how the model responds to different prompts and texts.
