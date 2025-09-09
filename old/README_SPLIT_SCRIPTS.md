# Dataset Reordering and LLM Inference Scripts

This folder contains scripts to perform dataset reordering and LLM inference, either separately or together.

## Scripts

1. **`dataset_reordering.py`**: Applies the reordering algorithm to a dataset
2. **`llm_inference.py`**: Runs LLM inference on a dataset
3. **`run_complete_experiment.py`**: Combines both operations in sequence
4. **`run_experiment_with_gpus.sh`**: Helper script to run experiments with specific GPU IDs

## Usage

### Dataset Reordering Only

```bash
python dataset_reordering.py --dataset path/to/dataset.csv --reorder [options]
```

Options:
- `--content_column COL_NAME`: Column name for deduplication (default: 'review_content')
- `--no_sort`: Skip row sorting step
- `--no_dedup`: Skip deduplication step
- `--max_rows N`: Limit processing to N rows (for testing)
- `--output_prefix PREFIX`: Prefix for output files

Output: Reordered dataset and statistics in `reordered/TIMESTAMP/` directory

### LLM Inference Only

```bash
python llm_inference.py --dataset path/to/dataset.csv --model MODEL_NAME [options]
```

Options:
- `--gpu_ids "0,1"`: Specific GPU IDs to use
- `--tp_size N`: Number of GPUs for tensor parallelism
- `--prompt_template TEMPLATE`: Custom prompt template
- `--include_columns COL1 COL2`: Columns to include in prompts
- `--max_rows N`: Limit processing to N rows (for testing)
- `--output_prefix PREFIX`: Prefix for output files

Output: Inference results and statistics in `inference_results/TIMESTAMP/` directory

### Complete Experiment (Reordering + Inference)

```bash
python run_complete_experiment.py --dataset path/to/dataset.csv --model MODEL_NAME --reorder [options]
```

Options: Combines options from both scripts above

## Examples

### Basic reordering:

```bash
python dataset_reordering.py --dataset movies.csv --reorder
```

### Inference with specific GPUs:

```bash
python llm_inference.py --dataset movies.csv --model meta-llama/Llama-2-7b-chat-hf --gpu_ids "6,7" --tp_size 2
```

### Complete experiment with custom prompt:

```bash
python run_complete_experiment.py --dataset movies.csv --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --reorder --prompt_template "Analyze this review: {review_content}" --gpu_ids "6,7"
```

## Output Directories

- Reordering results: `reordered/TIMESTAMP/`
- Inference results: `inference_results/TIMESTAMP/`

Each directory contains detailed statistics and output files from the corresponding process.
