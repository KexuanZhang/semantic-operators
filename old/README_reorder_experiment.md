# Reorder Inference Experiment

This script runs experiments to evaluate the impact of dataset reordering on LLM inference performance. It takes a dataset, optionally applies a reordering algorithm, runs LLM inference on each row, and saves the results and performance metrics.

## Features

- **Dataset Reordering**: Option to apply column reordering, row sorting, and deduplication
- **LLM Inference**: Uses vLLM to efficiently process each row with a specified prompt template
- **Result Tracking**: Saves all results and performance metrics in a timestamped folder

## Requirements

```
pandas
torch
vllm
```

## Usage

```bash
python reorder_inference_experiment.py --dataset path/to/dataset.csv [options]
```

### Command Line Arguments

#### Required Arguments:
- `--dataset`: Path to the input CSV dataset file

#### Dataset Configuration:
- `--content_column`: Column name for deduplication (default: 'review_content')
- `--max_rows`: Limit processing to this number of rows (optional)

#### Reordering Options:
- `--reorder`: Enable dataset reordering (default: disabled)
- `--no_sort`: Skip row sorting step when reordering
- `--no_dedup`: Skip deduplication step when reordering

#### LLM Configuration:
- `--model`: HuggingFace model to use (default: 'meta-llama/Llama-2-7b-hf')
- `--prompt_template`: Template string for generating prompts
- `--include_columns`: Columns to include in the prompt template

## Examples

### Run with Default Settings (No Reordering)
```bash
python reorder_inference_experiment.py --dataset movie_reviews.csv
```

### Apply Full Reordering Algorithm
```bash
python reorder_inference_experiment.py --dataset movie_reviews.csv --reorder
```

### Apply Partial Reordering (Column Reordering Only)
```bash
python reorder_inference_experiment.py --dataset movie_reviews.csv --reorder --no_sort --no_dedup
```

### Custom Prompt Template
```bash
python reorder_inference_experiment.py --dataset movie_reviews.csv --prompt_template "Is the movie {movie_title} suitable for children based on this review: {review_content}?"
```

### Process Limited Number of Rows (for Testing)
```bash
python reorder_inference_experiment.py --dataset movie_reviews.csv --max_rows 100
```

## Output

Results are saved in the `results/TIMESTAMP/` directory with the following files:
- `inference_results.json`: Contains LLM responses and per-row metrics
- `experiment_stats.json`: Overall experiment statistics
- `processed_dataset.csv`: The dataset after reordering (if applied)
- `experiment_summary.txt`: Human-readable summary of the experiment
