# Script Usage Guide

This document provides a summary of commands for using our data processing and LLM inference scripts.

## Dataset Sampler

Use the dataset sampler script to create smaller samples of large datasets.

```bash
python dataset_sampler.py --input path/to/original.csv --output path/to/sample.csv --sample_size 1000
```

**Key Options:**
- `--input`: Path to the original dataset CSV
- `--output`: Path to save the sampled dataset 
- `--sample_size`: Number of rows to sample
- `--random_seed`: Optional seed for reproducibility

**Example:**
```bash
python dataset_sampler.py --input rotten_tomatoes_critic_reviews.csv --output sampled_reviews.csv --sample_size 500 --random_seed 42
```

## Dataset Reordering

Apply the reordering algorithm to a dataset to potentially improve LLM performance.

```bash
python dataset_reordering.py --dataset path/to/dataset.csv --reorder [options]
```

**Key Options:**
- `--dataset`: Path to the dataset CSV
- `--reorder`: Flag to apply reordering algorithm
- `--content_column`: Column containing the main content (default: 'review_content')
- `--no_sort`: Skip row sorting
- `--no_dedup`: Skip deduplication
- `--max_rows`: Limit processing to specified number of rows
- `--output_prefix`: Custom prefix for output files

**Examples:**
```bash
# Basic reordering
python dataset_reordering.py --dataset movie_reviews.csv --reorder

# Reordering with custom options
python dataset_reordering.py --dataset movie_reviews.csv --reorder --content_column review_text --no_sort --max_rows 500
```

## LLM Inference

Run LLM inference on a dataset (original or reordered).

```bash
python llm_inference.py --dataset path/to/dataset.csv --model model_name [options]
```

**Key Options:**
- `--dataset`: Path to the dataset CSV
- `--model`: HuggingFace model name or path to local model
- `--gpu_ids`: Specific GPU IDs to use (e.g., "6,7")
- `--tp_size`: Tensor parallel size for multi-GPU inference
- `--prompt_template`: Custom prompt template
- `--include_columns`: Columns to include in prompts
- `--max_rows`: Limit processing to specified number of rows

**Examples:**
```bash
# Basic inference with default settings
python llm_inference.py --dataset movie_reviews.csv --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Advanced inference with specific GPUs and custom prompt
python llm_inference.py --dataset reordered_reviews.csv --model meta-llama/Llama-2-7b-chat-hf --gpu_ids "6,7" --tp_size 2 --prompt_template "Analyze this review: {review_content}. Is this a positive review?" --include_columns review_content movie_title
```



## Complete Experiment

To run reordering followed by inference in one command:

```bash
python run_complete_experiment.py --dataset path/to/dataset.csv --model model_name --reorder [options]
```

**Example:**
```bash
python run_complete_experiment.py --dataset movie_reviews.csv --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --reorder --gpu_ids "6,7" --tp_size 2 --max_rows 500
```

## Output Locations

- Dataset reordering: `reordered/TIMESTAMP/`
- LLM inference: `inference_results/TIMESTAMP/`

Each directory contains results, statistics, and summary files.
