# KVTuner Integration Scripts

This directory contains scripts for working with KVTuner, a tool for optimizing KV cache quantization in LLM inference.

## Overview

Two main scripts are provided:

1. `find_optimal_kv_config.py` - Finds optimal key-value quantization configurations for a given model
2. `run_kvtuner_inference.py` - Runs inference using KVTuner with a given configuration

## Requirements

- KVTuner repository cloned in parallel to this repository
- PyTorch
- Transformers
- The necessary dependencies for KVTuner

## Finding Optimal KV Configuration

The `find_optimal_kv_config.py` script helps you find the optimal KV configuration for your model.

### Usage

```bash
python find_optimal_kv_config.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --scheme pertoken \
    --target_bits 4.0 \
    --output_dir ./configs \
    --search_method optuna \
    --n_trials 30
```

### Parameters

- `--model_name` (required): Path to local model or HuggingFace model name
- `--scheme`: Quantization scheme - `pertoken` (per-token for both K&V) or `kivi` (per-channel for K, per-token for V)
- `--output_dir`: Directory to save KV configs
- `--target_bits`: Target average bit width (default: 4.0)
- `--search_method`: Search method - `brute` or `optuna` (default: optuna)
- `--n_trials`: Number of trials for optuna search (default: 30)
- `--sample_limit`: Number of samples for evaluation (default: 20)
- `--debug`: Enable debug logging

## Running Inference with KVTuner

The `run_kvtuner_inference.py` script runs inference on a dataset using KVTuner with a specified configuration.

### Usage

```bash
python run_kvtuner_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset dataset.jsonl \
    --kv_config ./configs/model_config.yaml \
    --scheme kivi \
    --output_dir ./results
```

### Parameters

#### Model and data parameters:
- `--model_name` (required): Path to local model or HuggingFace model name
- `--dataset` (required): Path to dataset file (JSON or JSONL)
- `--output_dir`: Directory to save results (default: ./results)

#### KVTuner configuration:
- `--kv_config`: Path to KV config YAML file (if using per-layer config)
- `--scheme`: Quantization scheme - `pertoken`, `kivi`, or `none` (for no quantization)
- `--key_bits`: Number of bits for keys when using uniform quantization (default: 4)
- `--value_bits`: Number of bits for values when using uniform quantization (default: 4)

#### Inference parameters:
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 256)
- `--batch_size`: Batch size for inference (default: 1)
- `--prompt_key`: Key in dataset entries to use as prompt (default: "prompt")
- `--sample_limit`: Limit the number of samples to process (default: None)
- `--gpu_ids`: Comma-separated list of GPU IDs to use (e.g., '0,1')

#### Logging parameters:
- `--debug`: Enable debug logging

## Examples

### 1. Find optimal KV config for Meta-Llama-3-8B with KiVi scheme:

```bash
python find_optimal_kv_config.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --scheme kivi \
    --target_bits 4.0 \
    --output_dir ./configs \
    --search_method optuna \
    --n_trials 50
```

### 2. Run inference with the found configuration:

```bash
python run_kvtuner_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset path/to/dataset.jsonl \
    --kv_config ./configs/Meta_Llama_3_8B_kivi_4_bits_20230909_123456.yaml \
    --scheme kivi \
    --output_dir ./results
```

### 3. Run inference with uniform quantization:

```bash
python run_kvtuner_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset path/to/dataset.jsonl \
    --scheme pertoken \
    --key_bits 4 \
    --value_bits 4 \
    --output_dir ./results
```

### 4. Use existing calibration presets from KVTuner:

```bash
python run_kvtuner_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset path/to/dataset.jsonl \
    --kv_config ../KVTuner/calibration_presets/Meta-Llama-3.1-8B-Instruct_kivi_KVTuner4_0.yaml \
    --scheme kivi \
    --output_dir ./results
```

### 5. Run on specific GPUs:

```bash
python run_kvtuner_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset path/to/dataset.jsonl \
    --scheme pertoken \
    --key_bits 4 \
    --value_bits 4 \
    --gpu_ids "0,1" \
    --output_dir ./results
```

## Notes

- Make sure the KVTuner repository is properly cloned and accessible
- For local models, ensure all model files are properly organized
- The inference script creates a new KV cache for each sample to avoid cross-contamination
- Results are saved in a timestamped directory for easy tracking
