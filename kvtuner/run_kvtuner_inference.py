#!/usr/bin/env python3
"""
KVTuner Inference Script for Local Models

This script loads a locally downloaded model, applies KVTuner with a specified
configuration file, and runs inference on a provided dataset.

Usage:
    python run_kvtuner_inference.py \
        --model_path "/path/to/local/model" \
        --dataset "/path/to/dataset.csv" \
        --kv_config "../KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml" \
        --scheme pertoken \
        --gpu_ids "0,1"
"""

import os
import sys
import argparse
import torch
import yaml
import json
import pandas as pd
import time
import logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Add KVTuner path
sys.path.append('/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner')

try:
    from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Failed to import KVTuner modules: {e}")
    print("Make sure the KVTuner repository is properly installed")
    sys.exit(1)


def setup_logging(debug=False):
    """Configure logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("kvtuner_inference")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with KVTuner on a local model')
    
    # Model and dataset configuration
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to local model directory')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to dataset file (CSV format)')
    parser.add_argument('--output_dir', type=str, default='./results', 
                        help='Directory to save results')
    
    # KVTuner configuration
    parser.add_argument('--kv_config', type=str, required=True,
                        help='Path to KVTuner configuration YAML file')
    parser.add_argument('--scheme', type=str, default='pertoken', choices=['pertoken', 'kivi'],
                        help='Quantization scheme: pertoken or kivi')
    
    # Text processing options
    parser.add_argument('--text_column', type=str, default=None,
                        help='Column name in the dataset to use as input text')
    parser.add_argument('--prompt_template', type=str, default='{text}',
                        help='Template for formatting prompts with dataset values')
    
    # Inference parameters
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,1")')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Maximum number of rows to process from the dataset')
    
    # Debugging and logging options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()


def set_gpu_environment(gpu_ids):
    """Set GPU environment variables for specific device selection"""
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return f"Using GPUs: {gpu_ids}"
    return "Using all available GPUs"


def format_prompt_by_model(model_name, text):
    """Format prompt based on model type"""
    model_name_lower = model_name.lower() if model_name else ""
    
    # Qwen models
    if "qwen" in model_name_lower:
        return f"<|im_start|>system\nYou are a helpful assistant that provides clear, concise, and accurate answers.\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
    
    # LLaMA family models (LLaMA, Mistral, etc)
    elif any(name in model_name_lower for name in ["llama", "mistral", "vicuna"]):
        return f"<s>[INST] {text} [/INST]"
    
    # Yi models
    elif "yi" in model_name_lower:
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    # Default format
    else:
        return f"### Instruction:\n{text}\n\n### Response:\n"


def clean_response_by_model(model_name, response):
    """Clean response based on model type"""
    model_name_lower = model_name.lower() if model_name else ""
    
    # Qwen models
    if "qwen" in model_name_lower and "<|im_end|>" in response:
        return response.split("<|im_end|>")[0].strip()
    
    # LLaMA family models
    elif any(name in model_name_lower for name in ["llama", "mistral", "vicuna"]):
        if "</s>" in response:
            return response.split("</s>")[0].strip()
    
    # Yi models
    elif "yi" in model_name_lower and "<|im_end|>" in response:
        return response.split("<|im_end|>")[0].strip()
    
    return response.strip()


def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from local path"""
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # Get model basename for prompt formatting
    model_name = os.path.basename(model_path)
    print(f"Model {model_name} loaded successfully")
    
    return model, tokenizer, model_name


def load_kv_config(config_path, scheme):
    """Load KVTuner configuration from YAML file"""
    print(f"Loading KV configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Empty or invalid KV configuration in {config_path}")
            
        print(f"KV configuration loaded successfully with {len(config_data)} layer entries")
        
        # Set up axis configuration based on scheme
        if scheme == 'kivi':
            axis_key = 1  # Per-channel for keys in KiVi
            axis_value = 0  # Per-token for values in KiVi
            q_group_size = 32
            residual_length = 32
        else:  # pertoken
            axis_key = 0  # Per-token for keys
            axis_value = 0  # Per-token for values
            q_group_size = -1
            residual_length = 0
            
        # Create FlexibleQuantizedCacheConfig
        cache_config = FlexibleQuantizedCacheConfig(
            device="cuda",
            per_layer_quant=True,
            per_layer_config=config_data,  # Directly use the loaded config
            asym=True,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length
        )
        
        return cache_config
        
    except Exception as e:
        print(f"Error loading KV configuration: {e}")
        raise


def load_dataset(dataset_path, text_column=None, max_rows=None):
    """Load dataset from CSV file"""
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Identify text column if not specified
        if text_column is None:
            # Try to find column with text content based on common naming patterns
            possible_text_columns = [
                col for col in df.columns if any(
                    name in col.lower() for name in [
                        'text', 'content', 'prompt', 'input', 'query', 'question', 'instruction'
                    ]
                )
            ]
            
            if possible_text_columns:
                text_column = possible_text_columns[0]
                print(f"Automatically selected text column: '{text_column}'")
            else:
                # Use the first column as default
                text_column = df.columns[0]
                print(f"No text column specified or detected, using first column: '{text_column}'")
        else:
            if text_column not in df.columns:
                raise ValueError(f"Specified text column '{text_column}' not found in dataset")
                
        # Apply row limit if specified
        if max_rows is not None:
            df = df.head(max_rows)
            print(f"Using first {max_rows} rows from dataset")
            
        return df, text_column
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def run_inference(model, tokenizer, text, kv_cache, model_name, max_new_tokens=256):
    """Run inference with the model using KVTuner"""
    # Format the prompt based on model type
    formatted_prompt = format_prompt_by_model(model_name, text)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            past_key_values=kv_cache,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract generated text (skip prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Clean response based on model type
    cleaned_response = clean_response_by_model(model_name, generated_text)
    
    return cleaned_response


def process_dataset(df, model, tokenizer, model_name, kv_config, text_column, prompt_template, max_new_tokens):
    """Process dataset with KVTuner inference"""
    results = []
    total_time = 0
    
    # Set up progress bar
    progress_bar = tqdm(total=len(df), desc="Processing dataset")
    
    for idx, row in df.iterrows():
        # Create a fresh KV cache for each example to avoid cross-contamination
        kv_cache = FlexibleVanillaQuantizedCache(cache_config=kv_config)
        
        # Get text from dataset
        text = row[text_column]
        
        # Apply prompt template
        try:
            # Try to format using all columns in the row
            prompt = prompt_template.format(**row.to_dict())
        except KeyError:
            # Fallback to just the text column
            prompt = prompt_template.format(**{text_column: text})
        
        # Run inference with timing
        start_time = time.time()
        response = run_inference(model, tokenizer, prompt, kv_cache, model_name, max_new_tokens)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        # Store results
        result_item = {
            'index': idx,
            'input_text': text,
            'prompt': prompt,
            'response': response,
            'inference_time': inference_time
        }
        
        # Add all original columns from the dataset
        for col in df.columns:
            result_item[f'original_{col}'] = str(row[col])
        
        results.append(result_item)
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(time=f"{inference_time:.2f}s")
        
        # Print occasional updates
        if idx % 10 == 0 and idx > 0:
            avg_time = total_time / (idx + 1)
            print(f"\nProcessed {idx+1}/{len(df)} examples. Average time: {avg_time:.2f}s per example")
    
    progress_bar.close()
    
    # Calculate statistics
    avg_time = total_time / len(df) if df.shape[0] > 0 else 0
    stats = {
        'total_examples': len(df),
        'total_time': total_time,
        'average_time': avg_time,
    }
    
    return results, stats


def save_results(results, stats, output_dir, model_name, config_name, scheme):
    """Save inference results to files"""
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get shortened model and config names for filenames
    model_short = os.path.basename(model_name)
    config_short = os.path.basename(config_name).replace('.yaml', '')
    
    # Create output directory
    result_dir = os.path.join(output_dir, f"{model_short}_{config_short}_{scheme}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save results as JSON
    results_path = os.path.join(result_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save statistics
    stats_path = os.path.join(result_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Also save as CSV for easier analysis
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(result_dir, "inference_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to directory: {result_dir}")
    print(f"  - JSON results: {results_path}")
    print(f"  - CSV results: {csv_path}")
    print(f"  - Statistics: {stats_path}")
    
    return result_dir


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.debug)
    
    # Configure GPU environment
    gpu_message = set_gpu_environment(args.gpu_ids)
    logger.info(gpu_message)
    
    try:
        # Load model and tokenizer
        model, tokenizer, model_name = load_model_and_tokenizer(args.model_path)
        
        # Load KV configuration
        kv_config = load_kv_config(args.kv_config, args.scheme)
        
        # Load dataset
        df, text_column = load_dataset(args.dataset, args.text_column, args.max_rows)
        
        # Process dataset with KVTuner
        logger.info("Starting inference with KVTuner")
        results, stats = process_dataset(
            df, 
            model, 
            tokenizer, 
            model_name, 
            kv_config, 
            text_column, 
            args.prompt_template, 
            args.max_new_tokens
        )
        
        # Save results
        output_dir = save_results(
            results, 
            stats, 
            args.output_dir, 
            args.model_path, 
            args.kv_config, 
            args.scheme
        )
        
        logger.info(f"Inference complete! Processed {len(df)} examples in {stats['total_time']:.2f}s")
        logger.info(f"Average inference time: {stats['average_time']:.4f}s per example")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
