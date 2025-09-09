#!/usr/bin/env python3
"""
KVTuner Inference Script for Qwen2.5-3B-Instruct

This script runs inference on a dataset using KVTuner quantization with preset configurations.
Supports both per-token and KiVi quantization schemes.

Usage:
    python kvtuner_inference.py --model_path /path/to/model --dataset dataset.csv
"""

import os
import sys
import argparse
import json
import csv
import time
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging

# Add KVTuner to Python path
sys.path.append("/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner")

try:
    from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import yaml
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure KVTuner is properly installed and accessible")
    sys.exit(1)

def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def set_gpu_environment(gpu_ids):
    """Set CUDA_VISIBLE_DEVICES environment variable"""
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return f"Using GPUs: {gpu_ids}"
    return "Using all available GPUs"

def load_kv_config(config_path):
    """Load KV configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"KV config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_kv_cache(kv_config_path, scheme):
    """Create KV cache with the specified configuration"""
    
    # Set axis configuration based on scheme
    if scheme == "kivi":
        axis_key = 1  # Per-channel for keys
        axis_value = 0  # Per-token for values
        q_group_size = 32
        residual_length = 32
    else:  # pertoken
        axis_key = 0  # Per-token for keys
        axis_value = 0  # Per-token for values
        q_group_size = -1
        residual_length = 0
    
    # Create cache configuration
    cache_config = FlexibleQuantizedCacheConfig(
        device="cuda",
        per_layer_quant=True,
        per_layer_config_path=kv_config_path,
        asym=True,
        axis_key=axis_key,
        axis_value=axis_value,
        q_group_size=q_group_size,
        residual_length=residual_length
    )
    
    # Create and return the KV cache
    return FlexibleVanillaQuantizedCache(cache_config=cache_config)

def load_model_and_tokenizer(model_path, logger):
    """Load the model and tokenizer"""
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def format_qwen_prompt(text):
    """Format prompt for Qwen models"""
    return f"<|im_start|>system\nYou are a helpful assistant that provides clear, concise, and accurate answers.\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"

def run_inference_single(model, tokenizer, kv_cache, prompt, max_new_tokens=256):
    """Run inference on a single prompt"""
    # Format prompt for Qwen
    formatted_prompt = format_qwen_prompt(prompt)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            past_key_values=kv_cache,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    
    # Decode output
    output_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    # Clean up output
    if "<|im_end|>" in output_text:
        output_text = output_text.split("<|im_end|>")[0].strip()
    
    return output_text.strip(), inputs.input_ids.shape[1], outputs.shape[1] - inputs.input_ids.shape[1]

def process_dataset(model, tokenizer, dataset, kv_config_path, scheme, text_column, 
                   prompt_template, max_new_tokens, logger):
    """Process the entire dataset with KVTuner inference"""
    
    results = []
    total_inference_time = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    logger.info(f"Processing {len(dataset)} samples...")
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Running inference"):
        # Create a fresh KV cache for each sample
        kv_cache = create_kv_cache(kv_config_path, scheme)
        
        # Get text from specified column
        if text_column not in row:
            logger.error(f"Column '{text_column}' not found in dataset row {idx}")
            continue
            
        text = str(row[text_column])
        
        # Format prompt using template
        try:
            if "{" in prompt_template and "}" in prompt_template:
                prompt = prompt_template.format(**{text_column: text})
            else:
                prompt = f"{prompt_template} {text}"
        except KeyError as e:
            logger.warning(f"Error formatting prompt at row {idx}: {e}")
            prompt = f"Analyze the following text: {text}"
        
        # Run inference with timing
        start_time = time.time()
        try:
            completion, prompt_tokens, completion_tokens = run_inference_single(
                model, tokenizer, kv_cache, prompt, max_new_tokens
            )
        except Exception as e:
            logger.error(f"Inference error at row {idx}: {e}")
            completion = f"Error: {str(e)}"
            prompt_tokens = 0
            completion_tokens = 0
            
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Update totals
        total_inference_time += inference_time
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        
        # Store result
        result = {
            "id": idx,
            "input_text": text,
            "prompt": prompt,
            "completion": completion,
            "inference_time": inference_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        
        # Add original row data
        for col, val in row.items():
            if col != text_column:
                result[f"original_{col}"] = val
                
        results.append(result)
        
        # Log progress every 10 samples
        if (idx + 1) % 10 == 0:
            avg_time = total_inference_time / (idx + 1)
            logger.info(f"Processed {idx+1} samples. Average time: {avg_time:.2f}s per sample")
    
    # Calculate final statistics
    stats = {
        "total_samples": len(results),
        "total_inference_time": total_inference_time,
        "average_inference_time": total_inference_time / len(results) if results else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "average_prompt_tokens": total_prompt_tokens / len(results) if results else 0,
        "average_completion_tokens": total_completion_tokens / len(results) if results else 0,
    }
    
    return results, stats

def save_results(results, stats, output_dir, logger):
    """Save results and statistics"""
    
    # Save results as JSON
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save results as CSV
    csv_file = os.path.join(output_dir, "results.csv")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
    
    # Save statistics
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"JSON results: {results_file}")
    logger.info(f"CSV results: {csv_file}")
    logger.info(f"Statistics: {stats_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="KVTuner inference with Qwen2.5-3B-Instruct")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local Qwen2.5-3B-Instruct model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset CSV file")
    
    # KVTuner configuration
    parser.add_argument("--scheme", type=str, default="pertoken", 
                        choices=["pertoken", "kivi"],
                        help="Quantization scheme (default: pertoken)")
    parser.add_argument("--kv_config", type=str, default=None,
                        help="Path to custom KV config file (if not provided, uses preset)")
    
    # Dataset configuration
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing text to process (default: text)")
    parser.add_argument("--prompt_template", type=str, 
                        default="Analyze the following text: {text}",
                        help="Template for prompts (use {text} as placeholder)")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Maximum number of rows to process")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate (default: 256)")
    
    # System configuration
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g., '0,1')")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    # Set GPU environment
    gpu_message = set_gpu_environment(args.gpu_ids)
    logger.info(gpu_message)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"kvtuner_inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Determine KV config path
        if args.kv_config:
            kv_config_path = args.kv_config
        else:
            # Use preset configuration
            config_name = f"Qwen2.5-3B-Instruct_{args.scheme}_KVTuner4_0.yaml"
            kv_config_path = f"/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner/calibration_presets/{config_name}"
        
        logger.info(f"Using KV config: {kv_config_path}")
        
        # Verify KV config exists
        if not os.path.exists(kv_config_path):
            raise FileNotFoundError(f"KV configuration not found: {kv_config_path}")
        
        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = pd.read_csv(args.dataset)
        
        if args.max_rows:
            dataset = dataset.head(args.max_rows)
            logger.info(f"Limited dataset to {args.max_rows} rows")
        
        logger.info(f"Dataset loaded: {len(dataset)} rows")
        logger.info(f"Available columns: {list(dataset.columns)}")
        
        # Verify text column exists
        if args.text_column not in dataset.columns:
            logger.warning(f"Column '{args.text_column}' not found. Available columns: {list(dataset.columns)}")
            # Try to find a suitable text column
            text_candidates = ['text', 'content', 'review', 'comment', 'description']
            found_column = None
            for candidate in text_candidates:
                if candidate in dataset.columns:
                    found_column = candidate
                    break
            
            if found_column:
                args.text_column = found_column
                logger.info(f"Using column '{args.text_column}' as text input")
            else:
                # Use first column
                args.text_column = dataset.columns[0]
                logger.info(f"Using first column '{args.text_column}' as text input")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, logger)
        
        # Process dataset
        logger.info(f"Starting inference with {args.scheme} scheme...")
        results, stats = process_dataset(
            model, tokenizer, dataset, kv_config_path, args.scheme,
            args.text_column, args.prompt_template, args.max_new_tokens, logger
        )
        
        # Save results
        save_results(results, stats, output_dir, logger)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("INFERENCE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Samples processed: {stats['total_samples']}")
        logger.info(f"Total time: {stats['total_inference_time']:.2f}s")
        logger.info(f"Average time per sample: {stats['average_inference_time']:.3f}s")
        logger.info(f"Total tokens: {stats['total_tokens']}")
        logger.info(f"Average tokens per sample: {stats['average_prompt_tokens'] + stats['average_completion_tokens']:.1f}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
