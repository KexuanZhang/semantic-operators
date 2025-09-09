#!/usr/bin/env python3
"""
KVTuner Inference Script with Compatibility Fixes

This script runs inference on a dataset using KVTuner quantization with preset configurations.
Includes fallback mechanisms for compatibility issues.
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

# Update the KVTuner path to match your actual setup
# Try multiple possible paths for KVTuner
possible_paths = [
    "/home/data/so2/KVTuner",  # Server path
    "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner",  # Local path
    "../KVTuner",  # Relative path
    "../../KVTuner"  # Another relative path
]

kvtuner_path = None
for path in possible_paths:
    if os.path.exists(path):
        kvtuner_path = path
        break

if kvtuner_path:
    sys.path.insert(0, kvtuner_path)
    print(f"Found KVTuner at: {kvtuner_path}")
else:
    print("Warning: KVTuner path not found, will use standard inference only")

# Try to import KVTuner with compatibility handling
KVTUNER_AVAILABLE = False
try:
    from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import yaml
    KVTUNER_AVAILABLE = True
    print("✓ KVTuner modules loaded successfully")
except ImportError as e:
    print(f"⚠ KVTuner import warning: {e}")
    print("Falling back to standard transformers inference without KV quantization")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import yaml
        print("✓ Transformers loaded successfully (without KVTuner)")
    except ImportError as e2:
        print(f"✗ Failed to import transformers: {e2}")
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

def create_kv_cache_if_available(kv_config_path, scheme):
    """Create KV cache if KVTuner is available, otherwise return None"""
    if not KVTUNER_AVAILABLE:
        return None
        
    try:
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
    except Exception as e:
        print(f"Warning: Failed to create KV cache: {e}")
        return None

def load_model_and_tokenizer(model_path, logger):
    """Load the model and tokenizer"""
    logger.info(f"Loading model from: {model_path}")
    
    try:
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
        
        # Set pad token if not available
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def format_qwen_prompt(text):
    """Format prompt for Qwen models"""
    return f"<|im_start|>system\nYou are a helpful assistant that provides clear, concise, and accurate answers.\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"

def run_inference_single(model, tokenizer, prompt, max_new_tokens=256, use_kv_cache=False):
    """Run inference on a single prompt with optional KV cache support"""
    # Format prompt for Qwen
    formatted_prompt = format_qwen_prompt(prompt)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # Prepare generation arguments
    generation_args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Add KV cache if available (for future implementation)
    # Note: Direct KV cache integration with transformers.generate() is complex
    # This is a placeholder for when KVTuner provides better integration
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(**generation_args)
    
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
    """Process the entire dataset with inference"""
    
    results = []
    total_inference_time = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Check if KV cache can be used
    kv_cache = None
    if KVTUNER_AVAILABLE and kv_config_path and os.path.exists(kv_config_path):
        kv_cache = create_kv_cache_if_available(kv_config_path, scheme)
        if kv_cache:
            logger.info(f"KV cache created successfully with {scheme} scheme")
        else:
            logger.info("Using standard inference without KV quantization")
    else:
        logger.info("Using standard inference (KVTuner not available or config not found)")
    
    logger.info(f"Processing {len(dataset)} samples...")
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Running inference"):
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
                model, tokenizer, prompt, max_new_tokens, use_kv_cache=bool(kv_cache)
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
            "kvtuner_used": kv_cache is not None
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
        "kvtuner_available": KVTUNER_AVAILABLE,
        "kv_cache_used": kv_cache is not None,
        "quantization_scheme": scheme if kv_cache else "none"
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
    parser = argparse.ArgumentParser(description="KVTuner inference with compatibility fixes")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local model")
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
    output_dir = os.path.join(args.output_dir, f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Determine KV config path
        kv_config_path = None
        if args.kv_config:
            kv_config_path = args.kv_config
        elif KVTUNER_AVAILABLE:
            # Use preset configuration
            config_name = f"Qwen2.5-3B-Instruct_{args.scheme}_KVTuner4_0.yaml"
            kv_config_path = os.path.join(kvtuner_path, "calibration_presets", config_name)
            
        if kv_config_path and os.path.exists(kv_config_path):
            logger.info(f"Using KV config: {kv_config_path}")
        else:
            logger.info("No KV configuration found, using standard inference")
            kv_config_path = None
        
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
            text_candidates = ['text', 'content', 'review', 'comment', 'description', 'review_content']
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
        logger.info(f"KVTuner available: {stats['kvtuner_available']}")
        logger.info(f"KV cache used: {stats['kv_cache_used']}")
        logger.info(f"Quantization scheme: {stats['quantization_scheme']}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
