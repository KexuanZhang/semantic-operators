#!/usr/bin/env python3
"""
KVTuner-Compatible Inference Script

This script runs inference on a dataset using either:
1. KVTuner's FlexibleQuantizedCache (if available)
2. Transformers' built-in QuantizedCache (as fallback)
3. Standard inference (if quantization fails)

Based on the flexible_quant_example.py usage pattern.
"""

import os
import sys
import argparse
import json
import time
import torch
import pandas as pd
import warnings
from datetime import datetime
from tqdm import tqdm
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try to find KVTuner paths
possible_kvtuner_paths = [
    "/home/data/so2/KVTuner",
    "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner",
    "../KVTuner",
    "../../KVTuner"
]

kvtuner_path = None
for path in possible_kvtuner_paths:
    if os.path.exists(path):
        kvtuner_path = path
        sys.path.insert(0, path)
        break

# Import transformers first (always available)
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import QuantizedCacheConfig, HQQQuantizedCache, QuantoQuantizedCache
    TRANSFORMERS_CACHE_AVAILABLE = True
except ImportError:
    TRANSFORMERS_CACHE_AVAILABLE = False

# Try to import KVTuner modules
KVTUNER_AVAILABLE = False
if kvtuner_path:
    try:
        from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache, FlexibleHQQQuantizedCache
        KVTUNER_AVAILABLE = True
        print(f"✓ KVTuner found at: {kvtuner_path}")
    except ImportError as e:
        print(f"⚠ KVTuner import failed: {e}")

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

def create_kvtuner_cache(scheme, config_path=None, nbits_key=4, nbits_value=4):
    """Create KVTuner cache following the flexible_quant_example.py pattern"""
    if not KVTUNER_AVAILABLE:
        return None
        
    try:
        if config_path and os.path.exists(config_path):
            # Use per-layer configuration from file (like in the example)
            cache_config = FlexibleQuantizedCacheConfig(
                nbits_key=nbits_key, 
                nbits_value=nbits_value, 
                asym=True, 
                axis_key=0 if scheme == "pertoken" else 1, 
                axis_value=0, 
                device='cuda', 
                per_layer_config=True, 
                per_layer_config_path=config_path
            )
        else:
            # Use uniform configuration (like in the example)
            cache_config = FlexibleQuantizedCacheConfig(
                nbits_key=nbits_key, 
                nbits_value=nbits_value, 
                asym=True, 
                axis_key=0 if scheme == "pertoken" else 1, 
                axis_value=0, 
                device='cuda', 
                q_group_size=-1 if scheme == "pertoken" else 32
            )
        
        # Create cache using FlexibleVanillaQuantizedCache (as in example)
        past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)
        return past_key_values
        
    except Exception as e:
        print(f"Warning: Failed to create KVTuner cache: {e}")
        return None

def create_transformers_cache(nbits=4):
    """Create transformers built-in quantized cache as fallback"""
    if not TRANSFORMERS_CACHE_AVAILABLE:
        return None
        
    try:
        # Try HQQQuantizedCache first (similar to KVTuner's approach)
        cache_config = QuantizedCacheConfig(nbits=nbits, axis_key=0, axis_value=0, device='cuda')
        past_key_values = HQQQuantizedCache(cache_config=cache_config)
        return past_key_values
    except Exception as e:
        print(f"Warning: Failed to create transformers cache: {e}")
        return None

def load_model_and_tokenizer(model_path, logger):
    """Load model and tokenizer following the example pattern"""
    logger.info(f"Loading model from: {model_path}")
    
    # Load model with similar settings as in flexible_quant_example.py
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
    
    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info("✓ Model and tokenizer loaded successfully")
    return model, tokenizer

def format_qwen_prompt(text):
    """Format prompt for Qwen models"""
    return f"<|im_start|>system\nYou are a helpful assistant that provides clear, concise, and accurate answers.\n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"

def run_inference_single(model, tokenizer, prompt, past_key_values=None, max_new_tokens=256):
    """Run inference with optional KV cache - following the example pattern"""
    
    # Format prompt for Qwen
    formatted_prompt = format_qwen_prompt(prompt)
    
    # Tokenize input (following the example)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Run inference with past_key_values if provided (like in the example)
    with torch.no_grad():
        if past_key_values is not None:
            # Use the cache (like in flexible_quant_example.py)
            outputs = model.generate(
                inputs.input_ids, 
                past_key_values=past_key_values, 
                use_cache=True, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            # Standard inference without cache
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
    
    # Decode output (following the example pattern)
    output_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Clean up Qwen-specific tokens
    if "<|im_end|>" in output_text:
        output_text = output_text.split("<|im_end|>")[0].strip()
    
    return output_text.strip(), inputs.input_ids.shape[1], outputs.shape[1] - inputs.input_ids.shape[1]

def process_dataset(model, tokenizer, dataset, cache_type, scheme, kv_config_path, 
                   text_column, prompt_template, max_new_tokens, logger):
    """Process the entire dataset with inference"""
    
    results = []
    total_inference_time = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Create cache based on type
    past_key_values = None
    cache_info = "No quantization"
    
    if cache_type == "kvtuner":
        past_key_values = create_kvtuner_cache(scheme, kv_config_path)
        if past_key_values:
            cache_info = f"KVTuner {scheme} quantization"
        else:
            logger.warning("KVTuner cache creation failed, falling back to transformers cache")
            past_key_values = create_transformers_cache()
            cache_info = "Transformers HQQ quantization (fallback)" if past_key_values else "No quantization"
    elif cache_type == "transformers":
        past_key_values = create_transformers_cache()
        cache_info = "Transformers HQQ quantization" if past_key_values else "No quantization"
    
    logger.info(f"Cache configuration: {cache_info}")
    logger.info(f"Processing {len(dataset)} samples...")
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Running inference"):
        # Get text from specified column
        if text_column not in row:
            logger.error(f"Column '{text_column}' not found in row {idx}")
            continue
            
        text = str(row[text_column])
        
        # Format prompt using template
        try:
            if "{" in prompt_template and "}" in prompt_template:
                prompt = prompt_template.format(**{text_column: text})
            else:
                prompt = f"{prompt_template} {text}"
        except Exception as e:
            logger.warning(f"Error formatting prompt at row {idx}: {e}")
            prompt = f"Analyze the following text: {text}"
        
        # For KVTuner, create a fresh cache for each inference (if using per-sample caching)
        current_cache = past_key_values
        if cache_type == "kvtuner" and past_key_values is not None:
            # For KVTuner, we can reuse the same cache or create fresh ones
            # Following the example, we'll reuse the same cache
            current_cache = past_key_values
        
        # Run inference with timing
        start_time = time.time()
        try:
            completion, prompt_tokens, completion_tokens = run_inference_single(
                model, tokenizer, prompt, current_cache, max_new_tokens
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
            "cache_type": cache_info
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
        "cache_configuration": cache_info,
        "kvtuner_available": KVTUNER_AVAILABLE,
        "transformers_cache_available": TRANSFORMERS_CACHE_AVAILABLE
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

def parse_args():
    parser = argparse.ArgumentParser(description="KVTuner-compatible inference script")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset CSV file")
    
    # Cache configuration
    parser.add_argument("--scheme", type=str, default="pertoken", 
                        choices=["pertoken", "kivi"],
                        help="Quantization scheme (default: pertoken)")
    parser.add_argument("--cache_type", type=str, default="auto",
                        choices=["kvtuner", "transformers", "none", "auto"],
                        help="Cache type to use (default: auto)")
    parser.add_argument("--kv_config", type=str, default=None,
                        help="Path to KV config file (for KVTuner)")
    parser.add_argument("--nbits", type=int, default=4,
                        help="Number of bits for quantization (default: 4)")
    
    # Dataset configuration  
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing text to process")
    parser.add_argument("--prompt_template", type=str, 
                        default="Analyze the following text: {text}",
                        help="Template for prompts")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Maximum number of rows to process")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    
    # System configuration
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use")
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
    
    # Print availability status
    logger.info(f"KVTuner available: {KVTUNER_AVAILABLE}")
    logger.info(f"Transformers cache available: {TRANSFORMERS_CACHE_AVAILABLE}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Determine cache type
        cache_type = args.cache_type
        if cache_type == "auto":
            if KVTUNER_AVAILABLE:
                cache_type = "kvtuner"
            elif TRANSFORMERS_CACHE_AVAILABLE:
                cache_type = "transformers"  
            else:
                cache_type = "none"
        
        logger.info(f"Using cache type: {cache_type}")
        
        # Determine KV config path
        kv_config_path = args.kv_config
        if not kv_config_path and kvtuner_path and cache_type == "kvtuner":
            # Use preset configuration
            config_name = f"Qwen2.5-3B-Instruct_{args.scheme}_KVTuner{args.nbits}_0.yaml"
            kv_config_path = os.path.join(kvtuner_path, "calibration_presets", config_name)
            if os.path.exists(kv_config_path):
                logger.info(f"Using preset config: {config_name}")
            else:
                logger.info(f"Preset config not found: {config_name}")
                kv_config_path = None
        
        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = pd.read_csv(args.dataset)
        
        if args.max_rows:
            dataset = dataset.head(args.max_rows)
            logger.info(f"Limited dataset to {args.max_rows} rows")
        
        logger.info(f"Dataset loaded: {len(dataset)} rows")
        
        # Verify text column exists
        if args.text_column not in dataset.columns:
            logger.warning(f"Column '{args.text_column}' not found")
            # Auto-detect text column
            text_candidates = ['text', 'content', 'review', 'comment', 'description', 'review_content']
            for candidate in text_candidates:
                if candidate in dataset.columns:
                    args.text_column = candidate
                    logger.info(f"Using column '{args.text_column}' as text input")
                    break
            else:
                args.text_column = dataset.columns[0]
                logger.info(f"Using first column '{args.text_column}' as fallback")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, logger)
        
        # Process dataset
        logger.info(f"Starting inference...")
        results, stats = process_dataset(
            model, tokenizer, dataset, cache_type, args.scheme, kv_config_path,
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
        logger.info(f"Cache configuration: {stats['cache_configuration']}")
        logger.info(f"Total tokens: {stats['total_tokens']}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
