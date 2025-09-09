#!/usr/bin/env python3
# Script to run inference with KVTuner using a specific KV configuration
# Usage: python run_kvtuner_inference.py --model_name "meta-llama/Meta-Llama-3-8B" --dataset dataset.jsonl --kv_config ./configs/model_config.yaml

import os
import sys
import argparse
import torch
import json
import time
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Add path to KVTuner
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "KVTuner"))

try:
    from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure KVTuner is properly installed and accessible")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with KVTuner")
    
    # Model and data parameters
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Path to local model or HF model name")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to dataset file (JSON or JSONL)")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory to save results")
    
    # KVTuner configuration
    parser.add_argument("--kv_config", type=str, 
                        help="Path to KV config YAML file (if using per-layer config)")
    parser.add_argument("--scheme", type=str, default="pertoken", 
                        choices=["pertoken", "kivi", "none"],
                        help="Quantization scheme: pertoken, kivi, or none (for no quantization)")
    parser.add_argument("--key_bits", type=int, default=4, 
                        help="Number of bits for keys (when using uniform quantization)")
    parser.add_argument("--value_bits", type=int, default=4, 
                        help="Number of bits for values (when using uniform quantization)")
    
    # Inference parameters
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for inference")
    parser.add_argument("--prompt_key", type=str, default="prompt", 
                        help="Key in dataset entries to use as prompt")
    parser.add_argument("--sample_limit", type=int, default=None, 
                        help="Limit the number of samples to process")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1')")
    
    # Logging parameters
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    
    return parser.parse_args()

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("kvtuner_inference")

def set_gpu_environment(gpu_ids):
    """Set CUDA_VISIBLE_DEVICES environment variable"""
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return f"Using GPUs: {gpu_ids}"
    return "Using all available GPUs"

def load_dataset(path, sample_limit=None, prompt_key="prompt"):
    """Load dataset from JSON or JSONL file"""
    path = Path(path)
    data = []
    
    if path.suffix.lower() == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, dict):
                # Convert dict to list of samples if needed
                if "samples" in data:
                    data = data["samples"]
                else:
                    # If it's a single sample, wrap it in a list
                    data = [data]
    
    # Validate data format
    for i, item in enumerate(data):
        if prompt_key not in item:
            raise ValueError(f"Item {i} in dataset does not contain the prompt key '{prompt_key}'")
    
    # Apply sample limit if specified
    if sample_limit is not None and sample_limit > 0:
        data = data[:sample_limit]
        
    return data

def configure_kv_cache(args):
    """Configure KV cache based on arguments"""
    
    # If no quantization is requested, return None
    if args.scheme == "none":
        return None
    
    # Set up axis configuration based on scheme
    if args.scheme == "kivi":
        axis_key = 1  # Per-channel for keys in KiVi
        axis_value = 0  # Per-token for values in KiVi
        q_group_size = 32
        residual_length = 32
    else:  # pertoken
        axis_key = 0  # Per-token for keys
        axis_value = 0  # Per-token for values
        q_group_size = -1
        residual_length = 0
    
    # If using per-layer configuration from file
    if args.kv_config:
        cache_config = FlexibleQuantizedCacheConfig(
            device="cuda",
            per_layer_quant=True,
            per_layer_config_path=args.kv_config,
            asym=True,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length
        )
    else:
        # Using uniform quantization
        cache_config = FlexibleQuantizedCacheConfig(
            nbits_key=args.key_bits,
            nbits_value=args.value_bits,
            asym=True,
            axis_key=axis_key,
            axis_value=axis_value,
            device="cuda",
            q_group_size=q_group_size,
            residual_length=residual_length
        )
    
    # Create and return the KV cache
    return FlexibleVanillaQuantizedCache(cache_config=cache_config)

def run_inference(model, tokenizer, dataset, kv_cache_creator, args, logger):
    """Run inference on the dataset using the model and KV cache"""
    results = []
    inference_times = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Running inference")):
        # Create a fresh KV cache for each sample to avoid cross-contamination
        kv_cache = kv_cache_creator() if kv_cache_creator else None
        
        # Get prompt from dataset
        prompt = item[args.prompt_key]
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Run inference with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                past_key_values=kv_cache,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False
            )
        end_time = time.time()
        
        # Calculate time
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # Decode output
        output_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Store results
        result = {
            "id": idx,
            "prompt": prompt,
            "completion": output_text,
            "inference_time": inference_time,
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
            "total_tokens": outputs.shape[1],
        }
        
        # Add any other fields from the dataset
        for key, value in item.items():
            if key != args.prompt_key:
                result[f"input_{key}"] = value
        
        results.append(result)
        
        # Log progress every 10 samples
        if (idx + 1) % 10 == 0 or idx == 0:
            avg_time = sum(inference_times[-10:]) / min(10, len(inference_times[-10:]))
            logger.info(f"Processed {idx+1}/{len(dataset)} samples. "
                       f"Last sample: {inference_time:.2f}s. "
                       f"Recent average: {avg_time:.2f}s")
    
    return results, inference_times

def main():
    args = parse_args()
    logger = setup_logging(args.debug)
    
    # Set GPU environment
    gpu_message = set_gpu_environment(args.gpu_ids)
    logger.info(gpu_message)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            use_fast=False, 
            trust_remote_code=True
        )
        
        # Load dataset
        logger.info(f"Loading dataset from: {args.dataset}")
        dataset = load_dataset(args.dataset, args.sample_limit, args.prompt_key)
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Configure KV cache
        logger.info("Configuring KV cache")
        if args.scheme == "none":
            logger.info("Running without KV cache quantization (standard inference)")
            kv_cache_creator = None
        else:
            config_type = "per-layer" if args.kv_config else f"uniform K{args.key_bits}V{args.value_bits}"
            scheme_name = "KiVi" if args.scheme == "kivi" else "Per-token"
            logger.info(f"Using {scheme_name} quantization with {config_type} configuration")
            
            # Create a factory function that returns a new KV cache each time
            kv_cache = configure_kv_cache(args)
            kv_cache_creator = lambda: configure_kv_cache(args)
        
        # Run inference
        logger.info("Starting inference...")
        results, inference_times = run_inference(model, tokenizer, dataset, kv_cache_creator, args, logger)
        
        # Save results
        results_path = os.path.join(output_dir, "results.jsonl")
        with open(results_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        median_time = np.median(inference_times)
        p90_time = np.percentile(inference_times, 90)
        
        # Save metadata
        metadata = {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "scheme": args.scheme,
            "kv_config": args.kv_config if args.kv_config else f"K{args.key_bits}V{args.value_bits} uniform",
            "samples_processed": len(results),
            "average_inference_time": float(avg_time),
            "median_inference_time": float(median_time),
            "p90_inference_time": float(p90_time),
            "total_inference_time": float(sum(inference_times)),
            "timestamp": timestamp,
            "gpu_ids": args.gpu_ids
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Inference complete!")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Statistics:")
        logger.info(f"  - Samples processed: {len(results)}")
        logger.info(f"  - Average time per sample: {avg_time:.4f}s")
        logger.info(f"  - Median time per sample: {median_time:.4f}s")
        logger.info(f"  - P90 time per sample: {p90_time:.4f}s")
        logger.info(f"  - Total inference time: {sum(inference_times):.2f}s")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
