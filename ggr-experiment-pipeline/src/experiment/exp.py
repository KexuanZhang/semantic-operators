#!/usr/bin/env python3
"""
Dummy Test Script for vLLM with Stats Logging
This script creates three simple queries and runs them through a model,
printing both LLM responses and vLLM statistics for each response.
"""

import argparse
import time
import os
import sys
import torch
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local vLLM setup with flexible path handling
try:
    # Try direct import first
    logger.info("Trying direct import of use_local_vllm...")
    try:
        from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm
    except ImportError:
        # If direct import fails, search for module
        logger.info("Direct import failed, searching for module...")
        
        # Search for module in potential paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            current_dir,
            os.path.dirname(current_dir),
            os.path.dirname(os.path.dirname(current_dir)),
            '/home/data/so/semantic-operators/ggr-experiment-pipeline',
            '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline'
        ]
        
        module_found = False
        for path in potential_paths:
            if os.path.exists(os.path.join(path, 'use_local_vllm.py')):
                logger.info(f"Found use_local_vllm.py in: {path}")
                sys.path.insert(0, path)
                from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm
                module_found = True
                break
        
        if not module_found:
            raise ImportError("Could not find use_local_vllm.py in any of the expected paths")
except Exception as e:
    logger.error(f"Failed to import use_local_vllm: {e}")
    sys.exit(1)

# Initialize local vLLM with stats logging
logger.info("Initializing local vLLM with stats logging...")
initialize_experiment_with_local_vllm()

# Import vLLM components
from vllm import SamplingParams
from vllm.offline_llm_with_stats import OfflineLLMWithStats

def run_dummy_test(model_path, gpu_ids=None, tensor_parallel_size=None, gpu_memory_utilization=0.7, max_model_len=2048):
    """
    Run a simple test with three queries and print responses with key stats
    """
    logger.info(f"🚀 Running test with model: {model_path}")
    
    # Set GPU devices if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_ids}")
        
        # Parse GPU IDs and set tensor_parallel_size if not specified
        gpu_id_list = [int(gpu.strip()) for gpu in gpu_ids.split(',')]
        if tensor_parallel_size is None:
            tensor_parallel_size = len(gpu_id_list)
            logger.info(f"Set tensor_parallel_size={tensor_parallel_size} based on number of GPUs")
    
    # Initialize LLM with specified parameters
    llm_kwargs = {
        'tensor_parallel_size': tensor_parallel_size or 1,
        'max_model_len': max_model_len,
        'gpu_memory_utilization': gpu_memory_utilization,
        'enable_prefix_caching': True,
    }
    
    logger.info(f"Initializing model with parameters: {llm_kwargs}")
    llm = create_enhanced_llm(model_path, **llm_kwargs)
    
    # Simple sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=200,
    )
    
    # Three test queries
    queries = [
        "Explain the concept of tensor parallelism in large language models.",
        "How does GPU memory utilization affect the performance of large language models?",
        "What is the impact of context length (max_model_len) on LLM inference speed and memory usage?"
    ]
    
    # Run each query
    for i, query in enumerate(queries):
        logger.info(f"Running query {i+1}/3...")
        
        # Generate response
        outputs = llm.generate([query], sampling_params, log_detailed_stats=True)
        response = outputs[0].outputs[0].text
        
        # Get stats
        stats = llm.get_current_stats()
        
        # Print response and key stats
        print(f"\n{'='*60}")
        print(f"QUERY {i+1}: {query}")
        print(f"{'='*60}")
        print(f"{response}")
        print(f"{'='*60}")
        print("Stats:")
        
        # KV Cache Usage
        kv_cache = stats.get('engine_kv_cache_usage', 0) * 100
        print(f"• KV Cache Usage: {kv_cache:.2f}%")
        
        # Prefix Cache Hit Rate
        hits = stats.get('engine_prefix_cache_stats_hits', 0)
        queries_count = stats.get('engine_prefix_cache_stats_queries', 0)
        hit_rate = (hits / queries_count * 100) if queries_count > 0 else 0
        print(f"• Prefix Cache Hit Rate: {hit_rate:.2f}% ({hits}/{queries_count})")
        print(f"{'='*60}\n")
        
        # Brief pause between queries
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Run a dummy test with vLLM stats logging")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat",
                        help="Model path or name")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3' or '4,5,6,7')")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Number of GPUs to use for tensor parallelism (calculated from --gpus if not specified)")
    parser.add_argument("--gpu-memory", type=float, default=0.7,
                        help="GPU memory utilization (0.0 to 1.0, default: 0.7)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum model context length (default: 2048)")
    
    args = parser.parse_args()
    
    # Run the dummy test
    run_dummy_test(
        model_path=args.model,
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len
    )

if __name__ == "__main__":
    main()