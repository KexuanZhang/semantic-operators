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
    Run a dummy test with three queries and print responses with stats
    """
    logger.info(f"🚀 Running dummy test with model: {model_path}")
    
    # Set GPU devices if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_ids}")
        
        # Parse GPU IDs and set tensor_parallel_size if not specified
        gpu_id_list = [int(gpu.strip()) for gpu in gpu_ids.split(',')]
        if tensor_parallel_size is None:
            tensor_parallel_size = len(gpu_id_list)
            logger.info(f"Set tensor_parallel_size={tensor_parallel_size} based on number of GPUs")
    
    # LLM parameters
    llm_kwargs = {
        'tensor_parallel_size': tensor_parallel_size or 1,
        'max_model_len': max_model_len,
        'gpu_memory_utilization': gpu_memory_utilization,
        'enable_prefix_caching': True,
        'log_stats_interval': 1  # Log stats after each query
    }
    
    # Initialize LLM
    logger.info(f"Initializing model with parameters: {llm_kwargs}")
    try:
        llm = create_enhanced_llm(model_path, **llm_kwargs)
        logger.info("✅ Model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        return
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=200,
        stop=["</s>", "<|endoftext|>"]
    )
    
    # Create three test queries with increasing similarity
    queries = [
        "Explain the concept of tensor parallelism in large language models.",
        "How does GPU memory utilization affect the performance of large language models?",
        "What is the impact of context length (max_model_len) on LLM inference speed and memory usage?"
    ]
    
    logger.info(f"Running inference on {len(queries)} test queries...")
    
    # Process each query individually to see stats for each
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}...")
        start_time = time.time()
        
        try:
            outputs = llm.generate([query], sampling_params, log_detailed_stats=True)
            response = outputs[0].outputs[0].text
            elapsed_time = time.time() - start_time
            
            # Print the response and stats
            print(f"\n{'='*60}")
            print(f"QUERY {i+1}: {query[:50]}...")
            print(f"{'='*60}")
            print(f"{response}")
            print(f"{'='*60}")
            
            # Get and print current stats
            stats = llm.get_current_stats()
            print(f"vLLM Stats:")
            
            # Print KV cache usage
            if 'engine_kv_cache_usage' in stats:
                print(f"• KV Cache Usage: {stats['engine_kv_cache_usage']:.2%}")
            
            # Print prefix cache hit rate
            if 'engine_prefix_cache_stats_hits' in stats and 'engine_prefix_cache_stats_queries' in stats:
                hits = stats['engine_prefix_cache_stats_hits']
                queries = stats['engine_prefix_cache_stats_queries']
                hit_rate = (hits / queries * 100) if queries > 0 else 0
                print(f"• Prefix Cache Hit Rate: {hit_rate:.2f}% ({hits}/{queries})")
            
            # Print token stats
            prompt_tokens = len(outputs[0].prompt_token_ids)
            output_tokens = len(outputs[0].outputs[0].token_ids)
            print(f"• Tokens: {prompt_tokens} prompt + {output_tokens} output = {prompt_tokens + output_tokens} total")
            
            # Print throughput
            if elapsed_time > 0:
                tokens_per_sec = output_tokens / elapsed_time
                print(f"• Throughput: {tokens_per_sec:.2f} tokens/sec")
            
            print(f"• Response time: {elapsed_time:.2f}s")
            print(f"{'='*60}\n")
            
            # Sleep briefly to make sure we get updated stats
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Error during generation: {e}")
            continue
    
    # Print final stats
    final_stats = llm.get_current_stats()
    logger.info("📊 Final vLLM Stats Summary:")
    
    # Print KV cache usage
    if 'engine_kv_cache_usage' in final_stats:
        logger.info(f"• Final KV Cache Usage: {final_stats['engine_kv_cache_usage']:.2%}")
    
    # Print prefix cache hit rate
    if 'engine_prefix_cache_stats_hits' in final_stats and 'engine_prefix_cache_stats_queries' in final_stats:
        hits = final_stats['engine_prefix_cache_stats_hits']
        queries = final_stats['engine_prefix_cache_stats_queries']
        hit_rate = (hits / queries * 100) if queries > 0 else 0
        logger.info(f"• Final Prefix Cache Hit Rate: {hit_rate:.2f}% ({hits}/{queries})")
    
    logger.info("✅ Dummy test completed successfully!")

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