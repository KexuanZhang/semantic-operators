#!/usr/bin/env python3
"""
KVTuner + vLLM Integration Script

This script demonstrates how to use KVTuner quantization configurations
with vLLM for memory-efficient inference serving.

Usage:
    python kvtuner_vllm_inference.py --model MODEL_PATH --kvtuner-config CONFIG_PATH [OPTIONS]

Example:
    python kvtuner_vllm_inference.py \
        --model /home/data/so2/models/Qwen2.5-3B-Instruct \
        --kvtuner-config /home/data/so2/KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml \
        --quantization kvtuner \
        --tensor-parallel-size 1

Author: GitHub Copilot Assistant
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import torch

# Add KVTuner to path if needed
kvtuner_path = "/home/data/so2/KVTuner"
if kvtuner_path not in sys.path:
    sys.path.insert(0, kvtuner_path)

# Import vLLM components
from vllm import LLM, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KVTuner + vLLM Integration for Memory-Efficient Inference"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the model (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to the tokenizer (default: use model path)"
    )
    
    # KVTuner arguments
    parser.add_argument(
        "--kvtuner-config",
        type=str,
        default=None,
        help="Path to KVTuner preset configuration YAML file"
    )
    parser.add_argument(
        "--kvtuner-scheme",
        type=str,
        default="per_token",
        choices=["per_token", "per_channel"],
        help="KVTuner quantization scheme"
    )
    parser.add_argument(
        "--kvtuner-backend",
        type=str,
        default="vanilla",
        choices=["vanilla", "quanto", "hqq"],
        help="KVTuner backend"
    )
    
    # vLLM arguments
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (use 'kvtuner' for KVTuner)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model data type"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph optimization"
    )
    
    # Sampling arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    
    # Input arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run simple benchmark"
    )
    
    return parser.parse_args()


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from file."""
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts


def create_llm(args) -> LLM:
    """Create vLLM LLM instance with KVTuner configuration."""
    logger.info("Initializing vLLM with KVTuner integration...")
    
    # Prepare LLM arguments
    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    
    # Add max_model_len if specified
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    
    # Add quantization configuration
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
        
        # Add KVTuner-specific parameters if using KVTuner quantization
        if args.quantization == "kvtuner":
            llm_kwargs["kvtuner_config_path"] = args.kvtuner_config
            llm_kwargs["kvtuner_scheme"] = args.kvtuner_scheme
            llm_kwargs["kvtuner_backend"] = args.kvtuner_backend
            
            logger.info(f"KVTuner config: {args.kvtuner_config}")
            logger.info(f"KVTuner scheme: {args.kvtuner_scheme}")
            logger.info(f"KVTuner backend: {args.kvtuner_backend}")
    
    # Create LLM instance
    try:
        llm = LLM(**llm_kwargs)
        logger.info("Successfully initialized vLLM with KVTuner")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        raise


def run_inference(llm: LLM, prompts: List[str], args) -> List[str]:
    """Run inference with the LLM."""
    logger.info(f"Running inference on {len(prompts)} prompt(s)...")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    # Run inference
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    # Extract generated texts
    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    
    # Log timing information
    total_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    logger.info(f"Inference completed in {total_time:.2f}s")
    logger.info(f"Generated {total_tokens} tokens")
    logger.info(f"Throughput: {throughput:.2f} tokens/second")
    
    return generated_texts


def run_benchmark(llm: LLM, args):
    """Run simple benchmark."""
    logger.info("Running benchmark...")
    
    # Benchmark prompts
    benchmark_prompts = [
        "Explain the concept of artificial intelligence.",
        "Write a short story about a robot.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "How does machine learning work?",
    ]
    
    # Run multiple rounds
    num_rounds = 3
    total_time = 0
    total_tokens = 0
    
    for round_idx in range(num_rounds):
        logger.info(f"Benchmark round {round_idx + 1}/{num_rounds}")
        
        # Run inference
        start_time = time.time()
        generated_texts = run_inference(llm, benchmark_prompts, args)
        round_time = time.time() - start_time
        
        # Count tokens (approximate)
        round_tokens = sum(len(text.split()) * 1.3 for text in generated_texts)
        
        total_time += round_time
        total_tokens += round_tokens
        
        logger.info(f"Round {round_idx + 1} completed in {round_time:.2f}s")
    
    # Calculate averages
    avg_time = total_time / num_rounds
    avg_tokens = total_tokens / num_rounds
    avg_throughput = avg_tokens / avg_time if avg_time > 0 else 0
    
    logger.info("=" * 50)
    logger.info("BENCHMARK RESULTS:")
    logger.info(f"Average time per round: {avg_time:.2f}s")
    logger.info(f"Average tokens per round: {avg_tokens:.0f}")
    logger.info(f"Average throughput: {avg_throughput:.2f} tokens/second")
    logger.info("=" * 50)


def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.getLogger("vllm").setLevel(logging.DEBUG)
    
    # Validate KVTuner configuration
    if args.quantization == "kvtuner":
        if not args.kvtuner_config:
            logger.warning("KVTuner quantization specified but no config path provided")
        elif not os.path.exists(args.kvtuner_config):
            logger.error(f"KVTuner config file not found: {args.kvtuner_config}")
            sys.exit(1)
    
    try:
        # Create LLM instance
        llm = create_llm(args)
        
        # Run benchmark if requested
        if args.benchmark:
            run_benchmark(llm, args)
            return
        
        # Prepare prompts
        if args.prompts_file:
            prompts = load_prompts(args.prompts_file)
        else:
            prompts = [args.prompt]
        
        # Run inference
        generated_texts = run_inference(llm, prompts, args)
        
        # Display results
        print("\n" + "=" * 80)
        print("GENERATION RESULTS:")
        print("=" * 80)
        
        for i, (prompt, generated_text) in enumerate(zip(prompts, generated_texts)):
            print(f"\nPrompt {i + 1}:")
            print(f"Input: {prompt}")
            print(f"Output: {generated_text}")
            print("-" * 80)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
