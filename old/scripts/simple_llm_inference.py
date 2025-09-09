#!/usr/bin/env python3
# filepath: /Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/old/simple_llm_inference.py
"""
Simple LLM Inference Script

This script focuses only on the essential LLM inference tasks:
1. Load a dataset
2. Initialize an LLM model
3. Run inference on each row using a specified template
4. Save the raw outputs and statistics

Usage:
    python simple_llm_inference.py --dataset path/to/dataset.csv --model model_name [options]
"""

import os
import pandas as pd
import argparse
import time
import json
import torch
import datetime
from vllm import LLM, SamplingParams
import torch.distributed as dist

def setup_directories(timestamp):
    """Create result directory with timestamp"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple_inference_results")
    result_dir = os.path.join(base_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def initialize_llm(model_name, tensor_parallel_size=None, gpu_ids=None):
    """Initialize the LLM with basic parameters"""
    # Set specific GPU devices if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using specific GPU IDs: {gpu_ids}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Default parameters
    llm_params = {
        "model": model_name,
        "trust_remote_code": True,
    }
    
    # Add tensor parallelism if specified
    if tensor_parallel_size is not None:
        llm_params["tensor_parallel_size"] = tensor_parallel_size
        print(f"Using tensor parallelism with {tensor_parallel_size} GPUs")
    
    try:
        llm = LLM(**llm_params)
        print(f"Model {model_name} loaded successfully")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Attempting to fallback to TinyLlama model...")
        llm_params["model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        llm = LLM(**llm_params)
        print("Fallback model loaded successfully")
    
    # Use simple sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=200
    )
    
    return llm, sampling_params

def process_dataset(df, llm, sampling_params, prompt_template, content_column=None):
    """Process each row in the dataset with LLM inference"""
    results = []
    start_time = time.time()
    
    # If content column not specified, try to find one or use the first column
    if content_column is None or content_column not in df.columns:
        content_candidates = [col for col in df.columns if any(term in col.lower() for term in 
                            ['content', 'text', 'review', 'description', 'body'])]
        if content_candidates:
            content_column = content_candidates[0]
            print(f"Using auto-detected content column: {content_column}")
        else:
            content_column = df.columns[0]
            print(f"Using first column as content: {content_column}")
    
    print(f"Using content column: {content_column}")
    
    # Process each row
    for index, row in df.iterrows():
        row_data = {}
        
        # Extract content value
        content = str(row[content_column])
        
        # Create prompt by replacing {content} with actual value
        prompt = prompt_template.replace("{content}", content)
        
        # Run inference
        inference_start = time.time()
        try:
            output = llm.generate(prompt, sampling_params)
            response = output[0].outputs[0].text if output and output[0].outputs else "Error: No response generated"
        except Exception as e:
            response = f"Error: {str(e)}"
        inference_end = time.time()
        
        # Store results
        row_data["row_index"] = index
        row_data["content"] = content
        row_data["prompt"] = prompt
        row_data["response"] = response
        row_data["inference_time"] = inference_end - inference_start
        
        results.append(row_data)
        
        # Print progress every 10 rows
        if index % 10 == 0:
            print(f"Processed {index} rows. Latest inference time: {row_data['inference_time']:.2f}s")
    
    end_time = time.time()
    
    # Calculate stats
    stats = {
        "total_rows": len(df),
        "total_time": end_time - start_time,
        "avg_time_per_row": (end_time - start_time) / len(df) if len(df) > 0 else 0
    }
    
    return results, stats

def save_results(results, stats, result_dir, filename_prefix):
    """Save inference results and stats to the specified directory"""
    # Save raw results as JSON
    results_path = os.path.join(result_dir, f"{filename_prefix}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save stats as JSON
    stats_path = os.path.join(result_dir, f"{filename_prefix}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save summary as text
    summary_path = os.path.join(result_dir, f"{filename_prefix}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("LLM Inference Summary\n")
        f.write("====================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Size: {stats['total_rows']} rows\n")
        f.write(f"Total Processing Time: {stats['total_time']:.2f} seconds\n")
        f.write(f"Average Time per Row: {stats['avg_time_per_row']:.4f} seconds\n")
        
    print(f"Results saved to {result_dir}")
    return results_path

def main():
    parser = argparse.ArgumentParser(description='Run simple LLM inference on a dataset.')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name or path to local model directory.')
    
    # Optional arguments
    parser.add_argument('--prompt_template', type=str, 
                        default='Based on this review: {content}, is this movie suitable for children under 12? Answer with Yes or No.',
                        help='Prompt template with {content} placeholder.')
    parser.add_argument('--content_column', type=str, default=None, 
                        help='Name of the column containing content to use in prompts.')
    parser.add_argument('--max_rows', type=int, default=None, 
                        help='Maximum number of rows to process (for testing).')
    parser.add_argument('--tp_size', type=int, default=None,
                        help='Tensor parallel size (number of GPUs to use).')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Specific GPU IDs to use, comma-separated (e.g., "0,1" or "6,7").')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Prefix for output files. Defaults to dataset filename without extension.')
    
    args = parser.parse_args()
    
    # Create timestamped result directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = setup_directories(timestamp)
    
    # Start timing
    start_time = time.time()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = pd.read_csv(args.dataset)
    
    # Get dataset filename without extension for output naming
    filename_prefix = args.output_prefix
    if filename_prefix is None:
        filename_prefix = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Apply max rows limit if specified
    if args.max_rows is not None:
        dataset = dataset.head(args.max_rows)
        print(f"Limited dataset to {args.max_rows} rows")
    
    # Initialize LLM
    llm, sampling_params = initialize_llm(
        model_name=args.model, 
        tensor_parallel_size=args.tp_size,
        gpu_ids=args.gpu_ids
    )
    
    # Process dataset with LLM inference
    print("Running LLM inference on dataset...")
    results, stats = process_dataset(
        dataset,
        llm,
        sampling_params,
        args.prompt_template,
        args.content_column
    )
    
    # End timing and update stats
    end_time = time.time()
    total_time = end_time - start_time
    stats["total_experiment_time"] = total_time
    
    print(f"Total experiment time: {total_time:.2f} seconds")
    
    # Save results
    output_path = save_results(results, stats, result_dir, filename_prefix)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    print(f"All results saved to {output_path}")
    print(f"Result directory: {result_dir}")

if __name__ == "__main__":
    main()
