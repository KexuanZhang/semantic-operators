#!/usr/bin/env python3
# filepath: /Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/old/run_complete_experiment.py
"""
Complete Experiment Runner Script

This script combines both dataset reordering and LLM inference by:
1. First applying the reordering algorithm to a dataset
2. Then running LLM inference on the reordered dataset
3. Saving all results in timestamped directories

Usage:
    python run_complete_experiment.py --dataset path/to/dataset.csv --model model_name [--reorder] [--gpu_ids "6,7"]
"""

import os
import argparse
import subprocess
import datetime
import sys

def main():
    parser = argparse.ArgumentParser(description='Run a complete reordering and LLM inference experiment.')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--content_column', type=str, default='review_content', 
                        help='Column name containing the main content for deduplication.')
    
    # Reordering options
    parser.add_argument('--reorder', action='store_true', help='Apply reordering algorithm to dataset.')
    parser.add_argument('--no_sort', action='store_true', help='Skip row sorting step when reordering.')
    parser.add_argument('--no_dedup', action='store_true', help='Skip deduplication step when reordering.')
    
    # LLM configuration
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name or path to local model directory.')
    parser.add_argument('--tp_size', type=int, default=None,
                        help='Tensor parallel size (number of GPUs to use).')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Specific GPU IDs to use, comma-separated (e.g., "0,1" or "6,7").')
    parser.add_argument('--prompt_template', type=str, 
                        default=None, help='Prompt template for LLM inference.')
    parser.add_argument('--include_columns', type=str, nargs='+', 
                        default=None, help='Columns to include in the prompt template.')
    
    # Execution options
    parser.add_argument('--max_rows', type=int, default=None, 
                        help='Maximum number of rows to process (for testing).')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip the inference step and only run reordering.')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Prefix for output files. Defaults to dataset filename without extension.')
                        
    args = parser.parse_args()
    
    # Get current timestamp for naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting experiment at {timestamp}")
    
    # Get dataset filename without extension for output naming
    if args.output_prefix is None:
        args.output_prefix = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Step 1: Reordering (if requested)
    dataset_path = args.dataset
    if args.reorder:
        print("\n=== STEP 1: REORDERING DATASET ===\n")
        
        # Construct reordering command
        reorder_cmd = [
            sys.executable,
            "dataset_reordering.py",
            "--dataset", args.dataset,
            "--output_prefix", args.output_prefix
        ]
        
        # Add optional arguments
        if args.content_column:
            reorder_cmd.extend(["--content_column", args.content_column])
        if args.no_sort:
            reorder_cmd.append("--no_sort")
        if args.no_dedup:
            reorder_cmd.append("--no_dedup")
        if args.max_rows:
            reorder_cmd.extend(["--max_rows", str(args.max_rows)])
        
        # Always add --reorder flag
        reorder_cmd.append("--reorder")
        
        print(f"Running: {' '.join(reorder_cmd)}")
        
        # Run reordering process
        try:
            reorder_process = subprocess.run(reorder_cmd, check=True, capture_output=True, text=True)
            print(reorder_process.stdout)
            
            # Extract the path to the reordered dataset from output
            for line in reorder_process.stdout.splitlines():
                if "Reordered dataset saved to" in line:
                    dataset_path = line.split("Reordered dataset saved to")[-1].strip()
                    break
            
            print(f"Reordered dataset path: {dataset_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Reordering failed with error: {e}")
            print(f"Error output: {e.stderr}")
            return 1
    
    # Step 2: LLM Inference (unless skipped)
    if not args.skip_inference:
        print("\n=== STEP 2: RUNNING LLM INFERENCE ===\n")
        
        # Construct inference command
        inference_cmd = [
            sys.executable,
            "llm_inference.py",
            "--dataset", dataset_path,
            "--model", args.model,
            "--output_prefix", args.output_prefix
        ]
        
        # Add optional arguments
        if args.gpu_ids:
            inference_cmd.extend(["--gpu_ids", args.gpu_ids])
        if args.tp_size:
            inference_cmd.extend(["--tp_size", str(args.tp_size)])
        if args.prompt_template:
            inference_cmd.extend(["--prompt_template", args.prompt_template])
        if args.include_columns:
            inference_cmd.extend(["--include_columns"] + args.include_columns)
        if args.max_rows:
            inference_cmd.extend(["--max_rows", str(args.max_rows)])
        
        print(f"Running: {' '.join(inference_cmd)}")
        
        # Run inference process
        try:
            inference_process = subprocess.run(inference_cmd, check=True, capture_output=False, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Inference failed with error: {e}")
            if hasattr(e, 'stderr'):
                print(f"Error output: {e.stderr}")
            return 2
    
    print("\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===\n")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
