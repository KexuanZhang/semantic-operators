#!/usr/bin/env python3
"""
Example usage of the vLLM Server Experiment Script

This script demonstrates how to run sentiment analysis experiments
using the vLLM server with different models and datasets.
"""

import subprocess
import sys
from pathlib import Path

def run_example_experiment():
    """Run an example experiment with sample data"""
    
    # Define paths
    script_path = Path(__file__).parent / "src" / "experiment" / "server_exp.py"
    dataset_path = Path(__file__).parent / "sample_dataset.csv"
    
    # Example 1: Basic usage with a small model
    print("="*60)
    print("Running Example vLLM Server Experiment")
    print("="*60)
    print(f"Script: {script_path}")
    print(f"Dataset: {dataset_path}")
    print()
    
    # You can customize these parameters
    model_name = "microsoft/DialoGPT-small"  # Use a smaller model for testing
    result_dir = "example_results"
    
    cmd = [
        sys.executable, str(script_path),
        "--model", model_name,
        "--dataset", str(dataset_path),
        "--result-dir", result_dir,
        "--max-tokens", "15",
        "--temperature", "0.0",
        "--port", "8001",  # Use different port to avoid conflicts
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("This will:")
    print("1. Start a vLLM server with the specified model")
    print("2. Process each row in the dataset")
    print("3. Query the LLM for sentiment analysis")
    print("4. Collect vLLM metrics (KV cache, inference times)")
    print("5. Save results to the specified directory")
    print()
    
    # Uncomment the next line to actually run the experiment
    # subprocess.run(cmd)
    
    print("To run this experiment, uncomment the subprocess.run line in this script")
    print("or run the command directly in your terminal.")

if __name__ == "__main__":
    run_example_experiment()
