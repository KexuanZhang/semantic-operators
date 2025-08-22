#!/usr/bin/env python3
"""
Example usage of the sentiment analysis experiment runner.

This script demonstrates how to use run_exp.py to perform sentiment analysis
on a dataset using a local LLM model.
"""

import os
import pandas as pd
from pathlib import Path

# Example: Create a simple test dataset
def create_sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        'text': [
            "I love this product! It's amazing and works perfectly.",
            "This is terrible. I hate it so much.",
            "It's okay, not great but not bad either.",
            "Absolutely fantastic! Best purchase ever!",
            "Worst experience of my life. Completely disappointed.",
            "Pretty good overall, would recommend to others."
        ],
        'id': [1, 2, 3, 4, 5, 6],
        'category': ['electronics', 'electronics', 'books', 'electronics', 'clothing', 'books']
    }
    
    df = pd.DataFrame(data)
    dataset_path = "sample_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Created sample dataset: {dataset_path}")
    return dataset_path

def main():
    """Run the example."""
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS EXPERIMENT EXAMPLE")
    print("="*60)
    
    print("\nTo run the sentiment analysis experiment, use the following command:")
    
    # Example command without GPU specification (uses default GPU)
    print("\n1. Basic usage (uses default GPU):")
    print("python src/experiment/run_exp.py \\")
    print("    --model-path /path/to/your/local/llm/model \\")
    print("    --dataset-path sample_dataset.csv \\")
    print("    --text-column text \\")
    print("    --results-dir results \\")
    print("    --experiment-name my_sentiment_test")
    
    # Example command with specific GPU device
    print("\n2. With specific GPU device:")
    print("python src/experiment/run_exp.py \\")
    print("    --model-path /path/to/your/local/llm/model \\")
    print("    --dataset-path sample_dataset.csv \\")
    print("    --text-column text \\")
    print("    --results-dir results \\")
    print("    --gpu-device cuda:0 \\")
    print("    --experiment-name my_sentiment_test_gpu0")
    
    # Example command using second GPU
    print("\n3. Using second GPU (if available):")
    print("python src/experiment/run_exp.py \\")
    print("    --model-path /path/to/your/local/llm/model \\")
    print("    --dataset-path sample_dataset.csv \\")
    print("    --text-column text \\")
    print("    --results-dir results \\")
    print("    --gpu-device cuda:1 \\")
    print("    --experiment-name my_sentiment_test_gpu1")
    
    print("\n" + "="*60)
    print("ARGUMENTS EXPLANATION:")
    print("="*60)
    
    print("\n--model-path: Path to your local LLM model directory")
    print("              (e.g., /data/models/llama2-7b-chat)")
    
    print("\n--dataset-path: Path to your CSV dataset file")
    print("                Must contain a text column for analysis")
    
    print("\n--text-column: Name of the column containing text to analyze")
    print("               Default: 'text'")
    
    print("\n--results-dir: Directory where results will be saved")
    print("               Default: 'results'")
    
    print("\n--gpu-device: GPU device to use for inference (OPTIONAL)")
    print("              Examples: 'cuda:0', 'cuda:1', 'cuda:2', etc.")
    print("              If not specified, uses default GPU")
    
    print("\n--experiment-name: Custom name for the experiment files (OPTIONAL)")
    print("                   If not specified, auto-generates timestamp-based name")
    
    print("\n" + "="*60)
    print("OUTPUT FILES:")
    print("="*60)
    
    print("\nThe experiment will create three files in the results directory:")
    print("1. {experiment_name}_results.json    - Detailed results with raw LLM responses")
    print("2. {experiment_name}_results.csv     - Results in CSV format for easy analysis")
    print("3. {experiment_name}_metadata.json   - Experiment metadata and timing info")
    
    print("\n" + "="*60)
    print("REQUIREMENTS:")
    print("="*60)
    
    print("\n- Python packages: vllm, torch, pandas, tqdm")
    print("- GPU with CUDA support")
    print("- Sufficient GPU memory for your model")
    print("- Local LLM model files")
    
    print(f"\nSample dataset created: {dataset_path}")
    print("You can now test the experiment runner with this sample data!")

if __name__ == "__main__":
    main()
