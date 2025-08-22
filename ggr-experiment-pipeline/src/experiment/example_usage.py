#!/usr/bin/env python3
"""
Example script demonstrating how to use the sentiment analysis experiment runner.
"""

from run_exp import SentimentAnalysisExperiment
import pandas as pd
import os

def create_sample_dataset():
    """Create a sample dataset for testing."""
    sample_data = [
        {"text": "I love this product! It's amazing and works perfectly.", "label": "positive"},
        {"text": "This is terrible. I hate it so much.", "label": "negative"},
        {"text": "The weather is beautiful today and I feel great!", "label": "positive"},
        {"text": "I'm so disappointed with this service. Very poor quality.", "label": "negative"},
        {"text": "The movie was fantastic! Best film I've seen this year.", "label": "positive"},
        {"text": "This is the worst experience I've ever had.", "label": "negative"},
        {"text": "Great customer service and fast delivery. Highly recommended!", "label": "positive"},
        {"text": "Don't waste your money on this. It's completely useless.", "label": "negative"}
    ]
    
    df = pd.DataFrame(sample_data)
    sample_path = "sample_sentiment_dataset.csv"
    df.to_csv(sample_path, index=False)
    print(f"Created sample dataset: {sample_path}")
    return sample_path

def main():
    """Example usage of the sentiment analysis experiment."""
    
    # Example model paths (you'll need to replace with actual model paths)
    example_model_paths = [
        "/path/to/your/llama2-7b",
        "/path/to/your/mistral-7b",
        "/path/to/your/local/model",
        "meta-llama/Llama-2-7b-chat-hf",  # Hugging Face model ID
    ]
    
    print("Sentiment Analysis Experiment Example")
    print("=" * 50)
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    
    print("\nTo run the experiment, use one of these commands:")
    print("\n1. Basic usage:")
    print(f"python run_exp.py --model-path /path/to/your/model --dataset-path {dataset_path}")
    
    print("\n2. Specify text column and results directory:")
    print(f"python run_exp.py --model-path /path/to/your/model --dataset-path {dataset_path} --text-column text --results-dir ./my_results")
    
    print("\n3. With custom experiment name:")
    print(f"python run_exp.py --model-path /path/to/your/model --dataset-path {dataset_path} --experiment-name my_sentiment_test")
    
    print("\n4. Full example with all parameters:")
    print(f"python run_exp.py \\")
    print(f"  --model-path /path/to/your/model \\")
    print(f"  --dataset-path {dataset_path} \\")
    print(f"  --text-column text \\")
    print(f"  --results-dir ./results \\")
    print(f"  --experiment-name sentiment_analysis_test \\")
    print(f"  --batch-size 1")
    
    print("\nExample with Hugging Face model:")
    print(f"python run_exp.py --model-path meta-llama/Llama-2-7b-chat-hf --dataset-path {dataset_path}")
    
    print("\nNote: Replace '/path/to/your/model' with the actual path to your local LLM model directory.")
    
    # Programmatic usage example (commented out to avoid requiring actual model)
    """
    # Example of programmatic usage:
    experiment = SentimentAnalysisExperiment(
        model_path="/path/to/your/model",
        results_dir="results"
    )
    
    experiment.load_model()
    
    results = experiment.process_dataset(
        dataset_path=dataset_path,
        text_column="text"
    )
    
    experiment.save_results(results, "example_experiment")
    experiment.cleanup()
    """

if __name__ == "__main__":
    main()
