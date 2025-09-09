#!/usr/bin/env python3
# filepath: /Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/old/reorder_inference_experiment.py

import os
import pandas as pd
import argparse
import time
import json
import torch
import datetime
from vllm import LLM, SamplingParams
import torch.distributed as dist
from collections import defaultdict

def setup_directories(timestamp):
    """Create result directory with timestamp"""
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def calculate_scores(df):
    """Calculate scores for each column based on average string length and cardinality."""
    column_scores = {}
    total_length = len(df)  # Total number of rows in the DataFrame
    avg_string_length = {col: df[col].astype(str).str.len().mean() for col in df.columns}

    for col in df.columns:
        cardinality = df[col].nunique()
        if cardinality > 0:  # Avoid division by zero
            score = avg_string_length[col] * (total_length / cardinality)
            column_scores[col] = score
    return column_scores

def reorder_columns(df):
    """Reorder columns based on precomputed scores."""
    column_scores = calculate_scores(df)  # Calculate scores for all columns once
    reordered_columns = []
    current_columns = list(df.columns)
    
    print("Original column order:")
    print(current_columns)

    while current_columns:
        # Select column with max score
        selected_column = max(column_scores, key=column_scores.get)
        reordered_columns.append(selected_column)
        current_columns.remove(selected_column)
        
        # Remove the selected column's score from the dictionary
        del column_scores[selected_column]

    print("Reordered columns:")
    print(reordered_columns)
    return df[reordered_columns]

def sort_rows_by_prefix(df):
    """Sort rows based on the concatenated string of all values in a row."""
    df['combined'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    sorted_df = df.sort_values(by='combined')
    sorted_df = sorted_df.drop(columns=['combined'])  # Drop the combined column after sorting
    return sorted_df

def deduplicate_rows(df, content_column='review_content'):
    """Remove duplicate rows based on the specified content column."""
    if content_column in df.columns:
        deduplicated_df = df.drop_duplicates(subset=[content_column])
        return deduplicated_df
    else:
        print(f"Warning: Column '{content_column}' not found. Skipping deduplication.")
        return df

def reorder_dataset(df, perform_sort=True, perform_dedup=True, content_column='review_content'):
    """Apply the full reordering algorithm to the dataset."""
    # Step 1: Reorder columns
    print("Reordering columns...")
    reordered_df = reorder_columns(df)
    
    # Step 2: Sort rows by prefix (optional)
    if perform_sort:
        print("Sorting rows by prefix...")
        reordered_df = sort_rows_by_prefix(reordered_df)
    
    # Step 3: Deduplicate rows (optional)
    if perform_dedup and content_column in df.columns:
        print(f"Deduplicating rows based on '{content_column}'...")
        reordered_df = deduplicate_rows(reordered_df, content_column)
    
    return reordered_df

def initialize_llm(model_name):
    """Initialize the LLM with vLLM"""
    # Set environment variable for cached outputs
    os.environ["VLLM_USE_CACHED_OUTPUTS"] = "True"
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print(f"Initializing LLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True
    )
    
    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95)
    
    return llm, sampling_params

def llm_inference(llm, sampling_params, prompt):
    """Run inference with the LLM and return the result"""
    output = llm.generate(prompt, sampling_params)
    return output[0].outputs[0].text

def process_dataset(df, llm, sampling_params, prompt_template, columns_to_include):
    """Process each row in the dataset with LLM inference"""
    results = []
    start_time = time.time()
    total_tokens = 0
    
    for index, row in df.iterrows():
        row_data = {}
        
        # Extract values for specified columns
        input_values = {col: str(row[col]) for col in columns_to_include if col in row}
        
        # Create prompt using template and row values
        prompt = prompt_template.format(**input_values)
        
        # Run inference
        inference_start = time.time()
        response = llm_inference(llm, sampling_params, prompt)
        inference_end = time.time()
        
        # Calculate tokens (approximate)
        prompt_tokens = len(prompt.split())
        response_tokens = len(response.split())
        total_tokens += prompt_tokens + response_tokens
        
        # Store results
        row_data.update(input_values)
        row_data["llm_response"] = response.strip()
        row_data["inference_time"] = inference_end - inference_start
        row_data["prompt_tokens"] = prompt_tokens
        row_data["response_tokens"] = response_tokens
        
        results.append(row_data)
        
        # Print progress every 10 rows
        if index % 10 == 0:
            print(f"Processed {index} rows. Latest inference time: {row_data['inference_time']:.2f}s")
    
    end_time = time.time()
    
    # Calculate stats
    stats = {
        "total_rows": len(df),
        "total_time": end_time - start_time,
        "avg_time_per_row": (end_time - start_time) / len(df),
        "total_tokens": total_tokens,
        "avg_tokens_per_row": total_tokens / len(df) if len(df) > 0 else 0
    }
    
    return results, stats

def save_results(results, stats, dataset, result_dir):
    """Save results and stats to the specified directory"""
    # Save processed results as JSON
    results_path = os.path.join(result_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save stats as JSON
    stats_path = os.path.join(result_dir, "experiment_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save reordered dataset as CSV
    dataset_path = os.path.join(result_dir, "processed_dataset.csv")
    dataset.to_csv(dataset_path, index=False)
    
    # Save summary as text
    summary_path = os.path.join(result_dir, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Experiment Summary\n")
        f.write("=================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Size: {stats['total_rows']} rows\n")
        f.write(f"Total Processing Time: {stats['total_time']:.2f} seconds\n")
        f.write(f"Average Time per Row: {stats['avg_time_per_row']:.4f} seconds\n")
        f.write(f"Total Tokens Processed: {stats['total_tokens']}\n")
        f.write(f"Average Tokens per Row: {stats['avg_tokens_per_row']:.2f}\n")
    
    print(f"Results saved to {result_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run a reordering and LLM inference experiment.')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--content_column', type=str, default='review_content', 
                        help='Column name containing the main content for deduplication.')
    
    # Reordering options
    parser.add_argument('--reorder', action='store_true', help='Apply reordering algorithm to dataset.')
    parser.add_argument('--no_sort', action='store_true', help='Skip row sorting step when reordering.')
    parser.add_argument('--no_dedup', action='store_true', help='Skip deduplication step when reordering.')
    
    # LLM configuration
    parser.add_argument('--model', type=str, default='Qwen/Qwen1.5-7B',
                        help='HuggingFace model to use for inference.')
    parser.add_argument('--prompt_template', type=str, 
                        default='Analyze whether this movie would be suitable for kids based on {movie_info} and {review_content}.',
                        help='Prompt template for LLM inference.')
    parser.add_argument('--include_columns', type=str, nargs='+', 
                        default=['movie_title', 'movie_info', 'review_content'],
                        help='Columns to include in the prompt template.')
    
    # Execution options
    parser.add_argument('--max_rows', type=int, default=None, 
                        help='Maximum number of rows to process (for testing).')
    
    args = parser.parse_args()
    
    # Create timestamped result directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = setup_directories(timestamp)
    
    # Start timing
    start_time = time.time()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = pd.read_csv(args.dataset)
    
    # Apply max rows limit if specified
    if args.max_rows is not None:
        dataset = dataset.head(args.max_rows)
        print(f"Limited dataset to {args.max_rows} rows")
    
    # Apply reordering if requested
    if args.reorder:
        print("Applying reordering algorithm...")
        dataset = reorder_dataset(
            dataset, 
            perform_sort=not args.no_sort,
            perform_dedup=not args.no_dedup,
            content_column=args.content_column
        )
    else:
        print("Skipping reordering as per command line argument")
    
    # Initialize LLM
    llm, sampling_params = initialize_llm(args.model)
    
    # Process dataset with LLM inference
    print("Running LLM inference on dataset...")
    results, stats = process_dataset(
        dataset,
        llm,
        sampling_params,
        args.prompt_template,
        args.include_columns
    )
    
    # End timing and update stats
    end_time = time.time()
    total_time = end_time - start_time
    stats["total_experiment_time"] = total_time
    
    print(f"Total experiment time: {total_time:.2f} seconds")
    
    # Save results
    save_results(results, stats, dataset, result_dir)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
