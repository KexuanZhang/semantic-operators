#!/usr/bin/env python3
# filepath: /Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/old/dataset_reordering.py
"""
Dataset Reordering Script

This script applies a reordering algorithm to a dataset:
1. Reorders columns based on computed scores
2. Optionally sorts rows by prefix
3. Optionally deduplicates rows based on a content column

Usage:
    python dataset_reordering.py --dataset path/to/dataset.csv --reorder [--no_sort] [--no_dedup] [--content_column col_name]
"""

import os
import pandas as pd
import argparse
import time
import json
import datetime
from collections import defaultdict

def setup_directories(timestamp):
    """Create result directory with timestamp"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reordered")
    result_dir = os.path.join(base_dir, timestamp)
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

def save_results(dataset, result_dir, original_dataset, filename_prefix):
    """Save reordered dataset and comparison stats"""
    # Save reordered dataset as CSV
    dataset_path = os.path.join(result_dir, f"{filename_prefix}_reordered.csv")
    dataset.to_csv(dataset_path, index=False)
    print(f"Reordered dataset saved to {dataset_path}")
    
    # Generate a basic diff report
    reordering_stats = {
        "original_columns": original_dataset.columns.tolist(),
        "processed_columns": dataset.columns.tolist(),
        "columns_reordered": original_dataset.columns.tolist() != dataset.columns.tolist(),
        "original_row_count": len(original_dataset),
        "processed_row_count": len(dataset),
        "rows_changed": len(original_dataset) != len(dataset),
    }
    
    # Save reordering stats
    reordering_path = os.path.join(result_dir, f"{filename_prefix}_reordering_stats.json")
    with open(reordering_path, 'w') as f:
        json.dump(reordering_stats, f, indent=2)
    
    # Save summary as text
    summary_path = os.path.join(result_dir, f"{filename_prefix}_reordering_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Dataset Reordering Summary\n")
        f.write("=========================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original dataset size: {len(original_dataset)} rows, {len(original_dataset.columns)} columns\n")
        f.write(f"Reordered dataset size: {len(dataset)} rows, {len(dataset.columns)} columns\n\n")
        
        f.write("Reordering Information:\n")
        f.write(f"- Columns reordered: {'Yes' if reordering_stats['columns_reordered'] else 'No'}\n")
        if reordering_stats['columns_reordered']:
            f.write("- Original column order (first 5): " + ", ".join(original_dataset.columns[:5]) + "\n")
            f.write("- Reordered column order (first 5): " + ", ".join(dataset.columns[:5]) + "\n")
        
        f.write(f"- Rows changed: {'Yes' if reordering_stats['rows_changed'] else 'No'}\n")
        if reordering_stats['rows_changed']:
            rows_removed = reordering_stats['original_row_count'] - reordering_stats['processed_row_count'] 
            f.write(f"- Rows removed: {rows_removed} ({rows_removed/reordering_stats['original_row_count']*100:.1f}%)\n")
    
    print(f"Reordering summary saved to {summary_path}")
    return dataset_path

def main():
    parser = argparse.ArgumentParser(description='Apply reordering algorithm to a dataset.')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--content_column', type=str, default='review_content', 
                        help='Column name containing the main content for deduplication.')
    
    # Reordering options
    parser.add_argument('--reorder', action='store_true', help='Apply reordering algorithm to dataset.')
    parser.add_argument('--no_sort', action='store_true', help='Skip row sorting step when reordering.')
    parser.add_argument('--no_dedup', action='store_true', help='Skip deduplication step when reordering.')
    parser.add_argument('--output_prefix', type=str, default='dataset',
                        help='Prefix for output files')
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
    original_dataset = pd.read_csv(args.dataset)
    
    # Get dataset filename without extension for output naming
    filename_prefix = args.output_prefix
    if filename_prefix == 'dataset':
        filename_prefix = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Apply max rows limit if specified
    if args.max_rows is not None:
        original_dataset = original_dataset.head(args.max_rows)
        print(f"Limited dataset to {args.max_rows} rows")
    
    # Create a copy for processing
    dataset = original_dataset.copy()
    
    # Apply reordering if requested
    if args.reorder:
        print("Applying reordering algorithm...")
        dataset = reorder_dataset(
            dataset, 
            perform_sort=not args.no_sort,
            perform_dedup=not args.no_dedup,
            content_column=args.content_column
        )
        
        # Compare before and after to verify changes
        columns_changed = original_dataset.columns.tolist() != dataset.columns.tolist()
        rows_changed = len(original_dataset) != len(dataset)
        
        print("\nReordering Results:")
        print(f"- Columns reordered: {'Yes' if columns_changed else 'No'}")
        if columns_changed:
            print(f"  Before: {original_dataset.columns.tolist()[:3]}... ({len(original_dataset.columns)} columns)")
            print(f"  After:  {dataset.columns.tolist()[:3]}... ({len(dataset.columns)} columns)")
        print(f"- Row count changed: {'Yes' if rows_changed else 'No'}")
        if rows_changed:
            print(f"  Before: {len(original_dataset)} rows")
            print(f"  After:  {len(dataset)} rows")
    else:
        print("Skipping reordering as per command line argument")
    
    # Save results
    output_path = save_results(dataset, result_dir, original_dataset, filename_prefix)
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Reordered dataset saved to: {output_path}")
    print(f"All results saved to directory: {result_dir}")
    
    # Return the output path for potential chaining with other scripts
    return output_path

if __name__ == "__main__":
    main()
