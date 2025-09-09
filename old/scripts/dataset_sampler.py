#!/usr/bin/env python3
# filepath: /Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/old/dataset_sampler.py

"""
Dataset Sampler

This script takes a CSV dataset and creates a randomly sampled subset with 
the specified number of rows. The sampled dataset is saved to a new file.
"""

import os
import pandas as pd
import argparse
import datetime
import random


def setup_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampled_data")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def sample_dataset(input_file, num_samples, random_seed=None, stratify_column=None):
    """
    Randomly sample a dataset to the specified number of rows.
    
    Args:
        input_file (str): Path to the input CSV file
        num_samples (int): Number of rows to sample
        random_seed (int, optional): Random seed for reproducibility
        stratify_column (str, optional): Column to use for stratified sampling
        
    Returns:
        pd.DataFrame: Sampled dataset
    """
    print(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    if num_samples >= total_rows:
        print(f"Warning: Requested sample size ({num_samples}) is >= total rows ({total_rows})")
        print("Returning the entire dataset")
        return df
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    
    # Perform stratified sampling if a column is specified
    if stratify_column and stratify_column in df.columns:
        print(f"Performing stratified sampling based on column: {stratify_column}")
        
        # Calculate sampling fractions for each group
        groups = df[stratify_column].value_counts()
        sampling_fractions = {}
        
        for group, count in groups.items():
            fraction = min(num_samples / total_rows, 1.0)
            sampling_fractions[group] = fraction
        
        # Sample from each group
        sampled_dfs = []
        for group, fraction in sampling_fractions.items():
            group_df = df[df[stratify_column] == group]
            group_sample_size = int(fraction * len(group_df))
            if group_sample_size > 0:
                sampled_dfs.append(group_df.sample(n=group_sample_size, random_state=random_seed))
        
        # Combine the stratified samples
        sampled_df = pd.concat(sampled_dfs)
        
        # If we have too many samples, take a random subset
        if len(sampled_df) > num_samples:
            sampled_df = sampled_df.sample(n=num_samples, random_state=random_seed)
        
        print(f"Created stratified sample with {len(sampled_df)} rows")
        return sampled_df
    else:
        # Simple random sampling
        print(f"Performing simple random sampling of {num_samples} rows")
        return df.sample(n=num_samples, random_state=random_seed)


def main():
    parser = argparse.ArgumentParser(description='Randomly sample a dataset to a specified number of rows.')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the input CSV file')
    parser.add_argument('--samples', type=int, required=True, 
                        help='Number of rows to sample')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Path to the output CSV file (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--stratify', type=str, default=None,
                        help='Column to use for stratified sampling')
    
    args = parser.parse_args()
    
    # Sample the dataset
    sampled_df = sample_dataset(args.input, args.samples, args.seed, args.stratify)
    
    # Create output filename if not specified
    if args.output:
        output_file = args.output
    else:
        output_dir = setup_output_directory()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(output_dir, f"{base_filename}_sampled_{args.samples}_{timestamp}.csv")
    
    # Save the sampled dataset
    sampled_df.to_csv(output_file, index=False)
    print(f"Sampled dataset saved to {output_file}")
    print(f"Original size: {len(pd.read_csv(args.input))} rows, Sampled size: {len(sampled_df)} rows")


if __name__ == "__main__":
    main()
