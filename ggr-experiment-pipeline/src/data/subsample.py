#!/usr/bin/env python3
"""
CSV Subsampling Script
Takes a CSV file, randomly samples 50 rows, and saves it to a new file.
"""

import argparse
import pandas as pd
import os
import sys
from datetime import datetime

def subsample_csv(input_file, output_file=None, n_rows=50, random_seed=42):
    """
    Subsample a CSV file to n_rows and save it
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        n_rows (int): Number of rows to sample
        random_seed (int): Random seed for reproducible sampling
    """
    
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Check if we have enough rows
        if len(df) <= n_rows:
            print(f"Warning: Dataset has {len(df)} rows, which is <= {n_rows}. Using all rows.")
            sampled_df = df.copy()
        else:
            # Random sampling
            print(f"Randomly sampling {n_rows} rows...")
            sampled_df = df.sample(n=n_rows, random_state=random_seed)
            print(f"Sampled dataset: {len(sampled_df)} rows")
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_sample{n_rows}_{timestamp}.csv"
        
        # Save the sampled data
        sampled_df.to_csv(output_file, index=False)
        print(f"Sampled data saved to: {output_file}")
        
        # Display first few rows
        print("\nFirst 5 rows of sampled data:")
        print(sampled_df.head())
        
        return output_file
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: '{input_file}' is empty or not a valid CSV file.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Subsample a CSV file to a specified number of rows")
    
    # Required argument
    parser.add_argument("input_csv", help="Path to input CSV file")
    
    # Optional arguments
    parser.add_argument("-o", "--output", help="Output CSV file path (optional)")
    parser.add_argument("-n", "--rows", type=int, default=50, 
                       help="Number of rows to sample (default: 50)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                       help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--head", action="store_true",
                       help="Take first n rows instead of random sampling")
    parser.add_argument("--tail", action="store_true", 
                       help="Take last n rows instead of random sampling")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' does not exist.")
        sys.exit(1)
    
    if not args.input_csv.lower().endswith('.csv'):
        print(f"Warning: Input file '{args.input_csv}' doesn't have .csv extension.")
    
    try:
        # Read the CSV file
        print(f"Reading CSV file: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Check if we have enough rows
        if len(df) <= args.rows:
            print(f"Warning: Dataset has {len(df)} rows, which is <= {args.rows}. Using all rows.")
            sampled_df = df.copy()
        else:
            # Choose sampling method
            if args.head:
                print(f"Taking first {args.rows} rows...")
                sampled_df = df.head(args.rows)
            elif args.tail:
                print(f"Taking last {args.rows} rows...")
                sampled_df = df.tail(args.rows)
            else:
                print(f"Randomly sampling {args.rows} rows (seed: {args.seed})...")
                sampled_df = df.sample(n=args.rows, random_state=args.seed)
        
        # Generate output filename if not provided
        if args.output is None:
            base_name = os.path.splitext(args.input_csv)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method = "head" if args.head else "tail" if args.tail else "random"
            args.output = f"{base_name}_sample{args.rows}_{method}_{timestamp}.csv"
        
        # Save the sampled data
        sampled_df.to_csv(args.output, index=False)
        print(f"Sampled data saved to: {args.output}")
        
        # Display summary
        print(f"\nSummary:")
        print(f"- Original rows: {len(df)}")
        print(f"- Sampled rows: {len(sampled_df)}")
        print(f"- Columns: {len(sampled_df.columns)}")
        print(f"- Output file: {args.output}")
        
        # Display first few rows of sampled data
        print(f"\nFirst 5 rows of sampled data:")
        print(sampled_df.head())
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_csv}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: '{args.input_csv}' is empty or not a valid CSV file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()