#!/usr/bin/env python3
"""
GGR Experiment Pipeline - Main Entry Point
Command-line interface for running GGR algorithm experiments
"""
import argparse
import sys
import os
import logging
from typing import Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.experiment_runner import run_single_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="GGR Algorithm Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-discovery of functional dependencies
  python main.py data/dataset.csv
  
  # Specify functional dependencies manually
  python main.py data/dataset.csv --fds "movie_id->title,movie_id->genre"
  
  # Select specific columns and customize parameters
  python main.py data/dataset.csv --columns "col1,col2,col3" --max-depth 50 --output results/
  
  # Disable FD discovery and provide custom FDs
  python main.py data/dataset.csv --fds "A->B,C->D" --no-discover-fds
        """
    )
    
    # Required arguments
    parser.add_argument('dataset_path', 
                       help='Path to the input CSV dataset')
    
    # Optional arguments
    parser.add_argument('--fds', '--functional-dependencies',
                       type=str,
                       help='Functional dependencies as comma-separated pairs (e.g., "col1->col2,col3->col4")')
    
    parser.add_argument('--columns', '--columns-of-interest',
                       type=str,
                       help='Comma-separated list of columns to analyze (if not specified, uses all columns)')
    
    parser.add_argument('--output', '--output-dir',
                       type=str,
                       default='results',
                       help='Output directory for results (default: results)')
    
    parser.add_argument('--name', '--experiment-name',
                       type=str,
                       help='Name for this experiment (auto-generated if not specified)')
    
    parser.add_argument('--max-depth',
                       type=int,
                       default=100,
                       help='Maximum recursion depth for GGR algorithm (default: 100)')
    
    parser.add_argument('--no-discover-fds',
                       action='store_true',
                       help='Disable automatic discovery of functional dependencies')
    
    parser.add_argument('--fd-confidence',
                       type=float,
                       default=0.95,
                       help='Confidence threshold for functional dependency discovery (default: 0.95)')
    
    parser.add_argument('--handle-missing',
                       choices=['drop', 'fill', 'keep'],
                       default='drop',
                       help='How to handle missing values: drop, fill, or keep (default: drop)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    
    if not args.dataset_path.endswith('.csv'):
        logger.warning("Input file does not have .csv extension. Proceeding anyway...")
    
    try:
        logger.info("Starting GGR experiment pipeline...")
        
        # Run experiment
        results = run_single_experiment(
            dataset_path=args.dataset_path,
            functional_dependencies=args.fds,
            columns=args.columns,
            output_dir=args.output,
            experiment_name=args.name,
            max_depth=args.max_depth,
            discover_fds=not args.no_discover_fds,
            fd_confidence=args.fd_confidence,
            handle_missing=args.handle_missing
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Experiment Name: {results['experiment_name']}")
        print(f"Dataset: {results['dataset_info']['path']}")
        print(f"Original Shape: {results['dataset_info']['original_shape']}")
        print(f"Processed Shape: {results['processed_shape']}")
        print(f"Functional Dependencies: {len(results['functional_dependencies'])}")
        print(f"Total Hits: {results['total_hits']}")
        print(f"Reordered Table Size: {results['reordered_table_size']}")
        print(f"GGR Execution Time: {results['ggr_execution_time']:.2f} seconds")
        print(f"Total Time: {results['total_execution_time']:.2f} seconds")
        
        if 'output_files' in results:
            print(f"\nOutput Files:")
            for file_type, file_path in results['output_files'].items():
                print(f"  {file_type}: {file_path}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
