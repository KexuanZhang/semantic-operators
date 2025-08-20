#!/usr/bin/env python3
"""
Greedy Group Recursion (GGR) Algorithm Implementation

This script applies the GGR algorithm to reorder datasets for improved prefix cache hit rates
in LLM inference. It can automatically discover functional dependencies or use provided ones.

Usage:
    python ggr.py dataset.csv [--fds "col1->col2,col1->col3"] [--output results/] [--max-depth 100]

Features:
- Automatic functional dependency discovery
- GGR algorithm implementation with configurable depth
- PHC (Prefix Hit Count) score calculation
- Result saving with metadata
- Comprehensive logging and analysis
"""

import argparse
import os
import sys
import json
import math
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import logging

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FunctionalDependencyDiscoverer:
    """Discover functional dependencies in a dataset using statistical analysis"""
    
    def __init__(self, min_confidence: float = 0.95, max_lhs_size: int = 2):
        self.min_confidence = min_confidence
        self.max_lhs_size = max_lhs_size
    
    def discover_dependencies(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Discover functional dependencies in the dataset
        Returns list of (source_column, target_column) tuples
        """
        logger.info(f"Discovering functional dependencies with confidence >= {self.min_confidence}")
        
        dependencies = []
        columns = df.columns.tolist()
        
        # Check single column dependencies (A -> B)
        for source in columns:
            for target in columns:
                if source != target:
                    confidence = self._calculate_confidence(df, [source], target)
                    if confidence >= self.min_confidence:
                        dependencies.append((source, target))
                        logger.info(f"Found FD: {source} -> {target} (confidence: {confidence:.3f})")
        
        # Check multi-column dependencies if requested
        if self.max_lhs_size > 1:
            for lhs_size in range(2, min(self.max_lhs_size + 1, len(columns))):
                for source_cols in combinations(columns, lhs_size):
                    for target in columns:
                        if target not in source_cols:
                            confidence = self._calculate_confidence(df, list(source_cols), target)
                            if confidence >= self.min_confidence:
                                source_str = ",".join(source_cols)
                                dependencies.append((source_str, target))
                                logger.info(f"Found FD: {source_str} -> {target} (confidence: {confidence:.3f})")
        
        logger.info(f"Discovered {len(dependencies)} functional dependencies")
        return dependencies
    
    def _calculate_confidence(self, df: pd.DataFrame, source_cols: List[str], target_col: str) -> float:
        """Calculate confidence of functional dependency source_cols -> target_col"""
        try:
            # Group by source columns and check if target is unique
            grouped = df.groupby(source_cols)[target_col].nunique()
            
            # Count how many groups have exactly one unique target value
            valid_groups = (grouped == 1).sum()
            total_groups = len(grouped)
            
            if total_groups == 0:
                return 0.0
            
            confidence = valid_groups / total_groups
            return confidence
        except Exception as e:
            logger.debug(f"Error calculating confidence for {source_cols} -> {target_col}: {e}")
            return 0.0


class GGRAlgorithm:
    """Implementation of the Greedy Group Recursion algorithm"""
    
    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self.total_phc_score = 0
        self.recursion_stats = {
            'max_depth_reached': 0,
            'total_calls': 0,
            'single_row_cases': 0,
            'single_col_cases': 0
        }
    
    def apply_ggr(self, table: pd.DataFrame, functional_dependencies: List[Tuple[str, str]]) -> Tuple[int, List[List[str]], Dict[str, Any]]:
        """
        Apply GGR algorithm to reorder the table
        Returns (total_phc_score, reordered_table, statistics)
        """
        logger.info(f"Applying GGR algorithm to table with shape {table.shape}")
        logger.info(f"Using {len(functional_dependencies)} functional dependencies")
        
        # Reset statistics
        self.total_phc_score = 0
        self.recursion_stats = {
            'max_depth_reached': 0,
            'total_calls': 0,
            'single_row_cases': 0,
            'single_col_cases': 0
        }
        
        start_time = time.time()
        phc_score, reordered_table = self._ggr_recursive(table, functional_dependencies, depth=0)
        end_time = time.time()
        
        # Compile statistics
        stats = {
            'phc_score': phc_score,
            'execution_time_seconds': end_time - start_time,
            'original_table_shape': table.shape,
            'reordered_table_length': len(reordered_table),
            'max_recursion_depth': self.recursion_stats['max_depth_reached'],
            'total_recursive_calls': self.recursion_stats['total_calls'],
            'single_row_cases': self.recursion_stats['single_row_cases'],
            'single_col_cases': self.recursion_stats['single_col_cases'],
            'functional_dependencies_count': len(functional_dependencies)
        }
        
        logger.info(f"GGR completed: PHC Score = {phc_score}, Time = {stats['execution_time_seconds']:.2f}s")
        
        return phc_score, reordered_table, stats
    
    def _ggr_recursive(self, table: pd.DataFrame, functional_dependencies: List[Tuple[str, str]], depth: int = 0) -> Tuple[int, List[List[str]]]:
        """Recursive GGR implementation"""
        self.recursion_stats['total_calls'] += 1
        self.recursion_stats['max_depth_reached'] = max(self.recursion_stats['max_depth_reached'], depth)
        
        logger.debug(f"GGR: Depth {depth}, Table Size: {table.shape}")
        
        # Base conditions
        if table.shape[0] == 1:  # Single row case
            self.recursion_stats['single_row_cases'] += 1
            return 0, table.iloc[0].tolist()
        
        if table.shape[1] == 1:  # Single column case
            self.recursion_stats['single_col_cases'] += 1
            sorted_table = table.sort_values(by=table.columns[0])
            phc_score = sum(
                9 if pd.isna(value)  # 'nan' represented as 3^2
                else len(str(value))**2
                for value in sorted_table.iloc[:, 0]
            )
            return phc_score, sorted_table.values.tolist()
        
        # Prevent excessive recursion
        if depth >= self.max_depth:
            logger.warning(f"Maximum recursion depth {self.max_depth} reached")
            return 0, []
        
        # Find the best field and value combination
        max_hit_count, best_value, best_field, best_cols = -1, None, None, []
        
        for field in table.columns:
            unique_values = table[field].unique()
            for value in unique_values:
                hit_count, cols = self._calculate_hit_count(value, field, table, functional_dependencies)
                if hit_count > max_hit_count:
                    max_hit_count, best_value, best_field, best_cols = hit_count, value, field, cols
        
        if best_field is None:  # No valid field found
            logger.warning("No valid field found, returning empty result")
            return 0, []
        
        logger.debug(f"Best choice: field={best_field}, value={best_value}, hit_count={max_hit_count}")
        
        # Split the table
        if pd.isna(best_value):
            rows_with_value = table[table[best_field].isna()]
            remaining_rows = table[~table[best_field].isna()]
        else:
            rows_with_value = table[table[best_field] == best_value]
            remaining_rows = table[table[best_field] != best_value]
        
        # Recursive calls
        hit_count_A, reordered_A = self._ggr_recursive(remaining_rows, functional_dependencies, depth + 1)
        
        # Remove the columns that are functionally determined
        remaining_cols = [col for col in rows_with_value.columns if col not in best_cols]
        if remaining_cols:
            rows_subset = rows_with_value[remaining_cols]
            hit_count_B, reordered_B = self._ggr_recursive(rows_subset, functional_dependencies, depth + 1)
        else:
            hit_count_B, reordered_B = 0, []
        
        # Combine results
        total_hit_count = hit_count_A + hit_count_B + max_hit_count
        
        # Create the reordered result list
        reordered_list = []
        
        # Handle best_value rows (with functional dependencies removed)
        if len(reordered_B) == 0:
            # Single value case - create a row with just the best value
            reordered_list.append([str(best_value)])
        else:
            # Multiple rows case - prepend best_value to each row in reordered_B  
            for row in reordered_B:
                if isinstance(row, list):
                    reordered_list.append([str(best_value)] + [str(item) for item in row])
                else:
                    reordered_list.append([str(best_value), str(row)])
        
        # Add the remaining rows (reordered_A)
        if isinstance(reordered_A, list):
            for row in reordered_A:
                if isinstance(row, list):
                    reordered_list.append([str(item) for item in row])
                else:
                    reordered_list.append([str(row)])
        
        return total_hit_count, reordered_list
    
    def _calculate_hit_count(self, value: Any, field: str, table: pd.DataFrame, functional_dependencies: List[Tuple[str, str]]) -> Tuple[int, List[str]]:
        """Calculate hit count for a value in a field"""
        try:
            # Get rows with this value
            if pd.isna(value):
                rows_with_value = table[table[field].isna()]
            else:
                rows_with_value = table[table[field] == value]
            
            if len(rows_with_value) == 0:
                return 0, [field]
            
            # Find functionally dependent columns
            inferred_columns = []
            for source, target in functional_dependencies:
                if source == field and target in table.columns:
                    inferred_columns.append(target)
            
            # Calculate total length contribution
            value_length = 9 if pd.isna(value) else len(str(value))**2
            
            inferred_length = 0
            for col in inferred_columns:
                if col in rows_with_value.columns:
                    col_lengths = rows_with_value[col].apply(lambda x: 9 if pd.isna(x) else len(str(x)))
                    if len(col_lengths) > 0:
                        inferred_length += col_lengths.sum() / len(rows_with_value)
            
            total_length = value_length + inferred_length
            hit_count = total_length * (len(rows_with_value) - 1)
            
            return int(hit_count), [field] + inferred_columns
            
        except Exception as e:
            logger.debug(f"Error calculating hit count for {field}={value}: {e}")
            return 0, [field]


class GGRProcessor:
    """Main processor for GGR algorithm with dataset handling"""
    
    def __init__(self, output_dir: str = "reorder_results"):
        self.output_dir = output_dir
        self.fd_discoverer = FunctionalDependencyDiscoverer()
        self.ggr_algorithm = GGRAlgorithm()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def process_dataset(self, dataset_path: str, functional_dependencies: Optional[List[Tuple[str, str]]] = None, max_depth: int = 100) -> Dict[str, Any]:
        """
        Process a dataset with GGR algorithm
        Returns comprehensive results dictionary
        """
        logger.info(f"Processing dataset: {dataset_path}")
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {"error": str(e)}
        
        # Preprocess dataset
        df = self._preprocess_data(df)
        
        # Discover or use provided functional dependencies
        if functional_dependencies is None:
            logger.info("No functional dependencies provided, discovering automatically...")
            functional_dependencies = self.fd_discoverer.discover_dependencies(df)
        else:
            logger.info(f"Using provided functional dependencies: {functional_dependencies}")
        
        # Apply GGR algorithm
        self.ggr_algorithm.max_depth = max_depth
        phc_score, reordered_table, ggr_stats = self.ggr_algorithm.apply_ggr(df, functional_dependencies)
        
        # Convert reordered table back to DataFrame
        if reordered_table:
            # Determine the appropriate number of columns
            max_cols = max(len(row) for row in reordered_table) if reordered_table else 0
            
            # Pad rows to have the same length
            padded_rows = []
            for row in reordered_table:
                padded_row = row + [''] * (max_cols - len(row))
                padded_rows.append(padded_row)
            
            # Create column names
            original_cols = df.columns.tolist()
            col_names = original_cols + [f"col_{i}" for i in range(len(original_cols), max_cols)]
            
            reordered_df = pd.DataFrame(padded_rows, columns=col_names[:max_cols])
        else:
            reordered_df = pd.DataFrame()
        
        # Prepare results
        results = {
            "dataset_info": {
                "path": dataset_path,
                "original_shape": df.shape,
                "columns": df.columns.tolist()
            },
            "functional_dependencies": functional_dependencies,
            "ggr_results": ggr_stats,
            "phc_score": phc_score,
            "reordered_shape": reordered_df.shape,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(dataset_path, reordered_df, results)
        
        return results
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset for GGR algorithm"""
        logger.info("Preprocessing dataset...")
        
        original_shape = df.shape
        
        # Convert all columns to string type for consistent processing
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Handle missing values (convert to NaN for proper handling)
        df = df.replace(['nan', 'None', ''], np.nan)
        
        logger.info(f"Preprocessing complete: {original_shape} -> {df.shape}")
        return df
    
    def _save_results(self, dataset_path: str, reordered_df: pd.DataFrame, results: Dict[str, Any]):
        """Save the reordered dataset and metadata in a timestamped folder"""
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped folder for this run
        run_folder = os.path.join(self.output_dir, f"{base_name}_ggr_run_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        logger.info(f"Created run folder: {run_folder}")
        
        # Save reordered dataset
        if not reordered_df.empty:
            reordered_path = os.path.join(run_folder, f"{base_name}_reordered.csv")
            reordered_df.to_csv(reordered_path, index=False)
            logger.info(f"Reordered dataset saved to: {reordered_path}")
            results["reordered_dataset_path"] = reordered_path
        
        # Save original dataset copy for reference
        original_copy_path = os.path.join(run_folder, f"{base_name}_original.csv")
        try:
            original_df = pd.read_csv(dataset_path)
            original_df.to_csv(original_copy_path, index=False)
            logger.info(f"Original dataset copy saved to: {original_copy_path}")
            results["original_dataset_copy_path"] = original_copy_path
        except Exception as e:
            logger.warning(f"Could not save original dataset copy: {e}")
        
        # Save metadata
        metadata_path = os.path.join(run_folder, "ggr_results.json")
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results metadata saved to: {metadata_path}")
        
        # Save summary report
        report_path = os.path.join(run_folder, "ggr_summary.txt")
        self._generate_summary_report(results, report_path)
        
        # Save run information
        run_info_path = os.path.join(run_folder, "run_info.txt")
        with open(run_info_path, 'w') as f:
            f.write(f"GGR Run Information\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Run Timestamp: {timestamp}\n")
            f.write(f"Run Folder: {os.path.basename(run_folder)}\n")
            f.write(f"Original Dataset: {dataset_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write(f"Files in this run:\n")
            f.write(f"- {base_name}_original.csv: Copy of original input dataset\n")
            f.write(f"- {base_name}_reordered.csv: GGR-reordered dataset\n")
            f.write(f"- ggr_results.json: Complete results with metadata\n")
            f.write(f"- ggr_summary.txt: Human-readable summary report\n")
            f.write(f"- run_info.txt: This run information file\n")
        logger.info(f"Run information saved to: {run_info_path}")
        
        # Update results with folder information
        results["run_folder"] = run_folder
        results["run_timestamp"] = timestamp
    
    def _generate_summary_report(self, results: Dict[str, Any], report_path: str):
        """Generate a human-readable summary report"""
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("GGR ALGORITHM RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Run information
            f.write("RUN INFORMATION:\n")
            f.write(f"- Run Folder: {os.path.basename(results.get('run_folder', ''))}\n")
            f.write(f"- Run Timestamp: {results.get('run_timestamp', results['processing_timestamp'])}\n")
            f.write(f"- Processing Time: {results['processing_timestamp']}\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION:\n")
            f.write(f"- Original file: {results['dataset_info']['path']}\n")
            f.write(f"- Original shape: {results['dataset_info']['original_shape']}\n")
            f.write(f"- Reordered shape: {results['reordered_shape']}\n")
            f.write(f"- Columns: {', '.join(results['dataset_info']['columns'])}\n\n")
            
            # Functional dependencies
            f.write("FUNCTIONAL DEPENDENCIES:\n")
            if results['functional_dependencies']:
                for i, (source, target) in enumerate(results['functional_dependencies'], 1):
                    f.write(f"  {i:2d}. {source} -> {target}\n")
            else:
                f.write("- No functional dependencies found or provided\n")
            f.write(f"- Total dependencies: {len(results['functional_dependencies'])}\n\n")
            
            # GGR results
            f.write("GGR ALGORITHM RESULTS:\n")
            ggr_stats = results['ggr_results']
            f.write(f"- PHC Score: {results['phc_score']:,}\n")
            f.write(f"- Execution time: {ggr_stats['execution_time_seconds']:.3f} seconds\n")
            f.write(f"- Max recursion depth: {ggr_stats['max_recursion_depth']}\n")
            f.write(f"- Total recursive calls: {ggr_stats['total_recursive_calls']:,}\n")
            f.write(f"- Single row cases: {ggr_stats['single_row_cases']:,}\n")
            f.write(f"- Single column cases: {ggr_stats['single_col_cases']:,}\n\n")
            
            # Performance analysis
            f.write("PERFORMANCE ANALYSIS:\n")
            original_size = results['dataset_info']['original_shape'][0] * results['dataset_info']['original_shape'][1]
            if original_size > 0:
                phc_per_cell = results['phc_score'] / original_size
                processing_rate = original_size / ggr_stats['execution_time_seconds']
                f.write(f"- PHC score per cell: {phc_per_cell:.2f}\n")
                f.write(f"- Processing rate: {processing_rate:.0f} cells/second\n")
                
                # Performance rating
                if results['phc_score'] > 1000:
                    f.write(f"- Performance rating: ✅ Excellent - High potential for cache optimization\n")
                elif results['phc_score'] > 100:
                    f.write(f"- Performance rating: ✅ Good - Moderate cache optimization benefits\n")
                elif results['phc_score'] > 10:
                    f.write(f"- Performance rating: ⚠️  Fair - Limited cache optimization benefits\n")
                else:
                    f.write(f"- Performance rating: ❌ Poor - Minimal cache optimization benefits\n")
            else:
                f.write(f"- Unable to calculate performance metrics (empty dataset)\n")
            
            f.write(f"\n")
            
            # Algorithm efficiency
            f.write("ALGORITHM EFFICIENCY:\n")
            if ggr_stats['max_recursion_depth'] < 50:
                f.write(f"- Recursion depth: ✅ Optimal (max depth: {ggr_stats['max_recursion_depth']})\n")
            elif ggr_stats['max_recursion_depth'] < 80:
                f.write(f"- Recursion depth: ⚠️  Moderate (max depth: {ggr_stats['max_recursion_depth']})\n")
            else:
                f.write(f"- Recursion depth: ❌ Deep (max depth: {ggr_stats['max_recursion_depth']})\n")
            
            calls_per_second = ggr_stats['total_recursive_calls'] / ggr_stats['execution_time_seconds']
            f.write(f"- Recursive calls per second: {calls_per_second:.0f}\n")
            
            # Files generated
            f.write(f"\nFILES GENERATED:\n")
            run_folder = results.get('run_folder', '')
            if run_folder:
                base_name = os.path.splitext(os.path.basename(results['dataset_info']['path']))[0]
                f.write(f"All files saved in folder: {os.path.basename(run_folder)}/\n")
                f.write(f"- {base_name}_original.csv: Copy of original input dataset\n")
                f.write(f"- {base_name}_reordered.csv: GGR-reordered dataset for optimal inference\n")
                f.write(f"- ggr_results.json: Complete results with detailed metadata\n")
                f.write(f"- ggr_summary.txt: This human-readable summary report\n")
                f.write(f"- run_info.txt: Run information and file descriptions\n")
            else:
                if results.get('reordered_dataset_path'):
                    f.write(f"- Reordered dataset: {results['reordered_dataset_path']}\n")
                f.write(f"- Results metadata: Available in JSON format\n")
            
            # Usage recommendations
            f.write(f"\nUSAGE RECOMMENDATIONS:\n")
            if results['phc_score'] > 100:
                f.write(f"✅ RECOMMENDED: Use the reordered dataset for LLM inference experiments\n")
                f.write(f"   - Expected benefits: Improved KV cache hit rates\n")
                f.write(f"   - Potential speedup: 1.5-3x faster inference\n")
                f.write(f"   - Memory efficiency: Reduced memory usage due to cache reuse\n")
            else:
                f.write(f"⚠️  MARGINAL: Dataset may have limited benefits from reordering\n")
                f.write(f"   - Consider: Adding more functional dependencies\n")
                f.write(f"   - Alternative: Try different grouping strategies\n")
                f.write(f"   - Analysis: Original data may already be well-organized\n")
            
            f.write(f"\nFor LLM inference experiments, use the reordered dataset with:\n")
            f.write(f"- vLLM with prefix caching enabled\n")
            f.write(f"- Batch processing to maximize cache utilization\n")
            f.write(f"- Performance monitoring to measure cache hit rates\n")
        
        logger.info(f"Summary report saved to: {report_path}")
        
        # Log the folder location for easy access
        run_folder = results.get('run_folder', '')
        if run_folder:
            logger.info(f"📁 All results saved in folder: {run_folder}")
            logger.info(f"🔍 To view results: ls -la '{run_folder}'")
            logger.info(f"📊 To view summary: cat '{report_path}'")


def parse_functional_dependencies(fd_string: str) -> List[Tuple[str, str]]:
    """Parse functional dependencies from command line string"""
    dependencies = []
    if fd_string:
        for fd in fd_string.split(','):
            fd = fd.strip()
            if '->' in fd:
                source, target = fd.split('->', 1)
                dependencies.append((source.strip(), target.strip()))
    return dependencies


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Apply GGR algorithm to reorder datasets for improved LLM inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ggr.py data.csv
  python ggr.py data.csv --fds "movie_id->title,movie_id->genre"
  python ggr.py data.csv --output results/ --max-depth 50 --verbose
        """
    )
    
    parser.add_argument('dataset', help='Path to the CSV dataset file')
    parser.add_argument('--fds', '--functional-dependencies', 
                       help='Functional dependencies as comma-separated list (e.g., "col1->col2,col1->col3")')
    parser.add_argument('--output', '-o', default='reorder_results',
                       help='Output directory for results (default: reorder_results)')
    parser.add_argument('--max-depth', type=int, default=100,
                       help='Maximum recursion depth for GGR algorithm (default: 100)')
    parser.add_argument('--fd-confidence', type=float, default=0.95,
                       help='Confidence threshold for automatic FD discovery (default: 0.95)')
    parser.add_argument('--fd-max-lhs', type=int, default=2,
                       help='Maximum left-hand side size for FD discovery (default: 2)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Parse functional dependencies
    functional_dependencies = None
    if args.fds:
        functional_dependencies = parse_functional_dependencies(args.fds)
        logger.info(f"Parsed {len(functional_dependencies)} functional dependencies")
    
    # Create processor with configuration
    processor = GGRProcessor(output_dir=args.output)
    processor.fd_discoverer.min_confidence = args.fd_confidence
    processor.fd_discoverer.max_lhs_size = args.fd_max_lhs
    
    # Process dataset
    try:
        logger.info("=" * 60)
        logger.info("STARTING GGR PROCESSING")
        logger.info("=" * 60)
        
        results = processor.process_dataset(
            dataset_path=args.dataset,
            functional_dependencies=functional_dependencies,
            max_depth=args.max_depth
        )
        
        if "error" in results:
            logger.error(f"Processing failed: {results['error']}")
            sys.exit(1)
        
        # Print summary
        logger.info("=" * 70)
        logger.info("GGR PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"📊 PHC Score: {results['phc_score']:,}")
        logger.info(f"⏱️  Processing time: {results['ggr_results']['execution_time_seconds']:.3f}s")
        logger.info(f"📐 Original shape: {results['dataset_info']['original_shape']}")
        logger.info(f"🔄 Reordered shape: {results['reordered_shape']}")
        logger.info(f"🔗 Functional dependencies: {len(results['functional_dependencies'])}")
        
        # Show run folder information
        if 'run_folder' in results:
            run_folder = results['run_folder']
            logger.info(f"📁 Results saved in: {os.path.basename(run_folder)}/")
            logger.info(f"🔍 Full path: {run_folder}")
            logger.info(f"📋 View summary: cat '{os.path.join(run_folder, 'ggr_summary.txt')}'")
            
            # List files in the run folder
            try:
                files = os.listdir(run_folder)
                logger.info(f"📄 Generated files: {', '.join(files)}")
            except Exception as e:
                logger.debug(f"Could not list run folder contents: {e}")
        else:
            logger.info(f"📁 Results saved to: {args.output}/")
        
        # Performance assessment
        if results['phc_score'] > 1000:
            logger.info(f"✅ EXCELLENT: High potential for LLM inference optimization!")
        elif results['phc_score'] > 100:
            logger.info(f"✅ GOOD: Moderate benefits expected for LLM inference")
        elif results['phc_score'] > 10:
            logger.info(f"⚠️  FAIR: Limited optimization benefits expected")
        else:
            logger.info(f"⚠️  LOW: Dataset may not benefit significantly from GGR reordering")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()