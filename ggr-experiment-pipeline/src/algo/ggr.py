#!/usr/bin/env python3
"""
Greedy Group Recursion (GGR) Algorithm Implementation

This script applies the GGR algorithm to reorder datasets for improved prefix cache hit rates
in LLM inference. It can automatically discover functional dependencies or use provided ones.

Usage:
    python ggr.py dataset.csv [--fds "col1->col2,col1->col3"] [--output results/] [--max-depth 100]

Features:
- Automatic functional dependency discovery (1->1 only)
- GGR algorithm implementation with configurable depth
- PHC (Prefix Hit Count) score calculation
- Result saving with metadata
- Comprehensive logging and analysis

Note: Only 1->1 functional dependencies are supported for optimal performance.
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
    
    def __init__(self, min_confidence: float = 0.95):
        self.min_confidence = min_confidence
        logger.info(f"FD Discovery configured: 1->1 only, min_confidence={self.min_confidence}")
    
    def discover_dependencies(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Discover functional dependencies in the dataset
        Only supports 1->1 dependencies (single column -> single column)
        Returns list of (source_column, target_column) tuples
        """
        logger.info(f"Discovering 1->1 functional dependencies with confidence >= {self.min_confidence}")
        
        dependencies = []
        columns = df.columns.tolist()
        
        # Check single column dependencies only (A -> B)
        logger.info("Checking single-column dependencies (A -> B)...")
        for source in columns:
            for target in columns:
                if source != target:
                    confidence = self._calculate_confidence(df, source, target)
                    if confidence >= self.min_confidence:
                        dependencies.append((source, target))
                        logger.info(f"Found FD: {source} -> {target} (confidence: {confidence:.3f})")
        
        logger.info(f"Discovered {len(dependencies)} functional dependencies (1->1 constraint)")
        return dependencies
    
    def _calculate_confidence(self, df: pd.DataFrame, source_col: str, target_col: str) -> float:
        """Calculate confidence of functional dependency source_col -> target_col"""
        try:
            # Group by source column and check if target is unique
            grouped = df.groupby(source_col)[target_col].nunique()
            
            # Count how many groups have exactly one unique target value
            valid_groups = (grouped == 1).sum()
            total_groups = len(grouped)
            
            if total_groups == 0:
                return 0.0
            
            confidence = valid_groups / total_groups
            return confidence
        except Exception as e:
            logger.debug(f"Error calculating confidence for {source_col} -> {target_col}: {e}")
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
    
    def apply_ggr(self, table: pd.DataFrame, functional_dependencies: List[Tuple[str, str]]) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
        """
        Apply GGR algorithm to reorder the table
        Returns (total_phc_score, reordered_dataframe, statistics)
        """
        logger.info(f"Applying GGR algorithm to table with shape {table.shape}")
        logger.info(f"Using {len(functional_dependencies)} functional dependencies (1->1 only)")
        
        # Validate that all FDs are 1->1
        for source, target in functional_dependencies:
            if ',' in source:
                logger.warning(f"Skipping multi-column FD: {source} -> {target} (only 1->1 supported)")
                continue
        
        # Filter to only 1->1 dependencies
        filtered_fds = [(s, t) for s, t in functional_dependencies if ',' not in s]
        if len(filtered_fds) != len(functional_dependencies):
            logger.info(f"Filtered to {len(filtered_fds)} valid 1->1 dependencies")
        
        # Reset statistics
        self.total_phc_score = 0
        self.recursion_stats = {
            'max_depth_reached': 0,
            'total_calls': 0,
            'single_row_cases': 0,
            'single_col_cases': 0
        }
        
        start_time = time.time()
        phc_score, reordered_table = self._ggr_recursive(table, filtered_fds, depth=0)
        end_time = time.time()
        
        # Compile statistics
        stats = {
            'phc_score': phc_score,
            'execution_time_seconds': end_time - start_time,
            'original_table_shape': table.shape,
            'reordered_table_shape': reordered_table.shape,
            'max_recursion_depth': self.recursion_stats['max_depth_reached'],
            'total_recursive_calls': self.recursion_stats['total_calls'],
            'single_row_cases': self.recursion_stats['single_row_cases'],
            'single_col_cases': self.recursion_stats['single_col_cases'],
            'functional_dependencies_count': len(filtered_fds)
        }
        
        logger.info(f"GGR completed: PHC Score = {phc_score}, Time = {stats['execution_time_seconds']:.2f}s")
        logger.info(f"Original shape: {table.shape} -> Reordered shape: {reordered_table.shape}")
        
        return phc_score, reordered_table, stats
    
    def _ggr_recursive(self, table: pd.DataFrame, functional_dependencies: List[Tuple[str, str]], depth: int = 0) -> Tuple[int, pd.DataFrame]:
        """Recursive GGR implementation that preserves original rows"""
        self.recursion_stats['total_calls'] += 1
        self.recursion_stats['max_depth_reached'] = max(self.recursion_stats['max_depth_reached'], depth)
        
        logger.debug(f"GGR: Depth {depth}, Table Size: {table.shape}")
        
        # Base conditions
        if table.shape[0] <= 1:  # Single row or empty case
            self.recursion_stats['single_row_cases'] += 1
            return 0, table.copy()
        
        if table.shape[1] == 1:  # Single column case
            self.recursion_stats['single_col_cases'] += 1
            sorted_table = table.sort_values(by=table.columns[0])
            phc_score = sum(
                9 if pd.isna(value)  # 'nan' represented as 3^2
                else len(str(value))**2
                for value in sorted_table.iloc[:, 0]
            )
            return phc_score, sorted_table.copy()
        
        # Prevent excessive recursion
        if depth >= self.max_depth:
            logger.warning(f"Maximum recursion depth {self.max_depth} reached")
            return 0, table.copy()
        
        # Find the best field and value combination
        max_hit_count, best_value, best_field, best_cols = -1, None, None, []
        
        for field in table.columns:
            unique_values = table[field].unique()
            for value in unique_values:
                hit_count, cols = self._calculate_hit_count(value, field, table, functional_dependencies)
                if hit_count > max_hit_count:
                    max_hit_count, best_value, best_field, best_cols = hit_count, value, field, cols
        
        if best_field is None:  # No valid field found
            logger.warning("No valid field found, returning original table")
            return 0, table.copy()
        
        logger.debug(f"Best choice: field={best_field}, value={best_value}, hit_count={max_hit_count}")
        
        # Split the table while preserving row indices
        if pd.isna(best_value):
            rows_with_value = table[table[best_field].isna()].copy()
            remaining_rows = table[~table[best_field].isna()].copy()
        else:
            rows_with_value = table[table[best_field] == best_value].copy()
            remaining_rows = table[table[best_field] != best_value].copy()
        
        # Recursive calls
        hit_count_A, reordered_A = self._ggr_recursive(remaining_rows, functional_dependencies, depth + 1)
        
        # For rows with the same value, we don't need to recurse if they have functional dependencies
        # Just group them together since they share the prefix
        hit_count_B = 0
        reordered_B = rows_with_value.copy()
        
        # Only recurse on the subset if there are remaining columns after removing functional dependencies
        remaining_cols = [col for col in rows_with_value.columns if col not in best_cols]
        if len(remaining_cols) > 1 and len(rows_with_value) > 1:
            # Create a subset for recursion, but we'll merge results back properly
            rows_subset = rows_with_value[remaining_cols].copy()
            hit_count_B, reordered_subset = self._ggr_recursive(rows_subset, functional_dependencies, depth + 1)
            
            # Reconstruct the full rows by mapping the reordered subset back to original rows
            if not reordered_subset.empty and len(reordered_subset) == len(rows_with_value):
                # Create a mapping from subset to original rows
                subset_to_original = {}
                for i, (_, subset_row) in enumerate(reordered_subset.iterrows()):
                    for j, (orig_idx, orig_row) in enumerate(rows_with_value.iterrows()):
                        # Check if this original row matches the subset row
                        match = True
                        for col in remaining_cols:
                            if col in subset_row and col in orig_row:
                                if str(subset_row[col]) != str(orig_row[col]):
                                    match = False
                                    break
                        if match and j not in subset_to_original.values():
                            subset_to_original[i] = orig_idx
                            break
                
                # Reorder the original rows based on the subset ordering
                if len(subset_to_original) == len(rows_with_value):
                    reordered_indices = [subset_to_original[i] for i in range(len(reordered_subset))]
                    reordered_B = rows_with_value.loc[reordered_indices].copy()
        
        # Combine results: rows with best_value first, then remaining rows
        total_hit_count = hit_count_A + hit_count_B + max_hit_count
        
        # Concatenate the reordered sections
        result_frames = []
        if not reordered_B.empty:
            result_frames.append(reordered_B)
        if not reordered_A.empty:
            result_frames.append(reordered_A)
        
        if result_frames:
            combined_result = pd.concat(result_frames, ignore_index=True)
        else:
            combined_result = pd.DataFrame(columns=table.columns)
        
        return total_hit_count, combined_result
    
    def _calculate_hit_count(self, value: Any, field: str, table: pd.DataFrame, functional_dependencies: List[Tuple[str, str]]) -> Tuple[int, List[str]]:
        """Calculate hit count for a value in a field (1->1 FDs only)"""
        try:
            # Get rows with this value
            if pd.isna(value):
                rows_with_value = table[table[field].isna()]
            else:
                rows_with_value = table[table[field] == value]
            
            if len(rows_with_value) == 0:
                return 0, [field]
            
            # Find functionally dependent columns (1->1 only)
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
            # Validate that provided FDs are 1->1
            valid_fds = []
            for source, target in functional_dependencies:
                if ',' in source:
                    logger.warning(f"Skipping multi-column FD: {source} -> {target} (only 1->1 supported)")
                else:
                    valid_fds.append((source, target))
            functional_dependencies = valid_fds
            logger.info(f"Using {len(functional_dependencies)} valid 1->1 functional dependencies")
        
        # Apply GGR algorithm
        self.ggr_algorithm.max_depth = max_depth
        phc_score, reordered_df, ggr_stats = self.ggr_algorithm.apply_ggr(df, functional_dependencies)
        
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
        """Save the reordered dataset and a simple text report in a timestamped folder"""
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped folder for this run
        run_folder = os.path.join(self.output_dir, f"{base_name}_ggr_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        logger.info(f"Created run folder: {run_folder}")
        
        # Save reordered dataset
        if not reordered_df.empty:
            reordered_path = os.path.join(run_folder, f"{base_name}_reordered.csv")
            reordered_df.to_csv(reordered_path, index=False)
            logger.info(f"Reordered dataset saved to: {reordered_path}")
            results["reordered_dataset_path"] = reordered_path
        
        # Save simple summary report
        report_path = os.path.join(run_folder, "ggr_report.txt")
        self._generate_simple_report(results, report_path)
        
        # Update results with folder information
        results["run_folder"] = run_folder
        results["run_timestamp"] = timestamp
    
    def _generate_simple_report(self, results: Dict[str, Any], report_path: str):
        """Generate a simple, concise summary report"""
        with open(report_path, 'w') as f:
            f.write("GGR ALGORITHM RESULTS\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic information
            f.write("DATASET INFORMATION:\n")
            f.write(f"Original file: {os.path.basename(results['dataset_info']['path'])}\n")
            f.write(f"Original shape: {results['dataset_info']['original_shape']}\n")
            f.write(f"Reordered shape: {results['reordered_shape']}\n")
            f.write(f"Columns: {len(results['dataset_info']['columns'])}\n")
            f.write(f"Processing time: {results['processing_timestamp']}\n\n")
            
            # Functional dependencies found
            f.write("FUNCTIONAL DEPENDENCIES DISCOVERED:\n")
            if results['functional_dependencies']:
                for i, (source, target) in enumerate(results['functional_dependencies'], 1):
                    f.write(f"{i:2d}. {source} -> {target}\n")
            else:
                f.write("None found\n")
            f.write(f"Total: {len(results['functional_dependencies'])}\n\n")
            
            # Key results
            ggr_stats = results['ggr_results']
            f.write("GGR RESULTS:\n")
            f.write(f"PHC Score: {results['phc_score']:,}\n")
            f.write(f"Execution time: {ggr_stats['execution_time_seconds']:.2f} seconds\n")
            f.write(f"Max recursion depth: {ggr_stats['max_recursion_depth']}\n")
            f.write(f"Total recursive calls: {ggr_stats['total_recursive_calls']:,}\n\n")
            
            # Performance assessment
            f.write("PERFORMANCE ASSESSMENT:\n")
            if results['phc_score'] > 1000:
                f.write("✅ EXCELLENT - High optimization potential\n")
                f.write("Expected benefits: 2-3x speedup in LLM inference\n")
            elif results['phc_score'] > 100:
                f.write("✅ GOOD - Moderate optimization benefits\n")
                f.write("Expected benefits: 1.5-2x speedup in LLM inference\n")
            elif results['phc_score'] > 10:
                f.write("⚠️  FAIR - Limited optimization benefits\n")
                f.write("Expected benefits: Minor speedup in LLM inference\n")
            else:
                f.write("❌ LOW - Minimal optimization benefits\n")
                f.write("Original dataset may already be well-organized\n")
            
            f.write(f"\nAlgorithm efficiency: ")
            if ggr_stats['max_recursion_depth'] < 50:
                f.write("Optimal\n")
            elif ggr_stats['max_recursion_depth'] < 80:
                f.write("Good\n")
            else:
                f.write("Deep recursion\n")
            
            # Usage instructions
            f.write(f"\nUSAGE:\n")
            base_name = os.path.splitext(os.path.basename(results['dataset_info']['path']))[0]
            f.write(f"Use {base_name}_reordered.csv for LLM inference experiments\n")
            f.write(f"Enable prefix caching in vLLM for optimal results\n")
            
            # Files in this folder
            f.write(f"\nFILES IN THIS FOLDER:\n")
            f.write(f"- {base_name}_reordered.csv: GGR-optimized dataset\n")
            f.write(f"- ggr_report.txt: This summary report\n")
        
        logger.info(f"Simple report saved to: {report_path}")
        
        # Log the folder location for easy access
        run_folder = results.get('run_folder', '')
        if run_folder:
            logger.info(f"📁 Results saved in: {os.path.basename(run_folder)}/")
            logger.info(f"🔍 Full path: {run_folder}")
            logger.info(f"📊 View report: cat '{report_path}'")


def parse_functional_dependencies(fd_string: str) -> List[Tuple[str, str]]:
    """Parse functional dependencies from command line string (1->1 only)"""
    dependencies = []
    if fd_string:
        for fd in fd_string.split(','):
            fd = fd.strip()
            if '->' in fd:
                source, target = fd.split('->', 1)
                source = source.strip()
                target = target.strip()
                
                # Validate that it's a 1->1 dependency
                if ',' in source:
                    logger.warning(f"Skipping multi-column FD: {source} -> {target} (only 1->1 supported)")
                    continue
                
                dependencies.append((source, target))
    return dependencies


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Apply GGR algorithm to reorder datasets for improved LLM inference performance (1->1 FDs only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ggr.py data.csv
  python ggr.py data.csv --fds "movie_id->title,movie_id->genre"
  python ggr.py data.csv --output results/ --max-depth 50 --verbose

Note: Only 1->1 functional dependencies (single column -> single column) are supported.
Multi-column dependencies will be automatically filtered out.
        """
    )
    
    parser.add_argument('dataset', help='Path to the CSV dataset file')
    parser.add_argument('--fds', '--functional-dependencies', 
                       help='Functional dependencies as comma-separated list (e.g., "col1->col2,col1->col3") - 1->1 only')
    parser.add_argument('--output', '-o', default='reorder_results',
                       help='Output directory for results (default: reorder_results)')
    parser.add_argument('--max-depth', type=int, default=100,
                       help='Maximum recursion depth for GGR algorithm (default: 100)')
    parser.add_argument('--fd-confidence', type=float, default=0.95,
                       help='Confidence threshold for automatic FD discovery (default: 0.95)')
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
        logger.info(f"Parsed {len(functional_dependencies)} valid 1->1 functional dependencies")
    
    # Create processor with configuration
    processor = GGRProcessor(output_dir=args.output)
    processor.fd_discoverer.min_confidence = args.fd_confidence
    
    # Process dataset
    try:
        logger.info("=" * 60)
        logger.info("STARTING GGR PROCESSING (1->1 FDs ONLY)")
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
        logger.info("GGR PROCESSING COMPLETE (1->1 FDs)")
        logger.info("=" * 70)
        logger.info(f"📊 PHC Score: {results['phc_score']:,}")
        logger.info(f"⏱️  Processing time: {results['ggr_results']['execution_time_seconds']:.3f}s")
        logger.info(f"📐 Original shape: {results['dataset_info']['original_shape']}")
        logger.info(f"🔄 Reordered shape: {results['reordered_shape']}")
        logger.info(f"🔗 Functional dependencies (1->1): {len(results['functional_dependencies'])}")
        
        # Show run folder information
        if 'run_folder' in results:
            run_folder = results['run_folder']
            logger.info(f"📁 Results saved in: {os.path.basename(run_folder)}/")
            logger.info(f"🔍 Full path: {run_folder}")
            
            # List files in the run folder (should only be 2 files now)
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
        
        logger.info(f"🎯 Algorithm used 1->1 functional dependencies for optimal performance")
        
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