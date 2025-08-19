"""
GGR Experiment Runner
Orchestrates the complete experiment pipeline
"""
import pandas as pd
import numpy as np
import os
import time
import json
from typing import List, Tuple, Optional, Dict, Any
import logging

from .data_preprocessing import (
    load_dataset, 
    discover_functional_dependencies, 
    preprocess_data,
    parse_functional_dependencies,
    validate_functional_dependencies
)
from .ggr_algorithm import ggr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GGRExperiment:
    """
    Main experiment class for running GGR algorithm experiments
    """
    
    def __init__(self, 
                 dataset_path: str,
                 output_dir: str = "results",
                 experiment_name: Optional[str] = None):
        """
        Initialize experiment
        
        Args:
            dataset_path: Path to the dataset CSV file
            output_dir: Directory to save results
            experiment_name: Name for this experiment (auto-generated if None)
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"ggr_exp_{int(time.time())}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        
    def run_experiment(self,
                      functional_dependencies: Optional[List[Tuple[str, str]]] = None,
                      columns_of_interest: Optional[List[str]] = None,
                      max_depth: int = 100,
                      discover_fds: bool = True,
                      fd_confidence: float = 0.95,
                      handle_missing: str = 'drop') -> Dict[str, Any]:
        """
        Run the complete GGR experiment
        
        Args:
            functional_dependencies: Pre-defined FDs (if None and discover_fds=True, will auto-discover)
            columns_of_interest: Specific columns to analyze
            max_depth: Maximum recursion depth for GGR
            discover_fds: Whether to auto-discover FDs if not provided
            fd_confidence: Confidence threshold for FD discovery
            handle_missing: How to handle missing values
            
        Returns:
            Dictionary containing experiment results
        """
        logger.info(f"Starting GGR experiment: {self.experiment_name}")
        start_time = time.time()
        
        try:
            # Step 1: Load dataset
            df = load_dataset(self.dataset_path)
            self.results['dataset_info'] = {
                'path': self.dataset_path,
                'original_shape': df.shape,
                'columns': df.columns.tolist()
            }
            
            # Step 2: Preprocess data
            processed_df = preprocess_data(df, columns_of_interest, handle_missing)
            self.results['processed_shape'] = processed_df.shape
            self.results['processed_columns'] = processed_df.columns.tolist()
            
            # Step 3: Handle functional dependencies
            if functional_dependencies is None and discover_fds:
                logger.info("No functional dependencies provided. Auto-discovering...")
                functional_dependencies = discover_functional_dependencies(
                    processed_df, 
                    max_lhs_size=2, 
                    min_confidence=fd_confidence
                )
            elif functional_dependencies is None:
                logger.info("No functional dependencies provided and discovery disabled. Using empty FDs.")
                functional_dependencies = []
            
            # Validate FDs
            functional_dependencies = validate_functional_dependencies(processed_df, functional_dependencies)
            self.results['functional_dependencies'] = functional_dependencies
            
            # Step 4: Apply GGR algorithm
            logger.info("Applying GGR algorithm...")
            ggr_start_time = time.time()
            
            total_hits, reordered_table = ggr(processed_df, functional_dependencies, max_depth=max_depth)
            
            ggr_end_time = time.time()
            
            # Step 5: Store results
            self.results.update({
                'total_hits': total_hits,
                'ggr_execution_time': ggr_end_time - ggr_start_time,
                'reordered_table_size': len(reordered_table) if reordered_table else 0,
                'experiment_name': self.experiment_name,
                'parameters': {
                    'max_depth': max_depth,
                    'discover_fds': discover_fds,
                    'fd_confidence': fd_confidence,
                    'handle_missing': handle_missing,
                    'columns_of_interest': columns_of_interest
                }
            })
            
            # Step 6: Save results
            self._save_results(reordered_table, processed_df.columns.tolist())
            
            total_time = time.time() - start_time
            self.results['total_execution_time'] = total_time
            
            logger.info(f"Experiment completed successfully in {total_time:.2f} seconds")
            logger.info(f"Total hits: {total_hits}")
            logger.info(f"Reordered table size: {len(reordered_table) if reordered_table else 0}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.results['error'] = str(e)
            self._save_results([], [])
            raise
    
    def _save_results(self, reordered_table: List[List], original_columns: List[str]):
        """
        Save experiment results to files
        
        Args:
            reordered_table: The reordered table from GGR algorithm
            original_columns: Original column names
        """
        # Save reordered table as CSV
        if reordered_table:
            # Convert reordered table back to DataFrame
            reordered_df = pd.DataFrame(reordered_table, columns=original_columns)
            csv_path = os.path.join(self.output_dir, f"{self.experiment_name}_reordered_table.csv")
            reordered_df.to_csv(csv_path, index=False)
            logger.info(f"Reordered table saved to: {csv_path}")
            self.results['output_files'] = {'reordered_table_csv': csv_path}
        
        # Save experiment metadata as JSON
        metadata_path = os.path.join(self.output_dir, f"{self.experiment_name}_metadata.json")
        metadata = {k: v for k, v in self.results.items() if k != 'reordered_table'}  # Exclude large data
        
        # Convert numpy types to native Python types for JSON serialization
        metadata = self._convert_numpy_types(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Experiment metadata saved to: {metadata_path}")
        
        if 'output_files' not in self.results:
            self.results['output_files'] = {}
        self.results['output_files']['metadata_json'] = metadata_path
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        return obj


def run_single_experiment(dataset_path: str,
                         functional_dependencies: Optional[str] = None,
                         columns: Optional[str] = None,
                         output_dir: str = "results",
                         experiment_name: Optional[str] = None,
                         max_depth: int = 100,
                         discover_fds: bool = True,
                         fd_confidence: float = 0.95,
                         handle_missing: str = 'drop') -> Dict[str, Any]:
    """
    Convenience function to run a single experiment
    
    Args:
        dataset_path: Path to dataset CSV
        functional_dependencies: FDs as string (e.g., "col1->col2,col3->col4")
        columns: Columns of interest as comma-separated string
        output_dir: Output directory
        experiment_name: Name for experiment
        max_depth: Maximum recursion depth
        discover_fds: Whether to auto-discover FDs
        fd_confidence: Confidence threshold for FD discovery
        handle_missing: How to handle missing values
        
    Returns:
        Experiment results dictionary
    """
    # Parse string arguments
    fds = parse_functional_dependencies(functional_dependencies) if functional_dependencies else None
    cols = [col.strip() for col in columns.split(',')] if columns else None
    
    # Create and run experiment
    experiment = GGRExperiment(dataset_path, output_dir, experiment_name)
    return experiment.run_experiment(
        functional_dependencies=fds,
        columns_of_interest=cols,
        max_depth=max_depth,
        discover_fds=discover_fds,
        fd_confidence=fd_confidence,
        handle_missing=handle_missing
    )
