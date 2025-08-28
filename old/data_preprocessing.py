import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FunctionalDependencyDiscovery:
    """Class to discover functional dependencies in a dataset."""
    
    def __init__(self, threshold: float = 0.95):
        """
        Initialize the FD discovery with a threshold for determining dependencies.
        
        Args:
            threshold: Minimum ratio of unique mappings to consider a functional dependency
        """
        self.threshold = threshold
    
    def discover_functional_dependencies(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Discover functional dependencies in the dataset.
        A functional dependency X -> Y means that for each value of X, 
        there is only one corresponding value of Y.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of tuples representing functional dependencies (source, target)
        """
        logger.info("Starting functional dependency discovery...")
        functional_dependencies = []
        columns = df.columns.tolist()
        
        # Check single column dependencies first
        for source_col in columns:
            for target_col in columns:
                if source_col != target_col:
                    if self._is_functional_dependency(df, source_col, target_col):
                        functional_dependencies.append((source_col, target_col))
        
        logger.info(f"Discovered {len(functional_dependencies)} functional dependencies")
        for fd in functional_dependencies:
            logger.info(f"  {fd[0]} -> {fd[1]}")
            
        return functional_dependencies
    
    def _is_functional_dependency(self, df: pd.DataFrame, source_col: str, target_col: str) -> bool:
        """
        Check if source_col functionally determines target_col.
        
        Args:
            df: Input DataFrame
            source_col: Source column name
            target_col: Target column name
            
        Returns:
            True if functional dependency exists
        """
        # Remove rows with NaN in either column for this check
        clean_df = df[[source_col, target_col]].dropna()
        
        if clean_df.empty:
            return False
            
        # Group by source column and check if target column has unique values
        grouped = clean_df.groupby(source_col)[target_col].nunique()
        
        # Check if all groups have exactly one unique value in target column
        fd_ratio = (grouped == 1).sum() / len(grouped)
        
        return fd_ratio >= self.threshold


class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self, dataset_path: str, functional_dependencies: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            dataset_path: Path to the CSV dataset
            functional_dependencies: Optional list of functional dependencies
        """
        self.dataset_path = dataset_path
        self.functional_dependencies = functional_dependencies
        self.df = None
        self.fd_discovery = FunctionalDependencyDiscovery()
    
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        try:
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and data types.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        if self.df is None:
            self.load_data()
        
        # Handle missing values - convert to string representation for consistency
        preprocessed_df = self.df.copy()
        
        # Convert all columns to string type for consistency with GGR algorithm
        for col in preprocessed_df.columns:
            preprocessed_df[col] = preprocessed_df[col].astype(str)
        
        # Replace 'nan' strings back to actual NaN for proper handling
        preprocessed_df = preprocessed_df.replace('nan', np.nan)
        
        logger.info(f"Preprocessing complete. Final shape: {preprocessed_df.shape}")
        return preprocessed_df
    
    def get_functional_dependencies(self) -> List[Tuple[str, str]]:
        """
        Get functional dependencies either from input or by discovery.
        
        Returns:
            List of functional dependencies
        """
        if self.functional_dependencies is not None:
            logger.info("Using provided functional dependencies")
            return self.functional_dependencies
        
        logger.info("No functional dependencies provided. Starting discovery...")
        if self.df is None:
            self.load_data()
        
        return self.fd_discovery.discover_functional_dependencies(self.df)
    
    def get_preprocessing_summary(self) -> dict:
        """
        Get a summary of the preprocessing results.
        
        Returns:
            Dictionary containing preprocessing statistics
        """
        if self.df is None:
            return {}
        
        summary = {
            'dataset_path': self.dataset_path,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'missing_values_per_column': self.df.isnull().sum().to_dict(),
            'functional_dependencies': self.get_functional_dependencies()
        }
        
        return summary


def parse_functional_dependencies(fd_string: str) -> List[Tuple[str, str]]:
    """
    Parse functional dependencies from command line string format.
    Expected format: "col1->col2,col3->col4"
    
    Args:
        fd_string: String representation of functional dependencies
        
    Returns:
        List of functional dependency tuples
    """
    if not fd_string:
        return None
    
    dependencies = []
    pairs = fd_string.split(',')
    
    for pair in pairs:
        if '->' in pair:
            source, target = pair.strip().split('->')
            dependencies.append((source.strip(), target.strip()))
        else:
            logger.warning(f"Invalid functional dependency format: {pair}")
    
    return dependencies


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Data Preprocessing for GGR Algorithm')
    parser.add_argument('dataset_path', help='Path to the CSV dataset')
    parser.add_argument('--functional_dependencies', '-fd', type=str, 
                       help='Functional dependencies in format "col1->col2,col3->col4"')
    parser.add_argument('--output_summary', '-o', type=str, default='preprocessing_summary.txt',
                       help='Output file for preprocessing summary')
    parser.add_argument('--discovery_threshold', '-t', type=float, default=0.95,
                       help='Threshold for functional dependency discovery (default: 0.95)')
    
    args = parser.parse_args()
    
    # Parse functional dependencies if provided
    functional_dependencies = parse_functional_dependencies(args.functional_dependencies)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(args.dataset_path, functional_dependencies)
    preprocessor.fd_discovery.threshold = args.discovery_threshold
    
    # Run preprocessing
    try:
        preprocessed_df = preprocessor.preprocess_data()
        fds = preprocessor.get_functional_dependencies()
        summary = preprocessor.get_preprocessing_summary()
        
        # Save summary
        with open(args.output_summary, 'w') as f:
            f.write("Data Preprocessing Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset Path: {summary['dataset_path']}\n")
            f.write(f"Total Rows: {summary['total_rows']}\n")
            f.write(f"Total Columns: {summary['total_columns']}\n")
            f.write(f"Columns: {', '.join(summary['columns'])}\n\n")
            
            f.write("Missing Values per Column:\n")
            for col, missing in summary['missing_values_per_column'].items():
                f.write(f"  {col}: {missing}\n")
            
            f.write(f"\nFunctional Dependencies ({len(fds)}):\n")
            for source, target in fds:
                f.write(f"  {source} -> {target}\n")
        
        logger.info(f"Preprocessing summary saved to {args.output_summary}")
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
