"""
Data Preprocessing Module for GGR Experiment Pipeline
Handles data loading, functional dependency discovery, and preprocessing
"""
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame containing the data
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def discover_functional_dependencies(df: pd.DataFrame, 
                                   max_lhs_size: int = 2,
                                   min_confidence: float = 0.95) -> List[Tuple[str, str]]:
    """
    Discover functional dependencies in the dataset
    
    Args:
        df: Input DataFrame
        max_lhs_size: Maximum size of left-hand side of FD
        min_confidence: Minimum confidence threshold for FD
        
    Returns:
        List of functional dependencies as (source, target) tuples
    """
    logger.info("Discovering functional dependencies...")
    functional_dependencies = []
    
    # Get all columns
    columns = df.columns.tolist()
    
    # For each possible target column
    for target_col in columns:
        # Try combinations of other columns as sources
        other_cols = [col for col in columns if col != target_col]
        
        for lhs_size in range(1, min(max_lhs_size + 1, len(other_cols) + 1)):
            for source_cols in combinations(other_cols, lhs_size):
                # Check if source_cols -> target_col is a valid FD
                if check_functional_dependency(df, list(source_cols), target_col, min_confidence):
                    # For simplicity, we'll store single-column FDs
                    if len(source_cols) == 1:
                        functional_dependencies.append((source_cols[0], target_col))
                    else:
                        # For multi-column FDs, create a composite key representation
                        composite_key = "_".join(source_cols)
                        functional_dependencies.append((composite_key, target_col))
    
    logger.info(f"Discovered {len(functional_dependencies)} functional dependencies")
    return functional_dependencies


def check_functional_dependency(df: pd.DataFrame, 
                              source_cols: List[str], 
                              target_col: str, 
                              min_confidence: float = 0.95) -> bool:
    """
    Check if source_cols -> target_col is a functional dependency
    
    Args:
        df: Input DataFrame
        source_cols: Source columns (left-hand side)
        target_col: Target column (right-hand side)
        min_confidence: Minimum confidence threshold
        
    Returns:
        True if FD holds with given confidence
    """
    # Group by source columns and check uniqueness of target
    grouped = df.groupby(source_cols)[target_col].nunique()
    
    # Calculate confidence: how often source determines target uniquely
    violations = (grouped > 1).sum()
    total_groups = len(grouped)
    
    if total_groups == 0:
        return False
        
    confidence = (total_groups - violations) / total_groups
    return confidence >= min_confidence


def preprocess_data(df: pd.DataFrame, 
                   columns_of_interest: Optional[List[str]] = None,
                   handle_missing: str = 'drop') -> pd.DataFrame:
    """
    Preprocess the data for GGR algorithm
    
    Args:
        df: Input DataFrame
        columns_of_interest: Specific columns to include (if None, use all)
        handle_missing: How to handle missing values ('drop', 'fill', 'keep')
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing data...")
    
    # Select columns of interest
    if columns_of_interest:
        logger.info(f"Selecting columns: {columns_of_interest}")
        df = df[columns_of_interest].copy()
    
    # Handle missing values
    if handle_missing == 'drop':
        logger.info("Dropping rows with missing values")
        df = df.dropna()
    elif handle_missing == 'fill':
        logger.info("Filling missing values")
        # Fill numeric columns with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    # If 'keep', do nothing with missing values
    
    # Convert all columns to string for consistency with GGR algorithm
    logger.info("Converting all columns to string type")
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def parse_functional_dependencies(fd_string: str) -> List[Tuple[str, str]]:
    """
    Parse functional dependencies from command line string format
    
    Args:
        fd_string: String in format "col1->col2,col3->col4"
        
    Returns:
        List of functional dependencies as tuples
    """
    if not fd_string:
        return []
    
    fds = []
    fd_pairs = fd_string.split(',')
    
    for pair in fd_pairs:
        if '->' in pair:
            source, target = pair.split('->', 1)
            fds.append((source.strip(), target.strip()))
        else:
            logger.warning(f"Invalid FD format: {pair}. Expected format: source->target")
    
    return fds


def validate_functional_dependencies(df: pd.DataFrame, 
                                   functional_dependencies: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Validate that functional dependencies reference existing columns
    
    Args:
        df: Input DataFrame
        functional_dependencies: List of FD tuples
        
    Returns:
        Filtered list of valid FDs
    """
    valid_fds = []
    columns = set(df.columns)
    
    for source, target in functional_dependencies:
        if source in columns and target in columns:
            valid_fds.append((source, target))
        else:
            logger.warning(f"Skipping invalid FD {source}->{target}: columns not found in dataset")
    
    logger.info(f"Validated {len(valid_fds)} out of {len(functional_dependencies)} functional dependencies")
    return valid_fds
