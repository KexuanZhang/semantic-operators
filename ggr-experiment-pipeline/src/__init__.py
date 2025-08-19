"""
GGR Experiment Pipeline Package
"""

from .ggr_algorithm import ggr, calculate_hit_count
from .data_preprocessing import (
    load_dataset,
    discover_functional_dependencies,
    preprocess_data,
    parse_functional_dependencies,
    validate_functional_dependencies
)
from .experiment_runner import GGRExperiment, run_single_experiment

__version__ = "1.0.0"
__all__ = [
    'ggr',
    'calculate_hit_count',
    'load_dataset',
    'discover_functional_dependencies',
    'preprocess_data',
    'parse_functional_dependencies',
    'validate_functional_dependencies',
    'GGRExperiment',
    'run_single_experiment'
]
