#!/usr/bin/env python3
"""
Integration script for using local modified vLLM in the semantic-operators experiment pipeline.
This script should be imported at the beginning of your experiment scripts.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def setup_local_vllm():
    """
    Setup the local modified vLLM for use in experiments.
    This function should be called before importing any vLLM modules.
    """
    # Path to the local vLLM installation - try multiple potential locations
    # First check for environment variable
    vllm_path = None
    if 'VLLM_PATH' in os.environ:
        vllm_path = Path(os.environ['VLLM_PATH'])
        logger.info(f"Using vLLM path from VLLM_PATH environment variable: {vllm_path}")
    
    # If not set via environment variable, try common relative paths
    if vllm_path is None or not vllm_path.exists():
        # Try relative to the current script's location
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            # Original path for backward compatibility
            Path("/Users/zhang/Desktop/huawei/so1/vllm"),
            # Server path based on error message
            Path("/home/data/so/vllm"),
            # One level up from the current directory
            current_dir.parent / "vllm",
            # Two levels up from the current directory
            current_dir.parent.parent / "vllm",
            # Adjacent to semantic-operators
            current_dir.parent.parent / "vllm"
        ]
        
        for candidate in candidates:
            if candidate.exists():
                vllm_path = candidate
                logger.info(f"Found vLLM at: {vllm_path}")
                break
    
    # Check if local vLLM exists after all attempts
    if vllm_path is None or not vllm_path.exists():
        raise FileNotFoundError(
            f"Local vLLM not found. Tried multiple locations. "
            f"Please set VLLM_PATH environment variable to point to your vLLM installation.")
    
    # Add to Python path (insert at beginning to take priority)
    vllm_path_str = str(vllm_path)
    if vllm_path_str not in sys.path:
        sys.path.insert(0, vllm_path_str)
        logger.info(f"Added local vLLM path to Python path: {vllm_path_str}")
    
    # Verify we can import the modified vLLM
    try:
        import vllm
        logger.info(f"Successfully imported local vLLM version: {vllm.__version__}")
        
        # Test custom wrapper import
        from vllm.offline_llm_with_stats import OfflineLLMWithStats
        logger.info("Successfully imported OfflineLLMWithStats wrapper")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import local vLLM: {e}")
        return False

def get_vllm_with_stats():
    """
    Get the enhanced LLM class with stats logging.
    
    Returns:
        OfflineLLMWithStats class
    """
    from vllm.offline_llm_with_stats import OfflineLLMWithStats
    return OfflineLLMWithStats

def create_enhanced_llm(model_path: str, **kwargs):
    """
    Create an enhanced LLM instance with stats logging enabled.
    
    Args:
        model_path: Path to the model
        **kwargs: Additional arguments passed to LLM constructor
        
    Returns:
        OfflineLLMWithStats instance
    """
    # Handle both log_stats and disable_log_stats parameters for compatibility
    if 'disable_log_stats' in kwargs:
        # If disable_log_stats is provided, use it to set log_stats
        log_stats_enabled = not kwargs.pop('disable_log_stats')
        kwargs['log_stats'] = log_stats_enabled
    else:
        # Otherwise use the default
        kwargs.setdefault('log_stats', True)
    
    kwargs.setdefault('log_stats_interval', 1)
    
    # Create the enhanced LLM
    OfflineLLMWithStats = get_vllm_with_stats()
    return OfflineLLMWithStats(model=model_path, **kwargs)

# Example usage function for the experiment pipeline
def run_inference_with_stats(model_path: str, prompts: list, sampling_params=None, **llm_kwargs):
    """
    Run inference with detailed stats logging.
    
    Args:
        model_path: Path to the model
        prompts: List of prompts to process
        sampling_params: vLLM SamplingParams object
        **llm_kwargs: Additional LLM initialization arguments
        
    Returns:
        tuple: (outputs, stats_dict)
    """
    from vllm import SamplingParams
    
    # Create sampling params if not provided
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
    
    # Create enhanced LLM
    llm = create_enhanced_llm(model_path, **llm_kwargs)
    
    # Run inference
    logger.info(f"Running inference on {len(prompts)} prompts with stats logging...")
    outputs = llm.generate(prompts, sampling_params, log_detailed_stats=True)
    
    # Get final stats
    stats = llm.get_current_stats()
    
    return outputs, stats

# Initialization function to be called by experiment scripts
def initialize_experiment_with_local_vllm():
    """
    Initialize the experiment environment with local vLLM.
    Call this at the beginning of your experiment scripts.
    """
    logger.info("Initializing experiment with local modified vLLM...")
    
    success = setup_local_vllm()
    if not success:
        raise RuntimeError("Failed to setup local vLLM")
    
    logger.info("✅ Local vLLM setup completed successfully")
    return True

if __name__ == "__main__":
    # Test the setup
    logging.basicConfig(level=logging.INFO)
    
    try:
        initialize_experiment_with_local_vllm()
        print("✅ Local vLLM integration test successful!")
        
        # Show available functions
        print("\n📝 Available functions:")
        print("  - setup_local_vllm(): Setup local vLLM path")
        print("  - get_vllm_with_stats(): Get enhanced LLM class")
        print("  - create_enhanced_llm(model_path, **kwargs): Create enhanced LLM instance")
        print("  - run_inference_with_stats(model_path, prompts, ...): Run inference with stats")
        print("  - initialize_experiment_with_local_vllm(): Initialize experiment environment")
        
        print("\n💡 Usage in your experiment scripts:")
        print("  from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm")
        print("  initialize_experiment_with_local_vllm()")
        print("  llm = create_enhanced_llm('your/model/path')")
        
    except Exception as e:
        print(f"❌ Local vLLM integration test failed: {e}")
        sys.exit(1)
