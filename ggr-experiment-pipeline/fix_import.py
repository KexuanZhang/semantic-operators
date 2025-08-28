#!/usr/bin/env python3
"""
Script to copy and fix the run_experiment_enhanced.py file
to properly handle imports on the server environment.
"""

import os
import sys
import shutil
import re

def fix_import_path():
    """Fix the import path in run_experiment_enhanced.py"""
    # Path to the original file
    src_dir = os.path.dirname(os.path.abspath(__file__))
    src_file = os.path.join(src_dir, "src/experiment/run_experiment_enhanced.py")
    
    # Path for the fixed file
    fixed_file = os.path.join(src_dir, "src/experiment/run_experiment_enhanced_fixed.py")
    
    print(f"Fixing import path in {src_file}")
    
    # Read the original file
    with open(src_file, 'r') as f:
        content = f.read()
    
    # Replace the hardcoded import path with a more flexible approach
    import_section = """# Import local vLLM setup
sys.path.insert(0, '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline')
from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm

# Initialize local vLLM
logger.info("Initializing local vLLM with stats logging...")
initialize_experiment_with_local_vllm()"""

    new_import_section = """# Import local vLLM setup with dynamic path detection
logger.info("Attempting to locate and import use_local_vllm...")

# Try to find the module in parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
potential_paths = [
    # Try current directory
    current_dir,
    # Try one level up
    os.path.dirname(current_dir),
    # Try two levels up
    os.path.dirname(os.path.dirname(current_dir)),
    # Try hardcoded paths for different environments
    '/home/data/so/semantic-operators/ggr-experiment-pipeline',
    '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline'
]

module_found = False
for path in potential_paths:
    module_path = os.path.join(path, 'use_local_vllm.py')
    if os.path.exists(module_path):
        logger.info(f"Found use_local_vllm.py in: {path}")
        sys.path.insert(0, path)
        try:
            from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm
            module_found = True
            logger.info(f"Successfully imported use_local_vllm from {path}")
            break
        except ImportError as e:
            logger.warning(f"Found module file but import failed: {e}")

if not module_found:
    logger.error("Could not locate use_local_vllm.py in any of the expected paths")
    raise ImportError("Failed to import use_local_vllm. Ensure the file exists and is in PYTHONPATH")

# Initialize local vLLM
logger.info("Initializing local vLLM with stats logging...")
initialize_experiment_with_local_vllm()"""

    # Replace the import section
    new_content = content.replace(import_section, new_import_section)
    
    # Write the fixed file
    with open(fixed_file, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed file saved to: {fixed_file}")
    print("To use the fixed file, run:")
    print(f"python {fixed_file} [your arguments]")

if __name__ == "__main__":
    fix_import_path()
