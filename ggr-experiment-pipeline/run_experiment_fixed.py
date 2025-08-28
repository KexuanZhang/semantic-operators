#!/usr/bin/env python3
"""
Runner script to properly execute run_experiment_enhanced.py with fixed imports.
This script wraps the original script and handles imports correctly on different environments.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_file(filename, search_paths=None):
    """Find a file in the given search paths."""
    if search_paths is None:
        # Default search paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            current_dir,
            os.path.join(current_dir, 'src/experiment'),
            os.path.join(current_dir, 'src'),
            '/home/data/so/semantic-operators/ggr-experiment-pipeline/src/experiment',
            '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline/src/experiment'
        ]
    
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    
    return None

def fix_import_and_run():
    """Fix the import path in run_experiment_enhanced.py and run it."""
    # Find the original script
    script_path = find_file('run_experiment_enhanced.py')
    if not script_path:
        logger.error("Could not find run_experiment_enhanced.py")
        sys.exit(1)
    
    logger.info(f"Found original script at: {script_path}")
    
    # Find use_local_vllm.py
    vllm_module = find_file('use_local_vllm.py', [
        os.path.dirname(script_path),
        os.path.dirname(os.path.dirname(script_path)),
        os.path.dirname(os.path.dirname(os.path.dirname(script_path))),
        '/home/data/so/semantic-operators/ggr-experiment-pipeline',
        '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline'
    ])
    
    if not vllm_module:
        logger.error("Could not find use_local_vllm.py")
        sys.exit(1)
    
    logger.info(f"Found vLLM module at: {vllm_module}")
    
    # Create a temporary directory for the fixed script
    temp_dir = tempfile.mkdtemp()
    fixed_script = os.path.join(temp_dir, 'run_experiment_enhanced_fixed.py')
    
    logger.info(f"Creating fixed script at: {fixed_script}")
    
    try:
        # Read the original script
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Replace the hardcoded import path
        import_pattern = r"# Import local vLLM setup.*?initialize_experiment_with_local_vllm\(\)"
        vllm_module_dir = os.path.dirname(vllm_module)
        
        new_import_section = f"""# Import local vLLM setup with dynamic path detection
logger.info("Using vLLM module from: {vllm_module_dir}")
sys.path.insert(0, "{vllm_module_dir}")
from use_local_vllm import initialize_experiment_with_local_vllm, create_enhanced_llm

# Initialize local vLLM
logger.info("Initializing local vLLM with stats logging...")
initialize_experiment_with_local_vllm()"""
        
        # Use a simple string replacement first
        if "sys.path.insert(0, '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline')" in content:
            new_content = content.replace(
                "sys.path.insert(0, '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline')",
                f'sys.path.insert(0, "{vllm_module_dir}")'
            )
        else:
            # Fall back to regex for more complex cases
            import re
            new_content = re.sub(import_pattern, new_import_section, content, flags=re.DOTALL)
        
        # Write the fixed script
        with open(fixed_script, 'w') as f:
            f.write(new_content)
        
        # Make the script executable
        os.chmod(fixed_script, 0o755)
        
        # Pass all command line arguments to the fixed script
        cmd = [sys.executable, fixed_script] + sys.argv[1:]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the fixed script
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
        
    finally:
        # Clean up
        try:
            shutil.rmtree(temp_dir)
        except:
            logger.warning(f"Failed to clean up temporary directory: {temp_dir}")

if __name__ == "__main__":
    fix_import_and_run()
