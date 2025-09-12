#!/usr/bin/env python3
"""
Test script to verify that verbose output suppression is working correctly.

This script runs a minimal inference to check if progress bars and verbose
output are properly suppressed.
"""

import os
import sys
import pandas as pd
import tempfile

# Add the same suppression as in the main script
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_SHOW_PROGRESS_BARS"] = "0"
os.environ["VLLM_DISABLE_TQDM"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["VLLM_TRACE_FUNCTION"] = "0"

import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Disable tqdm globally by monkey-patching
import tqdm
tqdm.tqdm.__init__ = lambda self, *args, **kwargs: None
tqdm.tqdm.update = lambda self, *args, **kwargs: None
tqdm.tqdm.close = lambda self, *args, **kwargs: None
tqdm.tqdm.__enter__ = lambda self: self
tqdm.tqdm.__exit__ = lambda self, *args, **kwargs: None

def create_test_dataset():
    """Create a small test dataset"""
    data = {
        'text_content': [
            'This is a positive review of the product.',
            'This is a negative comment about the service.',
            'This is a neutral statement about the company.'
        ]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name

def main():
    print("Testing verbose output suppression...")
    
    # Create test dataset
    test_dataset = create_test_dataset()
    print(f"Created test dataset: {test_dataset}")
    
    # Run the inference script with basic cache mode (safer for testing)
    cmd = [
        sys.executable, "llm_inference.py",
        "--dataset", test_dataset,
        "--model", "facebook/opt-125m",  # Small model for testing
        "--cache_mode", "basic",
        "--max_rows", "2",
        "--gpu_memory_utilization", "0.3",
        "--max_new_tokens", "50"
    ]
    
    print("\nRunning inference script with verbose suppression...")
    print("Command:", " ".join(cmd))
    
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("\n" + "="*50)
        print("STDOUT:")
        print("="*50)
        print(result.stdout)
        
        if result.stderr:
            print("\n" + "="*50)
            print("STDERR:")
            print("="*50)
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        # Check for common verbose patterns
        stdout_lines = result.stdout.split('\n')
        verbose_patterns = ['processed prompts', 'adding requests', 'tqdm', '%|']
        verbose_found = []
        
        for line in stdout_lines:
            for pattern in verbose_patterns:
                if pattern.lower() in line.lower():
                    verbose_found.append(line.strip())
        
        if verbose_found:
            print("\nWARNING: Found potential verbose output:")
            for line in verbose_found:
                print(f"  - {line}")
        else:
            print("\n✓ No verbose output patterns detected!")
            
    except subprocess.TimeoutExpired:
        print("ERROR: Script timed out after 5 minutes")
    except Exception as e:
        print(f"ERROR: Failed to run script: {e}")
    finally:
        # Clean up test file
        if os.path.exists(test_dataset):
            os.unlink(test_dataset)
            print(f"Cleaned up test dataset: {test_dataset}")

if __name__ == "__main__":
    main()
