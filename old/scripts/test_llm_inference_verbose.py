#!/usr/bin/env python3
"""
Test script to verify that verbose output has been removed from llm_inference.py
"""

import subprocess
import sys
import os
import tempfile
import csv

def create_test_dataset():
    """Create a small test CSV dataset"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['text_content', 'id'])
        writer.writerow(['This is a test sentence.', '1'])
        writer.writerow(['Another test sentence.', '2'])
        writer.writerow(['Final test sentence.', '3'])
        return f.name

def test_verbose_removal():
    """Test that verbose output is removed"""
    
    # Create test dataset
    test_csv = create_test_dataset()
    
    try:
        # Path to the llm_inference script
        script_path = "/Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/old/scripts/llm_inference.py"
        
        # Run the script with dry-run parameters (will likely fail due to missing dependencies, but we can check output)
        cmd = [
            sys.executable, script_path,
            "--dataset", test_csv,
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--max_rows", "1",
            "--help"  # Just show help to test basic parsing
        ]
        
        # Run with help to test argument parsing
        result = subprocess.run([sys.executable, script_path, "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        print("=== ARGUMENT PARSING TEST ===")
        if result.returncode == 0:
            print("✓ Script can be imported and argument parsing works")
        else:
            print("✗ Script has syntax or import errors")
            print("STDERR:", result.stderr)
        
        # Check if verbose strings are still in the code
        with open(script_path, 'r') as f:
            content = f.read()
            
        verbose_patterns = [
            "print(f\"Using specific GPU IDs:",
            "print(f\"Initializing LLM from",
            "print(f\"Model {model_name} loaded successfully",
            "print(f\"Using tensor parallelism",
            "print(f\"Loading dataset from",
            "print(f\"Auto-detected content columns:",
            "print(f\"Running LLM inference on dataset",
            "print(f\"Total experiment time:",
            "print(f\"Processed {index} rows",
            "print(f\"\\n----- INFERENCE REQUEST -----",
            "print(f\"----- RAW RESPONSE -----",
            "print(f\"----- CLEANED RESPONSE -----",
            "print(\"WARNING: Response was empty",
            "print(\"ERROR: Model returned empty"
        ]
        
        print("\n=== VERBOSE OUTPUT REMOVAL TEST ===")
        found_verbose = []
        for pattern in verbose_patterns:
            if pattern in content:
                found_verbose.append(pattern)
        
        if not found_verbose:
            print("✓ All verbose print statements have been removed")
        else:
            print("✗ Some verbose print statements still exist:")
            for pattern in found_verbose:
                print(f"  - {pattern}")
        
        # Check if tqdm import was added
        if "from tqdm import tqdm" in content:
            print("✓ tqdm import added for progress bar")
        else:
            print("✗ tqdm import not found")
            
        # Check if progress bar is used
        if "with tqdm(" in content:
            print("✓ Progress bar implementation found")
        else:
            print("✗ Progress bar implementation not found")
            
    except subprocess.TimeoutExpired:
        print("✗ Script timed out")
    except Exception as e:
        print(f"✗ Error running test: {e}")
    finally:
        # Clean up test file
        try:
            os.unlink(test_csv)
        except:
            pass

if __name__ == "__main__":
    test_verbose_removal()
