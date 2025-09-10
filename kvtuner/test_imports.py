#!/usr/bin/env python3

import os
import sys
import argparse

# Set up KVTuner path
kvtuner_path = "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner"
sys.path.append(kvtuner_path)
print(f"Added path: {kvtuner_path}")

# Check for search modules
modules = ['search_brute_force', 'search_optuna_vanilla', 'search_optuna_adaptive']
for module in modules:
    try:
        module_obj = __import__(module)
        print(f"Successfully imported {module}")
        
        # Check if run_search function exists
        if hasattr(module_obj, 'run_search'):
            print(f"  ✓ Found run_search function in {module}")
        else:
            print(f"  ✗ run_search function not found in {module}")
    except ImportError as e:
        print(f"Failed to import {module}: {e}")

def main():
    print("\nTest complete. If all modules imported successfully, the scripts should work correctly.")

if __name__ == "__main__":
    main()
