#!/usr/bin/env python3
"""
Quick test script for KVTuner integration fix
"""

import os
import sys

# Test configuration
TEST_CONFIG = {
    "model": "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct",
    "dataset": "/home/data/so2/semantic-operators/old/sampled_data/rotten_tomatoes_critic_reviews_sampled_500_20250909_150924.csv",
    "max_rows": 3,
    "gpu_ids": "6,7",
    "gpu_memory_utilization": 0.5
}

def test_basic_cache():
    """Test with basic cache mode"""
    print("=" * 60)
    print("Testing BASIC Cache Mode")
    print("=" * 60)
    
    cmd = f"""python llm_inference.py \\
  --dataset "{TEST_CONFIG['dataset']}" \\
  --model "{TEST_CONFIG['model']}" \\
  --cache_mode basic \\
  --max_rows {TEST_CONFIG['max_rows']} \\
  --gpu_ids "{TEST_CONFIG['gpu_ids']}" \\
  --gpu_memory_utilization {TEST_CONFIG['gpu_memory_utilization']}"""
    
    print("Command to run:")
    print(cmd)
    print("\nExpected result: Should load successfully with basic cache")
    return cmd

def test_kvtuner_cache():
    """Test with KVTuner cache mode"""
    print("=" * 60)
    print("Testing KVTUNER Cache Mode")
    print("=" * 60)
    
    cmd = f"""python llm_inference.py \\
  --dataset "{TEST_CONFIG['dataset']}" \\
  --model "{TEST_CONFIG['model']}" \\
  --cache_mode kvtuner \\
  --kvtuner_scheme pertoken \\
  --max_rows {TEST_CONFIG['max_rows']} \\
  --gpu_ids "{TEST_CONFIG['gpu_ids']}" \\
  --gpu_memory_utilization {TEST_CONFIG['gpu_memory_utilization']}"""
    
    print("Command to run:")
    print(cmd)
    print("\nExpected result: Should copy kvtuner_config.yaml to model directory and load with KVTuner")
    return cmd

def main():
    print("KVTuner Integration Test Commands")
    print("=" * 80)
    print(f"Model: {TEST_CONFIG['model']}")
    print(f"Dataset: {TEST_CONFIG['dataset']}")
    print(f"GPU IDs: {TEST_CONFIG['gpu_ids']}")
    print(f"Max rows: {TEST_CONFIG['max_rows']}")
    print()
    
    # Test basic cache first
    basic_cmd = test_basic_cache()
    print()
    
    # Test KVTuner cache
    kvtuner_cmd = test_kvtuner_cache()
    print()
    
    print("=" * 80)
    print("USAGE INSTRUCTIONS:")
    print("1. First run the BASIC cache test to ensure infrastructure works")
    print("2. Then run the KVTUNER cache test to verify the fix")
    print()
    print("If both work, the KVTuner integration is successfully fixed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
