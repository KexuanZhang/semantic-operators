#!/usr/bin/env python3
"""
Final Testing Guide for KVTuner Integration Fix

This script provides comprehensive testing instructions and commands 
for verifying that the KVTuner integration is working correctly.
"""

import os

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_subsection(title):
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print(f"{'-' * 60}")

def main():
    print("KVTuner Integration Fix - Final Testing Guide")
    print("=" * 80)
    print("This guide provides step-by-step testing instructions for the KVTuner integration fix.")
    
    print_section("PREREQUISITES")
    print("1. All code changes have been applied to vLLM and the inference script")
    print("2. You have access to the target machine with GPU resources")
    print("3. Model and dataset paths are accessible")
    print("4. Required Python packages are installed (PyYAML, vLLM, etc.)")
    
    print_section("TESTING SEQUENCE")
    
    print_subsection("Step 1: Verify Environment")
    print("Check that all required paths exist:")
    print("  • Model directory: /home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct")
    print("  • Dataset file: /home/data/so2/semantic-operators/old/sampled_data/rotten_tomatoes_critic_reviews_sampled_500_20250909_150924.csv")
    print("  • KVTuner directory: /home/data/so2/semantic-operators/KVTuner")
    print("  • GPU availability: nvidia-smi should show GPUs 6,7")
    
    print_subsection("Step 2: Test Basic Cache Mode (Baseline)")
    print("First, verify that the basic functionality works:")
    print()
    basic_cmd = """cd /home/data/so2/semantic-operators/kvtuner && python llm_inference.py \\
  --dataset "/home/data/so2/semantic-operators/old/sampled_data/rotten_tomatoes_critic_reviews_sampled_500_20250909_150924.csv" \\
  --model "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct" \\
  --cache_mode basic \\
  --max_rows 3 \\
  --gpu_ids "6,7" \\
  --gpu_memory_utilization 0.5"""
    print(basic_cmd)
    print()
    print("Expected Result:")
    print("  ✓ Model loads successfully")
    print("  ✓ Processes 3 rows from dataset")  
    print("  ✓ No quantization errors")
    print("  ✓ Generates text completions")
    
    print_subsection("Step 3: Test KVTuner Cache Mode (Main Fix)")
    print("Now test the KVTuner integration:")
    print()
    kvtuner_cmd = """cd /home/data/so2/semantic-operators/kvtuner && python llm_inference.py \\
  --dataset "/home/data/so2/semantic-operators/old/sampled_data/rotten_tomatoes_critic_reviews_sampled_500_20250909_150924.csv" \\
  --model "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct" \\
  --cache_mode kvtuner \\
  --kvtuner_scheme pertoken \\
  --max_rows 3 \\
  --gpu_ids "6,7" \\
  --gpu_memory_utilization 0.5"""
    print(kvtuner_cmd)
    print()
    print("Expected Result:")
    print("  ✓ Finds KVTuner config: Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml")
    print("  ✓ Copies config to model directory as kvtuner_config.yaml")
    print("  ✓ Creates JSON version as kvtuner_config.json")
    print("  ✓ Shows debug info: 'All config files in model dir: [...]'")
    print("  ✓ Model loads with quantization='kvtuner'")
    print("  ✓ No 'missing argument self' errors")
    print("  ✓ Processes data successfully")
    
    print_subsection("Step 4: Test Alternative KVTuner Schemes")
    print("Test other KVTuner schemes to ensure broad compatibility:")
    print()
    
    schemes = ["kivi", "pertoken"]
    for scheme in schemes:
        cmd = f"""python llm_inference.py \\
  --model "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct" \\
  --cache_mode kvtuner \\
  --kvtuner_scheme {scheme} \\
  --max_rows 1 \\
  --gpu_ids "6,7" \\
  --gpu_memory_utilization 0.5 \\
  --dataset "/home/data/so2/semantic-operators/old/sampled_data/rotten_tomatoes_critic_reviews_sampled_500_20250909_150924.csv\""""
        print(f"Scheme: {scheme}")
        print(cmd)
        print()
    
    print_section("DEBUGGING GUIDE")
    
    print_subsection("If Step 2 (Basic Mode) Fails:")
    print("• Check model path exists and is readable")
    print("• Verify GPU availability with: nvidia-smi")
    print("• Check dataset file format and accessibility") 
    print("• Review vLLM installation and dependencies")
    
    print_subsection("If Step 3 (KVTuner Mode) Fails:")
    print("Check these potential issues in order:")
    print()
    print("1. CONFIG FILE NOT FOUND:")
    print("   Error: 'No KVTuner config found for Qwen2.5-3B-Instruct'")
    print("   Solution: Check KVTuner calibration_presets directory")
    print("   Command: ls /home/data/so2/semantic-operators/KVTuner/calibration_presets/Qwen*")
    print()
    print("2. PERMISSION DENIED:")
    print("   Error: 'Permission denied' when copying config")
    print("   Solution: Check write permissions on model directory")
    print("   Command: ls -la /home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct/")
    print()
    print("3. MISSING SELF ARGUMENT (Original Error):")
    print("   Error: 'get_config_filenames() missing 1 required positional argument: self'")
    print("   Solution: Verify that the method signature fixes were applied correctly")
    print()
    print("4. YAML IMPORT ERROR:")
    print("   Error: 'No module named yaml'")
    print("   Solution: pip install PyYAML")
    print()
    print("5. CONFIG NOT FOUND BY VLLM:")
    print("   Error: 'Cannot find the config file for kvtuner'")
    print("   Debug: Check the log output for file discovery messages")
    print("   Look for: 'KVTuner: Found config files: [...]'")
    
    print_section("SUCCESS INDICATORS")
    print("The fix is working correctly if you see:")
    print()
    print("✓ Config file copying messages:")
    print("    'Copied KVTuner YAML config to: .../kvtuner_config.yaml'")
    print("    'Created KVTuner JSON config at: .../kvtuner_config.json'")
    print()
    print("✓ File verification messages:")
    print("    'YAML file exists: True'")
    print("    'JSON file exists: True'")
    print("    'All config files in model dir: [kvtuner_config.json, kvtuner_config.yaml, ...]'")
    print()
    print("✓ vLLM debug messages:")
    print("    'KVTuner: Searching for config files in: ...'")
    print("    'KVTuner: Found config files: [.../kvtuner_config.yaml]'")
    print()
    print("✓ Model loading without errors:")
    print("    No 'missing argument self' errors")
    print("    No 'Cannot find the config file' errors")
    print("    Successful text generation")
    
    print_section("PERFORMANCE VALIDATION")
    print("After confirming the fix works, validate performance:")
    print()
    print("1. Compare memory usage between basic and KVTuner modes")
    print("2. Verify that KVTuner actually reduces memory consumption")
    print("3. Check that text quality is maintained")
    print("4. Test with larger datasets to confirm scalability")
    
    print_section("QUICK REFERENCE COMMANDS")
    print()
    print("# Navigate to working directory:")
    print("cd /home/data/so2/semantic-operators/kvtuner")
    print()
    print("# Test basic mode:")
    print("python llm_inference.py --model /path/to/model --cache_mode basic --max_rows 1 --gpu_ids 6,7 --dataset /path/to/dataset.csv")
    print()
    print("# Test KVTuner mode:")
    print("python llm_inference.py --model /path/to/model --cache_mode kvtuner --kvtuner_scheme pertoken --max_rows 1 --gpu_ids 6,7 --dataset /path/to/dataset.csv")
    print()
    print("# Check GPU usage:")
    print("nvidia-smi")
    print()
    print("# Check file permissions:")
    print("ls -la /home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct/")
    print()
    print("# List available configs:")
    print("ls /home/data/so2/semantic-operators/KVTuner/calibration_presets/Qwen*")
    
    print("\n" + "=" * 80)
    print("END OF TESTING GUIDE")
    print("=" * 80)
    print("If all tests pass, the KVTuner integration fix is successfully implemented!")

if __name__ == "__main__":
    main()
