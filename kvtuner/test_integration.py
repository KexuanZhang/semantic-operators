#!/usr/bin/env python3
"""
Test script to verify KVTuner integration is working correctly.
This script tests imports and basic functionality without running full inference.
"""

import sys
import os
from pathlib import Path

# Add KVTuner to Python path
sys.path.append("/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner")

def test_imports():
    """Test that required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from transformers.cache_utils import CacheConfig, QuantizedCacheConfig
        print("✓ Transformers cache classes imported successfully")
    except ImportError as e:
        print(f"✗ Transformers cache import failed: {e}")
        print("  Note: This is expected if you don't have the correct transformers version")
    
    try:
        from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
        print("✓ KVTuner flexible_quant imported successfully")
    except ImportError as e:
        print(f"✗ KVTuner import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import yaml
        from tqdm import tqdm
        print("✓ Additional dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Additional dependencies import failed: {e}")
        return False
    
    return True

def test_paths():
    """Test that required paths exist"""
    print("\nTesting paths...")
    
    # Test KVTuner directory
    kvtuner_path = Path("/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner")
    if kvtuner_path.exists():
        print(f"✓ KVTuner directory found: {kvtuner_path}")
    else:
        print(f"✗ KVTuner directory not found: {kvtuner_path}")
        return False
    
    # Test model directory
    model_path = Path("/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct")
    if model_path.exists():
        print(f"✓ Model directory found: {model_path}")
    else:
        print(f"⚠ Model directory not found: {model_path}")
        print("  You'll need to update the model path in the scripts")
    
    # Test calibration presets
    preset_dir = kvtuner_path / "calibration_presets"
    if preset_dir.exists():
        print(f"✓ Calibration presets directory found: {preset_dir}")
        
        # Check for Qwen2.5-3B-Instruct presets
        qwen_presets = list(preset_dir.glob("Qwen2.5-3B-Instruct_*.yaml"))
        if qwen_presets:
            print(f"✓ Found {len(qwen_presets)} Qwen2.5-3B-Instruct preset files:")
            for preset in qwen_presets[:3]:  # Show first 3
                print(f"    - {preset.name}")
            if len(qwen_presets) > 3:
                print(f"    ... and {len(qwen_presets) - 3} more")
        else:
            print("⚠ No Qwen2.5-3B-Instruct preset files found")
    else:
        print(f"✗ Calibration presets directory not found: {preset_dir}")
        return False
    
    return True

def test_config_loading():
    """Test loading a sample configuration"""
    print("\nTesting configuration loading...")
    
    try:
        from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig
        
        # Test basic config creation
        config = FlexibleQuantizedCacheConfig(
            device="cuda",
            per_layer_quant=True,
            asym=True,
            axis_key=0,
            axis_value=0,
            q_group_size=-1,
            residual_length=0
        )
        print("✓ FlexibleQuantizedCacheConfig created successfully")
        
        # Test with preset file if available
        preset_path = Path("/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml")
        if preset_path.exists():
            import yaml
            with open(preset_path, 'r') as f:
                preset_data = yaml.safe_load(f)
            print(f"✓ Successfully loaded preset configuration with {len(preset_data)} entries")
        else:
            print("⚠ Preset configuration file not found, but basic config works")
        
        return True
    
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_script_files():
    """Test that all script files exist"""
    print("\nTesting script files...")
    
    script_dir = Path("/Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/kvtuner")
    
    required_files = [
        "kvtuner_inference.py",
        "kvtuner_simple.py", 
        "run_kvtuner_inference.py",
        "run_inference.sh"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = script_dir / filename
        if filepath.exists():
            print(f"✓ {filename} found")
        else:
            print(f"✗ {filename} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("KVTuner Integration Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Path Test", test_paths),
        ("Configuration Test", test_config_loading),
        ("Script Files Test", test_script_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KVTuner integration is ready to use.")
        print("\nNext steps:")
        print("1. Run: ./run_inference.sh simple-test")
        print("2. Or: python kvtuner_simple.py --test --model_path /path/to/model")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies")
        print("2. Update paths in the scripts")
        print("3. Ensure KVTuner is properly installed")

if __name__ == "__main__":
    main()
