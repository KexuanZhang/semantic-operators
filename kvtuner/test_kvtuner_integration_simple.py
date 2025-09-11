#!/usr/bin/env python3
"""
Simple integration test for KVTuner-vLLM integration.

This script tests the integration without requiring full vLLM dependencies.
"""

import os
import sys
import yaml
import importlib.util

def test_yaml_loading():
    """Test KVTuner YAML configuration loading."""
    print("Testing KVTuner YAML configuration loading...")
    
    try:
        yaml_path = "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml"
        
        if not os.path.exists(yaml_path):
            print(f"✗ YAML file not found: {yaml_path}")
            return False
            
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate structure
        if not isinstance(config, dict):
            print("✗ YAML config is not a dictionary")
            return False
            
        # Check if we have numeric layer keys
        numeric_keys = [k for k in config.keys() if isinstance(k, int)]
        if len(numeric_keys) == 0:
            print("✗ No numeric layer keys found")
            return False
            
        # Check first layer structure
        first_layer = config.get(0, {})
        if 'nbits_key' not in first_layer or 'nbits_value' not in first_layer:
            print("✗ Missing required KVTuner configuration fields")
            return False
            
        print(f"✓ Successfully loaded KVTuner config with {len(config)} layers")
        print(f"  - First layer: {config[0]}")
        return True
        
    except Exception as e:
        print(f"✗ Error loading YAML: {e}")
        return False

def test_kvtuner_files_exist():
    """Test that all KVTuner integration files exist."""
    print("Testing KVTuner integration files...")
    
    base_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    files_to_check = [
        "vllm/model_executor/layers/quantization/kvtuner.py",
        "vllm/model_executor/layers/kvtuner_cache.py",
        "vllm/model_executor/layers/quantization/__init__.py",
        "vllm/engine/arg_utils.py",
        "vllm/config/cache.py",
        "vllm/entrypoints/llm.py"
    ]
    
    for file_path in files_to_check:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            return False
            
    return True

def test_quantization_methods_modification():
    """Test that QuantizationMethods includes kvtuner."""
    print("Testing QuantizationMethods modification...")
    
    try:
        init_file = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm/vllm/model_executor/layers/quantization/__init__.py"
        with open(init_file, 'r') as f:
            content = f.read()
            
        if '"kvtuner",' in content:
            print("✓ kvtuner found in QuantizationMethods")
        else:
            print("✗ kvtuner not found in QuantizationMethods")
            return False
            
        if 'from .kvtuner import KVTunerConfig' in content:
            print("✓ KVTunerConfig import found")
        else:
            print("✗ KVTunerConfig import not found")
            return False
            
        if '"kvtuner": KVTunerConfig,' in content:
            print("✓ kvtuner mapping in method_to_config found")
        else:
            print("✗ kvtuner mapping not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking QuantizationMethods: {e}")
        return False

def test_engine_args_modification():
    """Test that EngineArgs includes KVTuner parameters."""
    print("Testing EngineArgs modification...")
    
    try:
        arg_utils_file = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm/vllm/engine/arg_utils.py"
        with open(arg_utils_file, 'r') as f:
            content = f.read()
            
        kvtuner_params = [
            'kvtuner_config_path: Optional[str] = None',
            'kvtuner_scheme: str = "per_token"',
            'kvtuner_backend: str = "vanilla"'
        ]
        
        for param in kvtuner_params:
            if param in content:
                print(f"✓ Found parameter: {param.split(':')[0]}")
            else:
                print(f"✗ Missing parameter: {param.split(':')[0]}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error checking EngineArgs: {e}")
        return False

def test_cache_config_modification():
    """Test that CacheConfig includes KVTuner fields."""
    print("Testing CacheConfig modification...")
    
    try:
        cache_file = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm/vllm/config/cache.py"
        with open(cache_file, 'r') as f:
            content = f.read()
            
        kvtuner_fields = [
            'kvtuner_config_path: Optional[str] = None',
            'kvtuner_scheme: str = "per_token"',
            'kvtuner_backend: str = "vanilla"'
        ]
        
        for field in kvtuner_fields:
            if field in content:
                print(f"✓ Found field: {field.split(':')[0]}")
            else:
                print(f"✗ Missing field: {field.split(':')[0]}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error checking CacheConfig: {e}")
        return False

def test_llm_entrypoint_modification():
    """Test that LLM class includes KVTuner parameters."""
    print("Testing LLM entrypoint modification...")
    
    try:
        llm_file = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm/vllm/entrypoints/llm.py"
        with open(llm_file, 'r') as f:
            content = f.read()
            
        # Check for KVTuner parameters in constructor
        kvtuner_params = [
            'kvtuner_config_path: Optional[str] = None',
            'kvtuner_scheme: str = "per_token"',
            'kvtuner_backend: str = "vanilla"'
        ]
        
        for param in kvtuner_params:
            if param in content:
                print(f"✓ Found parameter: {param.split(':')[0]}")
            else:
                print(f"✗ Missing parameter: {param.split(':')[0]}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error checking LLM entrypoint: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=== KVTuner-vLLM Integration Test ===\n")
    
    tests = [
        test_kvtuner_files_exist,
        test_yaml_loading,
        test_quantization_methods_modification,
        test_engine_args_modification,
        test_cache_config_modification,
        test_llm_entrypoint_modification
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}\n")
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\nNext steps:")
        print("1. Install vLLM with all dependencies")
        print("2. Test with actual model inference")
        print("3. Benchmark performance vs baseline")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    main()
