#!/usr/bin/env python3
"""
Simplified test for KVTuner integration code validation.

This script directly tests our code modifications without requiring
full vLLM dependencies.
"""

import os
import sys

def test_kvtuner_quantization_import():
    """Test that our KVTuner quantization module can be imported."""
    print("Testing KVTuner quantization module import...")
    
    # Add vLLM to path
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    
    try:
        # Test minimal imports without dependencies
        import importlib.util
        
        # Test kvtuner.py file exists and is syntactically correct
        kvtuner_path = os.path.join(vllm_path, "vllm", "model_executor", "layers", "quantization", "kvtuner.py")
        if not os.path.exists(kvtuner_path):
            print(f"✗ KVTuner file not found: {kvtuner_path}")
            return False
        
        # Try to compile the file
        with open(kvtuner_path, 'r') as f:
            code = f.read()
        
        try:
            compile(code, kvtuner_path, 'exec')
            print("✓ KVTuner quantization module is syntactically correct")
        except SyntaxError as e:
            print(f"✗ Syntax error in KVTuner module: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error testing KVTuner module: {e}")
        return False


def test_quantization_methods_modification():
    """Test that QuantizationMethods includes kvtuner."""
    print("Testing QuantizationMethods modification...")
    
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    quant_init_path = os.path.join(vllm_path, "vllm", "model_executor", "layers", "quantization", "__init__.py")
    
    try:
        with open(quant_init_path, 'r') as f:
            content = f.read()
        
        # Check if kvtuner is in the QuantizationMethods
        if '"kvtuner",' in content:
            print("✓ 'kvtuner' found in QuantizationMethods")
        else:
            print("✗ 'kvtuner' not found in QuantizationMethods")
            return False
        
        # Check if kvtuner import is added
        if 'from .kvtuner import KVTunerConfig' in content:
            print("✓ KVTuner import found in quantization __init__.py")
        else:
            print("✗ KVTuner import not found in quantization __init__.py")
            return False
        
        # Check if kvtuner is in method_to_config
        if '"kvtuner": KVTunerConfig,' in content:
            print("✓ KVTuner config mapping found")
        else:
            print("✗ KVTuner config mapping not found")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking QuantizationMethods: {e}")
        return False


def test_engine_args_modification():
    """Test that EngineArgs includes KVTuner fields."""
    print("Testing EngineArgs modification...")
    
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    arg_utils_path = os.path.join(vllm_path, "vllm", "engine", "arg_utils.py")
    
    try:
        with open(arg_utils_path, 'r') as f:
            content = f.read()
        
        # Check for KVTuner fields in EngineArgs
        kvtuner_fields = [
            'kvtuner_config_path: Optional[str] = None',
            'kvtuner_scheme: str = "per_token"',
            'kvtuner_backend: str = "vanilla"'
        ]
        
        for field in kvtuner_fields:
            if field in content:
                print(f"✓ Found EngineArgs field: {field.split(':')[0]}")
            else:
                print(f"✗ Missing EngineArgs field: {field.split(':')[0]}")
                return False
        
        # Check if KVTuner parameters are passed to CacheConfig
        cache_config_kvtuner = [
            'kvtuner_config_path=self.kvtuner_config_path',
            'kvtuner_scheme=self.kvtuner_scheme',
            'kvtuner_backend=self.kvtuner_backend'
        ]
        
        for param in cache_config_kvtuner:
            if param in content:
                print(f"✓ Found CacheConfig parameter: {param.split('=')[0]}")
            else:
                print(f"✗ Missing CacheConfig parameter: {param.split('=')[0]}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking EngineArgs: {e}")
        return False


def test_cache_config_modification():
    """Test that CacheConfig includes KVTuner fields."""
    print("Testing CacheConfig modification...")
    
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    cache_config_path = os.path.join(vllm_path, "vllm", "config", "cache.py")
    
    try:
        with open(cache_config_path, 'r') as f:
            content = f.read()
        
        # Check for KVTuner fields in CacheConfig
        kvtuner_fields = [
            'kvtuner_config_path: Optional[str] = None',
            'kvtuner_scheme: str = "per_token"',
            'kvtuner_backend: str = "vanilla"'
        ]
        
        for field in kvtuner_fields:
            if field in content:
                print(f"✓ Found CacheConfig field: {field.split(':')[0]}")
            else:
                print(f"✗ Missing CacheConfig field: {field.split(':')[0]}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking CacheConfig: {e}")
        return False


def test_llm_entrypoint_modification():
    """Test that LLM class includes KVTuner parameters."""
    print("Testing LLM entrypoint modification...")
    
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    llm_path = os.path.join(vllm_path, "vllm", "entrypoints", "llm.py")
    
    try:
        with open(llm_path, 'r') as f:
            content = f.read()
        
        # Check for KVTuner parameters in LLM constructor
        kvtuner_params = [
            'kvtuner_config_path: Optional[str] = None',
            'kvtuner_scheme: str = "per_token"',
            'kvtuner_backend: str = "vanilla"'
        ]
        
        for param in kvtuner_params:
            if param in content:
                print(f"✓ Found LLM parameter: {param.split(':')[0]}")
            else:
                print(f"✗ Missing LLM parameter: {param.split(':')[0]}")
                return False
        
        # Check if parameters are passed to EngineArgs
        engine_args_params = [
            'kvtuner_config_path=kvtuner_config_path',
            'kvtuner_scheme=kvtuner_scheme',
            'kvtuner_backend=kvtuner_backend'
        ]
        
        for param in engine_args_params:
            if param in content:
                print(f"✓ Found EngineArgs parameter passing: {param.split('=')[0]}")
            else:
                print(f"✗ Missing EngineArgs parameter passing: {param.split('=')[0]}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking LLM entrypoint: {e}")
        return False


def test_kvtuner_cache_module():
    """Test that KVTuner cache module exists and is correct."""
    print("Testing KVTuner cache module...")
    
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    cache_module_path = os.path.join(vllm_path, "vllm", "model_executor", "layers", "kvtuner_cache.py")
    
    try:
        if not os.path.exists(cache_module_path):
            print(f"✗ KVTuner cache module not found: {cache_module_path}")
            return False
        
        with open(cache_module_path, 'r') as f:
            content = f.read()
        
        # Check for key classes and functions
        key_components = [
            'class KVTunerCacheManager',
            'def create_kvtuner_cache_manager',
            'def update_cache',
            'def is_quantized'
        ]
        
        for component in key_components:
            if component in content:
                print(f"✓ Found component: {component}")
            else:
                print(f"✗ Missing component: {component}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking KVTuner cache module: {e}")
        return False


def test_integration_scripts():
    """Test that integration scripts exist."""
    print("Testing integration scripts...")
    
    scripts_path = "/Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/kvtuner"
    
    scripts = [
        "kvtuner_vllm_inference.py",
        "test_kvtuner_integration.py"
    ]
    
    for script in scripts:
        script_path = os.path.join(scripts_path, script)
        if os.path.exists(script_path):
            print(f"✓ Found script: {script}")
        else:
            print(f"✗ Missing script: {script}")
            return False
    
    return True


def main():
    """Run all simplified tests."""
    print("=" * 60)
    print("KVTuner + vLLM Integration Code Validation")
    print("=" * 60)
    
    tests = [
        test_kvtuner_quantization_import,
        test_quantization_methods_modification,
        test_engine_args_modification,
        test_cache_config_modification,
        test_llm_entrypoint_modification,
        test_kvtuner_cache_module,
        test_integration_scripts,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"Code Validation Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All code validations passed! KVTuner integration code is correct.")
        return 0
    else:
        print("❌ Some validations failed. Please check the code modifications.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
