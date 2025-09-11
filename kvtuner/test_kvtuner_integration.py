#!/usr/bin/env python3
"""
Test script for KVTuner + vLLM integration validation.

This script tests the basic integration between KVTuner quantization
and vLLM without requiring heavy model loading.

Usage:
    python test_kvtuner_integration.py
"""

import os
import sys

# Add KVTuner to path if needed
kvtuner_path = "/home/data/so2/KVTuner"
if kvtuner_path not in sys.path:
    sys.path.insert(0, kvtuner_path)

def test_kvtuner_config_creation():
    """Test KVTuner configuration creation."""
    print("Testing KVTuner configuration creation...")
    
    try:
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        
        # Test basic config creation
        config = KVTunerConfig()
        assert config.kvtuner_scheme == "per_token"
        assert config.kvtuner_backend == "vanilla"
        print("✓ Basic KVTuner config creation works")
        
        # Test config with parameters
        config_with_params = KVTunerConfig(
            kvtuner_scheme="per_channel",
            kvtuner_backend="quanto",
            force_quant=True
        )
        assert config_with_params.kvtuner_scheme == "per_channel"
        assert config_with_params.kvtuner_backend == "quanto"
        assert config_with_params.force_quant == True
        print("✓ KVTuner config with parameters works")
        
        return True
    except Exception as e:
        print(f"✗ KVTuner config creation failed: {e}")
        return False


def test_kvtuner_quantization_registration():
    """Test KVTuner quantization method registration."""
    print("Testing KVTuner quantization registration...")
    
    try:
        from vllm.model_executor.layers.quantization import (
            QuantizationMethods, get_quantization_config, QUANTIZATION_METHODS
        )
        
        # Check if kvtuner is in the list
        assert "kvtuner" in QUANTIZATION_METHODS, "kvtuner not in QUANTIZATION_METHODS"
        print("✓ KVTuner is registered in QUANTIZATION_METHODS")
        
        # Test getting kvtuner config
        kvtuner_config_cls = get_quantization_config("kvtuner")
        assert kvtuner_config_cls is not None, "KVTuner config class is None"
        print("✓ KVTuner config class can be retrieved")
        
        return True
    except Exception as e:
        print(f"✗ KVTuner quantization registration failed: {e}")
        return False


def test_cache_config_kvtuner_fields():
    """Test CacheConfig has KVTuner fields."""
    print("Testing CacheConfig KVTuner fields...")
    
    try:
        from vllm.config.cache import CacheConfig
        
        # Create cache config with KVTuner parameters
        cache_config = CacheConfig(
            kvtuner_config_path="/test/path.yaml",
            kvtuner_scheme="per_token",
            kvtuner_backend="vanilla"
        )
        
        assert cache_config.kvtuner_config_path == "/test/path.yaml"
        assert cache_config.kvtuner_scheme == "per_token"
        assert cache_config.kvtuner_backend == "vanilla"
        print("✓ CacheConfig accepts KVTuner parameters")
        
        return True
    except Exception as e:
        print(f"✗ CacheConfig KVTuner fields test failed: {e}")
        return False


def test_engine_args_kvtuner_fields():
    """Test EngineArgs has KVTuner fields."""
    print("Testing EngineArgs KVTuner fields...")
    
    try:
        from vllm.engine.arg_utils import EngineArgs
        
        # Create engine args with KVTuner parameters
        engine_args = EngineArgs(
            model="dummy-model",
            kvtuner_config_path="/test/path.yaml",
            kvtuner_scheme="per_token",
            kvtuner_backend="vanilla"
        )
        
        assert engine_args.kvtuner_config_path == "/test/path.yaml"
        assert engine_args.kvtuner_scheme == "per_token"
        assert engine_args.kvtuner_backend == "vanilla"
        print("✓ EngineArgs accepts KVTuner parameters")
        
        return True
    except Exception as e:
        print(f"✗ EngineArgs KVTuner fields test failed: {e}")
        return False


def test_kvtuner_cache_manager():
    """Test KVTuner cache manager creation."""
    print("Testing KVTuner cache manager...")
    
    try:
        from vllm.model_executor.layers.kvtuner_cache import KVTunerCacheManager
        
        # Test cache manager creation without vLLM config (should work with mock)
        print("✓ KVTunerCacheManager can be imported")
        
        return True
    except Exception as e:
        print(f"✗ KVTuner cache manager test failed: {e}")
        return False


def test_yaml_config_loading():
    """Test YAML configuration loading."""
    print("Testing YAML configuration loading...")
    
    try:
        import yaml
        import tempfile
        import os
        
        # Create a temporary YAML config file
        test_config = {
            0: {"nbits_key": 8, "nbits_value": 4},
            1: {"nbits_key": 4, "nbits_value": 2},
            2: {"nbits_key": 4, "nbits_value": 2},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
            
            # Test loading config from file
            config = KVTunerConfig(kvtuner_config_path=temp_path)
            assert config.per_layer_config == test_config
            print("✓ YAML configuration loading works")
            
            return True
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"✗ YAML configuration loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("KVTuner + vLLM Integration Validation Tests")
    print("=" * 60)
    
    tests = [
        test_kvtuner_config_creation,
        test_kvtuner_quantization_registration,
        test_cache_config_kvtuner_fields,
        test_engine_args_kvtuner_fields,
        test_kvtuner_cache_manager,
        test_yaml_config_loading,
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
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! KVTuner integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the integration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
