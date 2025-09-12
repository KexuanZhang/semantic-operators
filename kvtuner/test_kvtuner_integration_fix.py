#!/usr/bin/env python3
"""
Test KVTuner integration fix end-to-end.
This script tests the complete flow that was causing the "TypeError: keywords must be strings" error.
"""

import sys
import os
import tempfile
import json
import yaml

def setup_test_environment():
    """Set up the test environment with proper paths."""
    # Add vllm to Python path
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)

def test_yaml_config_structure():
    """Test that we can load the YAML config and understand its structure."""
    print("Testing YAML config structure...")
    
    config_path = "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ YAML loaded successfully")
    print(f"  Keys: {list(config.keys())[:5]}...")
    print(f"  Key types: {[type(k).__name__ for k in list(config.keys())[:3]]}")
    print(f"  Sample value: {config[0]}")
    
    return config

def test_kvtuner_from_config_direct(config):
    """Test KVTuner.from_config method directly."""
    print("\nTesting KVTuner.from_config method...")
    
    try:
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        
        # This was the line that failed before our fix
        kvtuner_config = KVTunerConfig.from_config(config)
        
        print(f"✓ KVTunerConfig.from_config() succeeded")
        print(f"  Config type: {type(kvtuner_config)}")
        print(f"  Per-layer config keys: {list(kvtuner_config.per_layer_config.keys())[:5]}...")
        print(f"  Per-layer config key types: {[type(k).__name__ for k in list(kvtuner_config.per_layer_config.keys())[:3]]}")
        
        return kvtuner_config
        
    except Exception as e:
        print(f"✗ KVTunerConfig.from_config() failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_weight_utils_integration(config):
    """Test the weight_utils.py integration that calls from_config."""
    print("\nTesting weight_utils integration...")
    
    try:
        # Create a temporary JSON file to simulate the weight_utils.py flow
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            json_path = f.name
        
        print(f"✓ Created temporary JSON config: {json_path}")
        
        # Simulate what weight_utils.py does
        with open(json_path, 'r') as f:
            loaded_config = json.load(f)
        
        print(f"✓ JSON config loaded")
        print(f"  Keys: {list(loaded_config.keys())[:3]}")
        print(f"  Key types: {[type(k).__name__ for k in list(loaded_config.keys())[:3]]}")
        
        # The issue was that JSON loads string keys, but YAML can load numeric keys
        # Our fix handles both cases
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        kvtuner_config = KVTunerConfig.from_config(loaded_config)
        
        print(f"✓ Integration test passed")
        
        # Clean up
        os.unlink(json_path)
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("KVTuner Integration Fix Test")
    print("=" * 50)
    
    # Setup
    setup_test_environment()
    
    try:
        # Test 1: YAML config structure
        config = test_yaml_config_structure()
        if not config:
            print("✗ YAML config test failed")
            return False
        
        # Test 2: KVTuner from_config method
        kvtuner_config = test_kvtuner_from_config_direct(config)
        if not kvtuner_config:
            print("✗ KVTuner from_config test failed")
            return False
        
        # Test 3: Weight utils integration
        if not test_weight_utils_integration(config):
            print("✗ Weight utils integration test failed")
            return False
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! The KVTuner integration fix is working.")
        return True
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
