#!/usr/bin/env python3
"""
Minimal test for KVTuner YAML config loading issue.
This tests specifically the "TypeError: keywords must be strings" fix.
"""

import sys
import os

# Add the local vllm to path
sys.path.insert(0, "/Users/zhang/Desktop/huawei/untitled folder 6/vllm")

def test_yaml_config_loading():
    """Test loading a KVTuner YAML config without the keyword arguments error."""
    try:
        import yaml
        
        # Load the YAML config directly to see its structure
        yaml_path = "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml"
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ YAML file loaded successfully")
        print(f"Config keys: {list(config.keys())[:5]}...")  # Show first 5 keys
        print(f"Key types: {[type(k) for k in list(config.keys())[:5]]}")
        
        # Now test the KVTuner config creation
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        
        # This should not fail with "TypeError: keywords must be strings"
        kvtuner_config = KVTunerConfig.from_config(config)
        
        print("✓ KVTunerConfig.from_config() succeeded")
        print(f"Per-layer config keys: {list(kvtuner_config.per_layer_config.keys())[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing KVTuner YAML Config Loading")
    print("=" * 50)
    
    success = test_yaml_config_loading()
    
    print("=" * 50)
    if success:
        print("✓ All tests passed! YAML config loading fix works.")
    else:
        print("✗ Tests failed. Please check the implementation.")
    
    sys.exit(0 if success else 1)
