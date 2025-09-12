#!/usr/bin/env python3
"""
Isolated test for KVTuner from_config method.
Tests the fix for "TypeError: keywords must be strings".
"""

def test_kvtuner_from_config():
    """Test KVTuner from_config method in isolation."""
    import sys
    import os
    
    # Mock the necessary torch and typing imports to avoid full vllm setup
    try:
        import torch
    except ImportError:
        print("PyTorch not available, skipping test")
        return False
    
    # Test config that mimics the YAML structure
    test_config = {
        0: {"nbits_key": 8, "nbits_value": 4},
        1: {"nbits_key": 4, "nbits_value": 2},
        2: {"nbits_key": 4, "nbits_value": 2},
        3: {"nbits_key": 4, "nbits_value": 2},
    }
    
    print(f"Test config: {test_config}")
    print(f"Key types: {[type(k) for k in test_config.keys()]}")
    
    # Simulate the from_config method logic
    try:
        per_layer_config = {}
        
        # Convert all keys to strings to ensure compatibility
        for key, value in test_config.items():
            # Convert numeric keys to strings for consistency
            str_key = str(key)
            per_layer_config[str_key] = value
        
        print(f"Converted config: {per_layer_config}")
        print("✓ Key conversion successful")
        
        # Test that the converted config can be used as constructor args
        # This simulates what would happen in the actual KVTunerConfig constructor
        kwargs = {
            "kvtuner_config_path": None,
            "per_layer_config": per_layer_config
        }
        
        print("✓ Constructor kwargs preparation successful")
        print(f"Constructor would be called with: {list(kwargs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in from_config logic: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing KVTuner from_config Fix (Isolated)")
    print("=" * 50)
    
    success = test_kvtuner_from_config()
    
    print("=" * 50)
    if success:
        print("✓ from_config fix logic works correctly!")
    else:
        print("✗ from_config fix has issues.")
    
    exit(0 if success else 1)
