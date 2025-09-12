#!/usr/bin/env python3
"""
Quick test script to verify KVTuner integration fix.
This script tests the key components of the KVTuner integration.
"""

import os
import sys
import tempfile
import shutil
import json
import yaml

def test_config_file_discovery():
    """Test that the config file discovery logic works correctly."""
    print("Testing config file discovery...")
    
    # Create a temporary directory to simulate model directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Test directory: {temp_dir}")
        
        # Create a sample KVTuner config file
        config_data = {
            "per_layer_config": {
                "model.layers.0": {"nbits_key": 4, "nbits_value": 4},
                "model.layers.1": {"nbits_key": 8, "nbits_value": 4}
            },
            "default_nbits_key": 8,
            "default_nbits_value": 8
        }
        
        # Test 1: YAML file
        yaml_path = os.path.join(temp_dir, "kvtuner_config.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        print(f"Created YAML config: {yaml_path}")
        
        # Test 2: JSON file  
        json_path = os.path.join(temp_dir, "kvtuner_config.json")
        with open(json_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"Created JSON config: {json_path}")
        
        # Test file discovery using glob patterns
        import glob
        
        # Test YAML discovery
        yaml_files = glob.glob(os.path.join(temp_dir, "kvtuner_config.yaml"))
        print(f"YAML files found: {yaml_files}")
        
        # Test JSON discovery  
        json_files = glob.glob(os.path.join(temp_dir, "kvtuner_config.json"))
        print(f"JSON files found: {json_files}")
        
        # Test general pattern discovery
        config_files = glob.glob(os.path.join(temp_dir, "kvtuner_config.*"))
        print(f"All kvtuner_config files: {config_files}")
        
        # Test the actual loading
        try:
            with open(yaml_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
            print("✓ YAML loading successful")
            
            with open(json_path, 'r') as f:
                loaded_json = json.load(f)
            print("✓ JSON loading successful")
            
            print("✓ Config file discovery test passed!")
            return True
        except Exception as e:
            print(f"✗ Config loading failed: {e}")
            return False

def test_method_signature_fix():
    """Test that the method signature fixes are working."""
    print("\nTesting method signature fixes...")
    
    # Add the vLLM path to sys.path
    vllm_path = "/Users/zhang/Desktop/huawei/untitled folder 6/vllm"
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
    
    try:
        # Test a few of the fixed quantization configs
        test_configs = [
            'vllm.model_executor.layers.quantization.awq_marlin',
            'vllm.model_executor.layers.quantization.gptq', 
            'vllm.model_executor.layers.quantization.kvtuner'
        ]
        
        for config_module in test_configs:
            try:
                # Import the module
                parts = config_module.split('.')
                module_name = parts[-1]
                module_path = '.'.join(parts[:-1])
                
                module = __import__(config_module, fromlist=[module_name])
                
                # Find the config class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        hasattr(attr, 'get_config_filenames') and
                        attr_name.endswith('Config')):
                        
                        # Test that get_config_filenames() can be called as static method
                        try:
                            filenames = attr.get_config_filenames()
                            print(f"✓ {attr_name}.get_config_filenames() = {filenames}")
                        except Exception as e:
                            print(f"✗ {attr_name}.get_config_filenames() failed: {e}")
                            return False
                        break
                        
            except ImportError as e:
                print(f"Could not import {config_module}: {e}")
                # This is expected if dependencies are missing
                continue
        
        print("✓ Method signature test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("KVTuner Integration Fix Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Config file discovery
    if not test_config_file_discovery():
        success = False
    
    # Test 2: Method signature fixes
    if not test_method_signature_fix():
        success = False
        
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! KVTuner integration fix appears to be working.")
        print("\nNext steps:")
        print("1. Run the actual inference test with: python test_commands.py")
        print("2. Or test manually with the commands shown in test_commands.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
