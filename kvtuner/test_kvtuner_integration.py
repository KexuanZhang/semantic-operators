#!/usr/bin/env python3
"""
KVTuner Integration Test Script

This script tests the KVTuner integration with vLLM to ensure all components are working correctly.
"""

import os
import sys
import traceback
from pathlib import Path

# Add paths for vLLM and KVTuner
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
vllm_path = str(project_root / "vllm")
kvtuner_path = str(project_root / "KVTuner")

print("=== KVTuner Integration Test ===")
print(f"Project root: {project_root}")
print(f"vLLM path: {vllm_path}")
print(f"KVTuner path: {kvtuner_path}")
print(f"Working directory: {os.getcwd()}")

# Add paths to Python path
sys.path.insert(0, vllm_path)
sys.path.insert(0, kvtuner_path)

def test_basic_imports():
    """Test basic module imports"""
    print("\n1. Testing basic imports...")
    
    # Test vLLM import
    try:
        from vllm import LLM, SamplingParams
        print("   ✓ vLLM core modules imported successfully")
    except ImportError as e:
        print(f"   ✗ vLLM import failed: {e}")
        return False
    except Exception as e:
        print(f"   ✗ vLLM import error: {e}")
        return False
    
    # Test KVTuner quantization config import
    try:
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        print("   ✓ KVTuner quantization config imported successfully")
    except ImportError as e:
        print(f"   ✗ KVTuner config import failed: {e}")
        return False
    except Exception as e:
        print(f"   ✗ KVTuner config import error: {e}")
        return False
    
    # Test KVTuner cache import
    try:
        from vllm.model_executor.layers.kvtuner_cache import KVTunerCacheManager
        print("   ✓ KVTuner cache manager imported successfully")
    except ImportError as e:
        print(f"   ✗ KVTuner cache manager import failed: {e}")
        return False
    except Exception as e:
        print(f"   ✗ KVTuner cache manager import error: {e}")
        return False
        
    return True

def test_quantization_registration():
    """Test if KVTuner is properly registered as a quantization method"""
    print("\n2. Testing quantization method registration...")
    
    try:
        from vllm.model_executor.layers.quantization import QuantizationMethods
        
        # Check if kvtuner is in the quantization methods
        if hasattr(QuantizationMethods, '__args__'):
            methods = QuantizationMethods.__args__
        else:
            # Fallback for different vLLM versions
            methods = []
            
        if "kvtuner" in methods:
            print("   ✓ KVTuner is registered as a quantization method")
            return True
        else:
            print(f"   ✗ KVTuner not found in quantization methods: {methods}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error checking quantization registration: {e}")
        return False

def test_config_creation():
    """Test KVTuner config creation"""
    print("\n3. Testing KVTuner config creation...")
    
    try:
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        
        # Test basic config creation
        config = KVTunerConfig(
            kvtuner_scheme="per_token",
            kvtuner_backend="vanilla"
        )
        
        print("   ✓ Basic KVTuner config created successfully")
        print(f"      - Scheme: {config.kvtuner_scheme}")
        print(f"      - Backend: {config.kvtuner_backend}")
        
        # Test config methods
        assert config.get_name() == "kvtuner"
        print("   ✓ Config name method working")
        
        supported_dtypes = config.get_supported_act_dtypes()
        print(f"   ✓ Supported dtypes: {supported_dtypes}")
        
        config_filenames = config.get_config_filenames()
        print(f"   ✓ Config filenames: {config_filenames}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Config creation failed: {e}")
        traceback.print_exc()
        return False

def test_preset_loading():
    """Test loading KVTuner preset configurations"""
    print("\n4. Testing preset configuration loading...")
    
    # Check if calibration presets directory exists
    presets_dir = Path(kvtuner_path) / "calibration_presets"
    if not presets_dir.exists():
        print(f"   ✗ Calibration presets directory not found: {presets_dir}")
        return False
    
    # List available presets
    presets = list(presets_dir.glob("*.yaml"))
    if not presets:
        print(f"   ✗ No YAML presets found in {presets_dir}")
        return False
        
    print(f"   ✓ Found {len(presets)} preset configurations")
    
    # Test loading a specific preset
    try:
        import yaml
        test_preset = presets[0]
        print(f"   Testing preset: {test_preset.name}")
        
        with open(test_preset, 'r') as f:
            preset_config = yaml.safe_load(f)
            
        print(f"   ✓ Successfully loaded preset with {len(preset_config)} layers")
        
        # Test creating config from preset
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        
        config = KVTunerConfig(
            kvtuner_config_path=str(test_preset),
            kvtuner_scheme="per_token"
        )
        
        print("   ✓ Successfully created config from preset")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Preset loading failed: {e}")
        traceback.print_exc()
        return False

def test_kvtuner_cache_manager():
    """Test KVTuner cache manager functionality"""
    print("\n5. Testing KVTuner cache manager...")
    
    try:
        from vllm.model_executor.layers.kvtuner_cache import KVTUNER_AVAILABLE
        
        if KVTUNER_AVAILABLE:
            print("   ✓ KVTuner flexible_quant available")
        else:
            print("   ⚠ KVTuner flexible_quant not available (expected in some environments)")
            print("     This is normal if KVTuner's flexible_quant is not installed")
        
        # Test cache manager import
        from vllm.model_executor.layers.kvtuner_cache import KVTunerCacheManager, create_kvtuner_cache_manager
        print("   ✓ KVTuner cache manager classes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Cache manager test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_workflow():
    """Test the complete integration workflow"""
    print("\n6. Testing integration workflow...")
    
    try:
        # Test the workflow similar to llm_inference.py
        print("   Testing model basename extraction...")
        
        def get_model_basename(model_path):
            if os.path.exists(model_path):
                return os.path.basename(os.path.normpath(model_path))
            else:
                return model_path.replace("/", "_")
        
        test_model_name = "Qwen/Qwen2.5-3B-Instruct"
        basename = get_model_basename(test_model_name)
        expected_basename = "Qwen_Qwen2.5-3B-Instruct"
        
        if basename == expected_basename:
            print(f"   ✓ Model basename extraction working: {basename}")
        else:
            print(f"   ✗ Model basename mismatch: got {basename}, expected {expected_basename}")
            return False
        
        # Test config path construction
        scheme = "pertoken"
        config_filename = f"{basename}_{scheme}_KVTuner4_0.yaml"
        config_path = os.path.join(kvtuner_path, "calibration_presets", config_filename)
        
        expected_config = "Qwen_Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml"
        if config_filename == expected_config:
            print(f"   ✓ Config filename construction working: {config_filename}")
        else:
            print(f"   ✗ Config filename mismatch: got {config_filename}, expected {expected_config}")
            return False
        
        # Check if the specific config exists
        if os.path.exists(config_path):
            print(f"   ✓ Expected config file exists: {config_filename}")
        else:
            print(f"   ⚠ Expected config file not found: {config_filename}")
            print("     This is normal - not all model configs are available")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Integration workflow test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing KVTuner integration with vLLM...")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Quantization Registration", test_quantization_registration),
        ("Config Creation", test_config_creation),
        ("Preset Loading", test_preset_loading),
        ("Cache Manager", test_kvtuner_cache_manager),
        ("Integration Workflow", test_integration_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n   ✗ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! KVTuner integration is working correctly.")
        print("\nNext steps:")
        print("1. You can now use the LLM inference script:")
        print("   python llm_inference.py --dataset your_data.csv --model your_model --cache_mode kvtuner")
        print("\n2. For basic cache mode:")
        print("   python llm_inference.py --dataset your_data.csv --model your_model --cache_mode basic")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please check the errors above.")
        
        if passed >= 4:  # Most core functionality working
            print("\nCore functionality appears to be working. You may still be able to use KVTuner.")
            print("Try using the basic cache mode if KVTuner mode fails:")
            print("   python llm_inference.py --dataset your_data.csv --model your_model --cache_mode basic")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
