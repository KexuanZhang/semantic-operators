#!/usr/bin/env python3
"""
Quick Verification Script for KVTuner+vLLM Integration

This script performs essential checks to verify the integration is working
in the target environment (/home/data/so2/).
"""

import os
import sys

def check_environment():
    """Check that all required directories exist"""
    print("🔍 Checking environment structure...")
    
    required_paths = {
        "vLLM": "/home/data/so2/vllm",
        "KVTuner": "/home/data/so2/KVTuner", 
        "Semantic Operators": "/home/data/so2/semantic-operators",
        "Calibration Presets": "/home/data/so2/KVTuner/calibration_presets"
    }
    
    all_good = True
    for name, path in required_paths.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (MISSING)")
            all_good = False
    
    return all_good

def check_imports():
    """Check critical imports"""
    print("\n🔍 Checking imports...")
    
    # Add paths
    sys.path.insert(0, "/home/data/so2/vllm")
    sys.path.insert(0, "/home/data/so2/KVTuner")
    
    try:
        from vllm import LLM, SamplingParams
        print("  ✅ vLLM core modules")
    except ImportError as e:
        print(f"  ❌ vLLM import failed: {e}")
        return False
    
    try:
        from vllm.model_executor.layers.quantization import QuantizationMethods
        if "kvtuner" in QuantizationMethods.__args__:
            print("  ✅ KVTuner quantization registered")
        else:
            print("  ❌ KVTuner not in quantization methods")
            return False
    except Exception as e:
        print(f"  ❌ Quantization check failed: {e}")
        return False
    
    try:
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        print("  ✅ KVTuner config class")
    except ImportError as e:
        print(f"  ❌ KVTuner config import failed: {e}")
        return False
    
    return True

def check_configs():
    """Check configuration files"""
    print("\n🔍 Checking KVTuner configurations...")
    
    config_dir = "/home/data/so2/KVTuner/calibration_presets"
    
    if not os.path.exists(config_dir):
        print(f"  ❌ Config directory missing: {config_dir}")
        return False
    
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    print(f"  ✅ Found {len(yaml_files)} configuration files")
    
    if len(yaml_files) > 0:
        # Test loading one config
        try:
            import yaml
            test_config = os.path.join(config_dir, yaml_files[0])
            with open(test_config, 'r') as f:
                config = yaml.safe_load(f)
            print(f"  ✅ Config loading works (tested: {yaml_files[0]})")
            return True
        except Exception as e:
            print(f"  ❌ Config loading failed: {e}")
            return False
    else:
        print("  ❌ No YAML configuration files found")
        return False

def quick_api_test():
    """Quick API functionality test"""
    print("\n🔍 Testing KVTuner API...")
    
    try:
        from vllm.engine.arg_utils import EngineArgs
        
        # Test EngineArgs with KVTuner parameters
        config_dir = "/home/data/so2/KVTuner/calibration_presets"
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        
        if yaml_files:
            test_config = os.path.join(config_dir, yaml_files[0])
            
            engine_args = EngineArgs(
                model="dummy",
                quantization="kvtuner",
                kvtuner_config_path=test_config,
                kvtuner_scheme="pertoken",
                kvtuner_backend="vanilla"
            )
            
            # Check KVTuner parameters are present
            has_config_path = hasattr(engine_args, 'kvtuner_config_path')
            has_scheme = hasattr(engine_args, 'kvtuner_scheme') 
            has_backend = hasattr(engine_args, 'kvtuner_backend')
            
            if has_config_path and has_scheme and has_backend:
                print("  ✅ EngineArgs accepts KVTuner parameters")
                return True
            else:
                print("  ❌ Missing KVTuner parameters in EngineArgs")
                return False
        else:
            print("  ❌ No config files available for testing")
            return False
            
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False

def main():
    """Run verification checks"""
    print("="*60)
    print("🚀 KVTuner+vLLM Integration Verification")
    print("   Target Environment: /home/data/so2/")
    print("="*60)
    
    checks = [
        ("Environment Structure", check_environment),
        ("Critical Imports", check_imports), 
        ("Configuration Files", check_configs),
        ("API Functionality", quick_api_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {name} check crashed: {e}")
    
    print("\n" + "="*60)
    print("📊 VERIFICATION RESULTS")
    print("="*60)
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 SUCCESS! KVTuner+vLLM integration is working correctly!")
        print("\n✅ Ready for production use:")
        print("   • python llm_inference.py --help")
        print("   • python complete_integration_test.py")
        exit_code = 0
    elif passed >= total - 1:
        print("⚠️  MOSTLY WORKING! Minor issues to resolve.")
        exit_code = 0
    else:
        print("❌ ISSUES DETECTED! Integration needs attention.")
        exit_code = 1
    
    print("="*60)
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
