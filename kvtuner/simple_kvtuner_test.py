#!/usr/bin/env python3
"""
Simple KVTuner Integration Test - Non-vLLM Components Only

This script tests KVTuner components without loading the full vLLM engine,
which can cause threading/mutex issues in some environments.
"""

import os
import sys
import yaml
from pathlib import Path

def test_file_structure():
    """Test if required files and directories exist"""
    print("=== Testing File Structure ===")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Check vLLM directory
    vllm_path = project_root / "vllm"
    print(f"vLLM directory: {vllm_path.exists()} - {vllm_path}")
    
    # Check KVTuner directory  
    kvtuner_path = project_root / "KVTuner"
    print(f"KVTuner directory: {kvtuner_path.exists()} - {kvtuner_path}")
    
    # Check calibration presets
    presets_path = kvtuner_path / "calibration_presets"
    print(f"Calibration presets: {presets_path.exists()} - {presets_path}")
    
    if presets_path.exists():
        presets = list(presets_path.glob("*.yaml"))
        print(f"  Found {len(presets)} preset files")
        if presets:
            print(f"  Example: {presets[0].name}")
    
    # Check flexible_quant
    flex_quant_path = kvtuner_path / "flexible_quant"
    print(f"Flexible quant: {flex_quant_path.exists()} - {flex_quant_path}")
    
    # Check vLLM KVTuner files
    kvtuner_py = vllm_path / "vllm" / "model_executor" / "layers" / "quantization" / "kvtuner.py"
    print(f"KVTuner config: {kvtuner_py.exists()} - {kvtuner_py}")
    
    kvtuner_cache_py = vllm_path / "vllm" / "model_executor" / "layers" / "kvtuner_cache.py"
    print(f"KVTuner cache: {kvtuner_cache_py.exists()} - {kvtuner_cache_py}")
    
    return all([vllm_path.exists(), kvtuner_path.exists(), presets_path.exists(), 
                kvtuner_py.exists(), kvtuner_cache_py.exists()])

def test_preset_loading():
    """Test loading and parsing preset configurations"""
    print("\n=== Testing Preset Loading ===")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    presets_path = project_root / "KVTuner" / "calibration_presets"
    
    if not presets_path.exists():
        print("❌ Calibration presets directory not found")
        return False
    
    # Test loading a few different presets
    test_patterns = [
        "*Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml",
        "*Llama-3.1-8B-Instruct_pertoken_KVTuner4_0.yaml", 
        "*Mistral-7B-Instruct*_pertoken_KVTuner4_0.yaml"
    ]
    
    loaded_count = 0
    
    for pattern in test_patterns:
        matching_files = list(presets_path.glob(pattern))
        if matching_files:
            preset_file = matching_files[0]
            try:
                with open(preset_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                print(f"✅ Loaded {preset_file.name}")
                print(f"   Layers: {len(config)}")
                
                # Check structure
                if isinstance(config, dict) and len(config) > 0:
                    sample_key = list(config.keys())[0]
                    sample_config = config[sample_key]
                    if isinstance(sample_config, dict):
                        print(f"   Sample layer config: {sample_config}")
                    loaded_count += 1
                else:
                    print(f"   ⚠️  Unexpected config structure")
                    
            except Exception as e:
                print(f"❌ Failed to load {preset_file.name}: {e}")
        else:
            print(f"⚠️  No files found matching {pattern}")
    
    print(f"\nSuccessfully loaded {loaded_count} preset configurations")
    return loaded_count > 0

def test_config_file_structure():
    """Test the structure of KVTuner config files"""
    print("\n=== Testing Config File Structure ===")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Read the KVTuner config Python file
    kvtuner_config_file = project_root / "vllm" / "vllm" / "model_executor" / "layers" / "quantization" / "kvtuner.py"
    
    if not kvtuner_config_file.exists():
        print("❌ KVTuner config file not found")
        return False
    
    try:
        with open(kvtuner_config_file, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ("KVTunerConfig class", "class KVTunerConfig"),
            ("get_config_filenames method", "def get_config_filenames"),
            ("get_quant_method", "def get_quant_method"),
            ("from_config method", "def from_config"),
            ("KVTunerKVCacheMethod class", "class KVTunerKVCacheMethod"),
        ]
        
        all_found = True
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ Found {check_name}")
            else:
                print(f"❌ Missing {check_name}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return False

def test_cache_file_structure():
    """Test the KVTuner cache integration file"""
    print("\n=== Testing Cache File Structure ===")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    cache_file = project_root / "vllm" / "vllm" / "model_executor" / "layers" / "kvtuner_cache.py"
    
    if not cache_file.exists():
        print("❌ KVTuner cache file not found")
        return False
    
    try:
        with open(cache_file, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ("KVTunerCacheManager class", "class KVTunerCacheManager"),
            ("KVTUNER_AVAILABLE check", "KVTUNER_AVAILABLE"),
            ("create_kvtuner_cache_manager", "def create_kvtuner_cache_manager"),
            ("Path handling", "possible_kvtuner_paths"),
        ]
        
        all_found = True
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ Found {check_name}")
            else:
                print(f"❌ Missing {check_name}")
                all_found = False
        
        # Check if current path is included
        current_kvtuner_path = str(project_root / "KVTuner")
        if current_kvtuner_path in content:
            print(f"✅ Current KVTuner path found in cache file")
        else:
            print(f"⚠️  Current KVTuner path not found: {current_kvtuner_path}")
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading cache file: {e}")
        return False

def test_llm_inference_script():
    """Test the main LLM inference script"""
    print("\n=== Testing LLM Inference Script ===")
    
    script_path = Path(__file__).parent / "llm_inference.py"
    
    if not script_path.exists():
        print("❌ llm_inference.py not found")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check if paths have been updated
        current_base_path = "/Users/zhang/Desktop/huawei/untitled folder 6"
        
        checks = [
            ("Updated vLLM path", f'vllm_path = "{current_base_path}/vllm"'),
            ("Updated KVTuner path", f'kvtuner_path = "{current_base_path}/KVTuner"'),
            ("initialize_llm_vllm function", "def initialize_llm_vllm"),
            ("KVTuner cache mode", "cache_mode == 'kvtuner'"),
            ("Basic cache mode", "cache_mode == 'basic'"),
        ]
        
        all_found = True
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ Found {check_name}")
            else:
                print(f"❌ Missing {check_name}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading inference script: {e}")
        return False

def test_model_config_workflow():
    """Test the model configuration workflow"""
    print("\n=== Testing Model Config Workflow ===")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Test model name processing
    def get_model_basename(model_path):
        if os.path.exists(model_path):
            return os.path.basename(os.path.normpath(model_path))
        else:
            return model_path.replace("/", "_")
    
    test_cases = [
        ("Qwen/Qwen2.5-3B-Instruct", "Qwen_Qwen2.5-3B-Instruct"),
        ("meta-llama/Llama-2-7b-chat-hf", "meta-llama_Llama-2-7b-chat-hf"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai_Mistral-7B-Instruct-v0.3"),
    ]
    
    all_passed = True
    
    for model_name, expected_basename in test_cases:
        actual_basename = get_model_basename(model_name)
        if actual_basename == expected_basename:
            print(f"✅ {model_name} -> {actual_basename}")
        else:
            print(f"❌ {model_name} -> {actual_basename} (expected {expected_basename})")
            all_passed = False
    
    # Test config file lookup
    print("\nTesting config file lookup:")
    presets_path = project_root / "KVTuner" / "calibration_presets"
    
    for model_name, basename in test_cases:
        config_filename = f"{basename}_pertoken_KVTuner4_0.yaml"
        config_path = presets_path / config_filename
        
        if config_path.exists():
            print(f"✅ Config exists: {config_filename}")
        else:
            print(f"⚠️  Config not found: {config_filename}")
            # This is not necessarily an error - not all model configs exist
    
    return all_passed

def create_sample_dataset():
    """Create a sample dataset for testing"""
    print("\n=== Creating Sample Dataset ===")
    
    sample_data = [
        {"text_content": "I love this product! It works perfectly.", "label": "positive"},
        {"text_content": "This is terrible, completely broken.", "label": "negative"},
        {"text_content": "The weather is nice today.", "label": "neutral"},
        {"text_content": "Amazing service and great quality!", "label": "positive"},
        {"text_content": "Could be better, not satisfied.", "label": "negative"},
    ]
    
    try:
        import pandas as pd
        df = pd.DataFrame(sample_data)
        
        sample_file = Path(__file__).parent / "sample_dataset.csv"
        df.to_csv(sample_file, index=False)
        
        print(f"✅ Created sample dataset: {sample_file}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        return str(sample_file)
        
    except ImportError:
        print("⚠️  pandas not available, creating manual CSV")
        
        sample_file = Path(__file__).parent / "sample_dataset.csv"
        with open(sample_file, 'w') as f:
            f.write("text_content,label\n")
            for item in sample_data:
                f.write(f'"{item["text_content"]}",{item["label"]}\n')
        
        print(f"✅ Created sample dataset: {sample_file}")
        return str(sample_file)
        
    except Exception as e:
        print(f"❌ Failed to create sample dataset: {e}")
        return None

def main():
    """Run all tests"""
    print("🔍 KVTuner Integration Diagnostic Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Preset Loading", test_preset_loading),
        ("Config File Structure", test_config_file_structure),
        ("Cache File Structure", test_cache_file_structure),
        ("LLM Inference Script", test_llm_inference_script),
        ("Model Config Workflow", test_model_config_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Create sample dataset
    print(f"\n{'='*20} Sample Dataset {'='*20}")
    sample_dataset = create_sample_dataset()
    
    # Summary
    print("\n" + "="*60)
    print("🏁 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= 4:  # Most tests passed
        print("\n🎉 Core integration appears to be working!")
        print("\n📝 Next Steps:")
        print("1. Try running the inference script with basic cache:")
        print("   python llm_inference.py --dataset sample_dataset.csv --model tiny-model --cache_mode basic")
        print("\n2. If basic cache works, try KVTuner mode:")
        print("   python llm_inference.py --dataset sample_dataset.csv --model Qwen/Qwen2.5-3B-Instruct --cache_mode kvtuner")
        print("\n3. For troubleshooting, check vLLM installation:")
        print("   pip install -e /Users/zhang/Desktop/huawei/untitled\\ folder\\ 6/vllm")
        
        if sample_dataset:
            print(f"\n📊 Sample dataset created: {sample_dataset}")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review the issues above.")
        print("\n🔧 Common fixes:")
        print("1. Ensure all files are in the correct locations")
        print("2. Check file permissions")
        print("3. Verify vLLM integration files are properly updated")
    
    return passed >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
