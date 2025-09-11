#!/usr/bin/env python3
"""
Environment Setup and Validation Script for KVTuner+vLLM Integration

This script validates that all required directories and dependencies are properly configured
for the updated working directory structure:
- /home/data/so2/vllm
- /home/data/so2/KVTuner  
- /home/data/so2/semantic-operators
"""

import os
import sys

def validate_directories():
    """Validate that all required directories exist"""
    print("Validating directory structure...")
    
    required_dirs = {
        "vLLM": "/home/data/so2/vllm",
        "KVTuner": "/home/data/so2/KVTuner", 
        "Semantic Operators": "/home/data/so2/semantic-operators"
    }
    
    all_exist = True
    for name, path in required_dirs.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def validate_integration_files():
    """Validate that KVTuner integration files exist in vLLM"""
    print("\nValidating KVTuner integration files...")
    
    vllm_base = "/home/data/so2/vllm"
    integration_files = [
        "vllm/model_executor/layers/quantization/kvtuner.py",
        "vllm/model_executor/layers/quantization/__init__.py",
        "vllm/engine/arg_utils.py", 
        "vllm/config/cache.py",
        "vllm/entrypoints/llm.py",
        "vllm/model_executor/layers/kvtuner_cache.py"
    ]
    
    all_exist = True
    for file_path in integration_files:
        full_path = os.path.join(vllm_base, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def validate_kvtuner_configs():
    """Validate that KVTuner configuration presets exist"""
    print("\nValidating KVTuner configuration presets...")
    
    config_dir = "/home/data/so2/KVTuner/calibration_presets"
    
    if not os.path.exists(config_dir):
        print(f"✗ Configuration directory not found: {config_dir}")
        return False
    
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    
    if config_files:
        print(f"✓ Found {len(config_files)} configuration files")
        
        # Show a few examples
        examples = config_files[:3]
        for example in examples:
            print(f"  - {example}")
        if len(config_files) > 3:
            print(f"  - ... and {len(config_files) - 3} more")
        return True
    else:
        print(f"✗ No YAML configuration files found in {config_dir}")
        return False

def validate_experiment_scripts():
    """Validate that experiment scripts exist and are updated"""
    print("\nValidating experiment scripts...")
    
    script_dir = "/home/data/so2/semantic-operators/kvtuner"
    required_scripts = [
        "llm_inference.py",
        "complete_integration_test.py", 
        "test_dataset.csv",
        "USAGE_GUIDE.md"
    ]
    
    all_exist = True
    for script in required_scripts:
        script_path = os.path.join(script_dir, script)
        if os.path.exists(script_path):
            print(f"✓ {script}")
            
            # Check if Python scripts have correct path configurations
            if script.endswith('.py'):
                with open(script_path, 'r') as f:
                    content = f.read()
                    if '/home/data/so2/' in content:
                        print(f"  ✓ Updated with correct paths")
                    elif '/Users/zhang/Desktop/huawei/untitled folder 6/' in content:
                        print(f"  ⚠ Still contains old paths - needs update")
                    else:
                        print(f"  ? Could not verify path configuration")
        else:
            print(f"✗ {script} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def create_env_script():
    """Create a convenient environment setup script"""
    print("\nCreating environment setup script...")
    
    script_content = '''#!/bin/bash
# KVTuner+vLLM Environment Setup Script

# Set environment variables
export VLLM_PATH="/home/data/so2/vllm"
export KVTUNER_PATH="/home/data/so2/KVTuner"
export SEMANTIC_OPERATORS_PATH="/home/data/so2/semantic-operators"

# Add to Python path
export PYTHONPATH="$VLLM_PATH:$KVTUNER_PATH:$PYTHONPATH"

# Navigate to experiment directory
cd "$SEMANTIC_OPERATORS_PATH/kvtuner"

echo "Environment configured for KVTuner+vLLM integration"
echo "Current directory: $(pwd)"
echo "Available scripts:"
ls -la *.py *.csv *.md 2>/dev/null

# Usage examples
echo ""
echo "Usage examples:"
echo "  python complete_integration_test.py"
echo "  python llm_inference.py --dataset test_dataset.csv --model microsoft/DialoGPT-small --kvtuner_scheme pertoken"
'''
    
    script_path = "/home/data/so2/semantic-operators/kvtuner/setup_env.sh"
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        print(f"✓ Created setup script: {script_path}")
        print("  Usage: source setup_env.sh")
        return True
    except Exception as e:
        print(f"✗ Failed to create setup script: {e}")
        return False

def main():
    """Run all validation checks"""
    print("="*60)
    print("KVTuner+vLLM Environment Validation")
    print("Updated Working Directory Structure")
    print("="*60)
    
    checks = [
        ("Directory Structure", validate_directories),
        ("Integration Files", validate_integration_files),
        ("KVTuner Configurations", validate_kvtuner_configs),
        ("Experiment Scripts", validate_experiment_scripts)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        if check_func():
            passed_checks += 1
        print()  # Add spacing between checks
    
    # Create environment script
    if create_env_script():
        passed_checks += 0.5  # Bonus for successful script creation
    
    # Final summary
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if passed_checks >= total_checks:
        print(f"✓ ALL CHECKS PASSED ({passed_checks}/{total_checks})")
        print("\nEnvironment is properly configured!")
        print("\nNext steps:")
        print("1. source /home/data/so2/semantic-operators/kvtuner/setup_env.sh")
        print("2. python complete_integration_test.py")
        print("3. python llm_inference.py --help")
        exit_code = 0
    else:
        print(f"✗ SOME CHECKS FAILED ({passed_checks}/{total_checks})")
        print("\nPlease fix the issues above before proceeding.")
        exit_code = 1
    
    print("="*60)
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
