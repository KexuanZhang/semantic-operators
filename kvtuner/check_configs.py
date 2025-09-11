#!/usr/bin/env python3
"""
Simple test to check what configuration files are available
"""

import os
import sys

# Add paths
kvtuner_path = "/home/data/so2/KVTuner"
config_dir = os.path.join(kvtuner_path, "calibration_presets")

print("Checking KVTuner configuration files...")
print(f"Config directory: {config_dir}")

if os.path.exists(config_dir):
    all_files = os.listdir(config_dir)
    yaml_files = [f for f in all_files if f.endswith('.yaml')]
    
    print(f"Total files: {len(all_files)}")
    print(f"YAML files: {len(yaml_files)}")
    
    if yaml_files:
        print("\nFirst 10 YAML config files:")
        for i, config_file in enumerate(yaml_files[:10]):
            print(f"  {i+1}. {config_file}")
            
        # Try loading the first one
        print(f"\nTesting loading first config: {yaml_files[0]}")
        try:
            import yaml
            config_path = os.path.join(config_dir, yaml_files[0])
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Successfully loaded config")
            print(f"  Type: {type(config)}")
            if isinstance(config, dict):
                print(f"  Keys: {len(config)} entries")
                sample_keys = list(config.keys())[:5]
                print(f"  Sample keys: {sample_keys}")
            else:
                print(f"  Content: {config}")
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
    else:
        print("No YAML files found!")
else:
    print(f"Config directory does not exist: {config_dir}")
