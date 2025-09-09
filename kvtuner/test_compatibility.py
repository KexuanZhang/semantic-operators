#!/usr/bin/env python3
"""
Test script to check Python and library compatibility for KVTuner
"""

import sys
import os

print("=" * 60)
print("KVTUNER COMPATIBILITY CHECK")
print("=" * 60)

# Check Python version
print(f"Python version: {sys.version}")
if sys.version_info >= (3, 12):
    print("⚠️  Warning: Python 3.12+ may have compatibility issues")
elif sys.version_info < (3, 8):
    print("❌ Error: Python version too old (need 3.8+)")
else:
    print("✅ Python version should be compatible")

print("\nChecking core dependencies:")

# Test core imports
deps = [
    ("torch", "PyTorch"),
    ("transformers", "Hugging Face Transformers"), 
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm")
]

for module, name in deps:
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError as e:
        print(f"❌ {name}: {e}")

# Check KVTuner paths
print("\nChecking KVTuner paths:")
possible_paths = [
    "/home/data/so2/KVTuner",
    "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner",
    "../KVTuner", 
    "../../KVTuner"
]

kvtuner_found = False
for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Found KVTuner at: {path}")
        kvtuner_found = True
        
        # Check for key files
        flexible_quant = os.path.join(path, "flexible_quant")
        presets = os.path.join(path, "calibration_presets")
        qwen_config = os.path.join(presets, "Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml")
        
        print(f"   - flexible_quant directory: {'✅' if os.path.exists(flexible_quant) else '❌'}")
        print(f"   - calibration_presets directory: {'✅' if os.path.exists(presets) else '❌'}")
        print(f"   - Qwen config file: {'✅' if os.path.exists(qwen_config) else '❌'}")
        break

if not kvtuner_found:
    print("❌ KVTuner not found in any expected location")

# Test KVTuner imports if found
if kvtuner_found:
    print("\nTesting KVTuner imports:")
    sys.path.insert(0, path)
    
    try:
        from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig
        print("✅ FlexibleQuantizedCacheConfig")
    except ImportError as e:
        print(f"❌ FlexibleQuantizedCacheConfig: {e}")
    
    try:
        from flexible_quant.flexible_quantized_cache import FlexibleVanillaQuantizedCache
        print("✅ FlexibleVanillaQuantizedCache")
    except ImportError as e:
        print(f"❌ FlexibleVanillaQuantizedCache: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if sys.version_info >= (3, 12):
    print("1. Consider using Python 3.9-3.11 for better compatibility")
    
if not kvtuner_found:
    print("2. Clone KVTuner repository to one of the expected locations")
    print("3. Or update the kvtuner_path in the inference script")

print("4. If imports fail, try: pip install transformers torch accelerate")
print("5. For KVTuner: cd /path/to/KVTuner/flexible_quant && pip install -e .")

print("\nTest completed!")
