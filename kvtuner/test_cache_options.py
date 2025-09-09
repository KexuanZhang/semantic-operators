#!/usr/bin/env python3
"""
Test script to verify cache options work correctly
"""

import os
import sys

# Add KVTuner paths
possible_paths = [
    "/home/data/so2/KVTuner",
    "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner",
    "../KVTuner"
]

kvtuner_path = None
for path in possible_paths:
    if os.path.exists(path):
        kvtuner_path = path
        sys.path.insert(0, path)
        break

print("=" * 60)
print("CACHE COMPATIBILITY TEST")
print("=" * 60)

# Test 1: Transformers cache
print("\n1. Testing Transformers Cache:")
try:
    from transformers import QuantizedCacheConfig, HQQQuantizedCache
    cache_config = QuantizedCacheConfig(nbits=4, axis_key=0, axis_value=0, device='cpu')
    cache = HQQQuantizedCache(cache_config=cache_config)
    print("✅ Transformers HQQQuantizedCache works")
except Exception as e:
    print(f"❌ Transformers cache failed: {e}")

# Test 2: KVTuner cache
print("\n2. Testing KVTuner Cache:")
if kvtuner_path:
    print(f"KVTuner found at: {kvtuner_path}")
    try:
        from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
        
        # Test basic config
        cache_config = FlexibleQuantizedCacheConfig(
            nbits_key=4, 
            nbits_value=4, 
            asym=True, 
            axis_key=0, 
            axis_value=0, 
            device='cpu',
            q_group_size=-1
        )
        cache = FlexibleVanillaQuantizedCache(cache_config=cache_config)
        print("✅ KVTuner FlexibleVanillaQuantizedCache works")
        
        # Test preset config
        preset_path = os.path.join(kvtuner_path, "calibration_presets", "Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml")
        if os.path.exists(preset_path):
            print(f"✅ Found Qwen preset config: {os.path.basename(preset_path)}")
        else:
            print("⚠️  Qwen preset config not found")
            
    except Exception as e:
        print(f"❌ KVTuner cache failed: {e}")
else:
    print("❌ KVTuner not found in any expected location")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if kvtuner_path:
    print("✅ Use --cache_type kvtuner for KVTuner quantization")
print("✅ Use --cache_type transformers for HuggingFace quantization")  
print("✅ Use --cache_type auto to automatically choose best available")
print("✅ Use --cache_type none for standard inference")

print("\nTest completed!")
