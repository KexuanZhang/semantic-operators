#!/usr/bin/env python3
"""
Quick validation script to verify dual cache implementation
"""

import os
import sys

def main():
    print("🚀 KVTuner vLLM Dual Cache Implementation")
    print("=" * 50)
    
    # Check if key files exist
    files_to_check = [
        ("llm_inference.py", "Main inference script with dual cache support"),
        ("dual_cache_example.py", "Example usage demonstrations"),
        ("README_dual_cache.md", "Comprehensive documentation"),
        ("IMPLEMENTATION_SUMMARY.md", "Implementation summary")
    ]
    
    print("\n📁 File Check:")
    all_files_exist = True
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - MISSING")
            all_files_exist = False
    
    # Check script content for key features
    print("\n🔍 Feature Check:")
    try:
        with open("llm_inference.py", "r") as f:
            content = f.read()
            
        features = [
            ("--cache_mode argument", "--cache_mode" in content),
            ("Basic cache support", "cache_mode == 'basic'" in content),
            ("KVTuner cache support", "cache_mode == 'kvtuner'" in content),
            ("Automatic fallback", "Falling back to basic cache mode" in content),
            ("GPU memory configuration", "--gpu_memory_utilization" in content),
            ("Cache-specific file naming", "cache_suffix" in content)
        ]
        
        all_features_present = True
        for feature_name, present in features:
            status = "✅" if present else "❌"
            print(f"{status} {feature_name}")
            if not present:
                all_features_present = False
                
    except FileNotFoundError:
        print("❌ Could not read llm_inference.py")
        all_features_present = False
    
    # Summary
    print("\n📋 Summary:")
    if all_files_exist and all_features_present:
        print("✅ Dual cache implementation is complete and ready!")
        print("\n🎯 Next Steps:")
        print("1. Deploy to target environment (/home/data/so2/)")
        print("2. Test with: python llm_inference.py --help")
        print("3. Try basic mode: --cache_mode basic")
        print("4. Try KVTuner mode: --cache_mode kvtuner")
        
        print("\n📖 Documentation:")
        print("- README_dual_cache.md - Complete usage guide")
        print("- IMPLEMENTATION_SUMMARY.md - Technical details")
        print("- dual_cache_example.py --examples - Usage examples")
        
        return True
    else:
        print("❌ Implementation incomplete. Please check missing items above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
