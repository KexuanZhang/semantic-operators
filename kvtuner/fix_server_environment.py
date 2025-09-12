#!/usr/bin/env python3
"""
Fix script for get_quant_method signature issue in user's server environment.
This script applies the necessary fixes to resolve the TypeError.
"""

import os
import sys

def check_and_fix_kvtuner_config(vllm_path):
    """Check and fix the KVTuner config file"""
    kvtuner_file = os.path.join(vllm_path, "vllm", "model_executor", "layers", "quantization", "kvtuner.py")
    
    if not os.path.exists(kvtuner_file):
        print(f"❌ KVTuner file not found: {kvtuner_file}")
        return False
    
    print(f"📝 Checking: {kvtuner_file}")
    
    with open(kvtuner_file, 'r') as f:
        content = f.read()
    
    # Check if the fix is already applied
    if "def get_quant_method(self, layer: torch.nn.Module," in content and "prefix: str" in content:
        print("✅ get_quant_method signature is already correct")
        
        if "return UnquantizedLinearMethod()" in content:
            print("✅ UnquantizedLinearMethod return is already present")
            return True
        else:
            print("⚠️  UnquantizedLinearMethod return may be missing")
    
    print("🔧 Applying fix...")
    
    # Apply the fix
    fixed_content = content
    
    # Fix 1: Update import section to include UnquantizedLinearMethod
    if "from vllm.model_executor.layers.linear import UnquantizedLinearMethod" not in content:
        # Find the import section and add the import
        if "from vllm.model_executor.layers.quantization.base_config import" in content:
            fixed_content = fixed_content.replace(
                "from vllm.model_executor.layers.quantization.base_config import (\n    QuantizationConfig, QuantizeMethodBase)",
                "from vllm.model_executor.layers.quantization.base_config import (\n    QuantizationConfig, QuantizeMethodBase)\nfrom vllm.model_executor.layers.linear import UnquantizedLinearMethod"
            )
        print("✅ Added UnquantizedLinearMethod import")
    
    # Fix 2: Update get_quant_method signature and implementation
    old_method = """    def get_quant_method(self) -> Optional["QuantizeMethodBase"]:
        \"\"\"Get quantization method for KVTuner.
        
        Since KVTuner primarily focuses on KV cache quantization rather than
        weight quantization, this returns None for most cases.
        \"\"\"
        return None"""
    
    new_method = """    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        \"\"\"Get quantization method for KVTuner.
        
        KVTuner primarily focuses on KV cache quantization rather than weight quantization.
        For linear layers, we return UnquantizedLinearMethod to satisfy vLLM's requirements.
        \"\"\"
        # Import here to avoid circular imports
        from vllm.model_executor.layers.linear import LinearBase, ParallelLMHead
        
        # For linear layers, return UnquantizedLinearMethod. This is necessary
        # to satisfy vLLM's requirement that get_quant_method() returns a valid method.
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return UnquantizedLinearMethod()
        
        # For other layer types, return None (KV cache quantization is handled 
        # through get_kv_cache_method())
        return None"""
    
    if old_method in content:
        fixed_content = fixed_content.replace(old_method, new_method)
        print("✅ Updated get_quant_method signature and implementation")
    else:
        print("⚠️  Could not find exact old method to replace")
        print("    You may need to manually update the get_quant_method")
    
    # Write the fixed content back
    try:
        with open(kvtuner_file, 'w') as f:
            f.write(fixed_content)
        print("✅ File updated successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to write file: {e}")
        return False

def main():
    print("🔧 KVTuner get_quant_method Fix Script")
    print("=" * 50)
    
    # Determine the correct path based on environment
    possible_paths = [
        "/home/data/so2/vllm",  # User's server environment
        "/Users/zhang/Desktop/huawei/untitled folder 6/vllm",  # Current development environment
    ]
    
    vllm_path = None
    for path in possible_paths:
        if os.path.exists(path):
            vllm_path = path
            break
    
    if not vllm_path:
        print("❌ Could not find vLLM directory")
        print("Please specify the correct path to your vLLM installation")
        return False
    
    print(f"📂 Using vLLM path: {vllm_path}")
    
    # Apply the fix
    success = check_and_fix_kvtuner_config(vllm_path)
    
    if success:
        print("\n🎉 Fix applied successfully!")
        print("\n📝 Next steps:")
        print("1. Try running your test again:")
        print("   python test_kvtuner_final.py")
        print("\n2. The get_quant_method() error should now be resolved")
        print("\n3. If you encounter other issues, they may be related to:")
        print("   - Missing dependencies (flexible_quant)")
        print("   - GPU/CUDA configuration")
        print("   - Model loading issues")
    else:
        print("\n❌ Fix could not be applied automatically")
        print("\n🛠️  Manual fix required:")
        print("1. Open your vLLM kvtuner.py file")
        print("2. Update get_quant_method to include 'layer' and 'prefix' parameters")
        print("3. Return UnquantizedLinearMethod() for linear layers")
    
    return success

if __name__ == "__main__":
    main()
