#!/usr/bin/env python3
"""
KVTuner Multi-GPU Compatibility Test

This script tests different combinations of cache modes and GPU configurations
to find the optimal setup for your environment.
"""

import sys
import os
import json

# Add paths for vLLM and KVTuner
vllm_path = "/home/data/so2/vllm"
kvtuner_path = "/home/data/so2/KVTuner"
sys.path.insert(0, vllm_path)
sys.path.insert(0, kvtuner_path)

def test_gpu_configuration():
    """Test different GPU configurations with KVTuner"""
    
    print("KVTuner Multi-GPU Compatibility Test")
    print("=" * 60)
    
    # Test configurations to try
    test_configs = [
        {
            "name": "Single GPU + KVTuner",
            "model": "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct",
            "cache_mode": "kvtuner",
            "gpu_ids": "0",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.6
        },
        {
            "name": "Single GPU + Basic Cache",
            "model": "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct",
            "cache_mode": "basic",
            "gpu_ids": "0",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.6
        },
        {
            "name": "Dual GPU + Basic Cache",
            "model": "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct",
            "cache_mode": "basic",
            "gpu_ids": "0,1",
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.6
        },
        {
            "name": "Dual GPU + KVTuner (risky)",
            "model": "/home/data/so2/semantic-operators/models/Qwen2.5-3B-Instruct", 
            "cache_mode": "kvtuner",
            "gpu_ids": "0,1",
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.5
        }
    ]
    
    successful_configs = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}. Testing: {config['name']}")
        print("-" * 40)
        
        # Set GPU environment
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_ids"]
        
        try:
            from vllm import LLM
            
            llm_kwargs = {
                "model": config["model"],
                "tensor_parallel_size": config["tensor_parallel_size"],
                "trust_remote_code": True,
                "dtype": "float16",
                "gpu_memory_utilization": config["gpu_memory_utilization"],
                "enforce_eager": True,
                "disable_log_stats": True,
                "max_model_len": 1024,  # Small context for testing
            }
            
            if config["cache_mode"] == "kvtuner":
                llm_kwargs["quantization"] = "kvtuner"
            
            print(f"   Model: {config['model']}")
            print(f"   Cache mode: {config['cache_mode']}")
            print(f"   GPU IDs: {config['gpu_ids']}")
            print(f"   Tensor parallel size: {config['tensor_parallel_size']}")
            print(f"   GPU memory: {config['gpu_memory_utilization']:.1%}")
            
            print("   Initializing...")
            llm = LLM(**llm_kwargs)
            print("   ✅ SUCCESS!")
            
            # Test a simple generation
            print("   Testing generation...")
            sampling_params = {"temperature": 0.1, "max_tokens": 10}
            from vllm import SamplingParams
            outputs = llm.generate(["Hello, world!"], SamplingParams(**sampling_params))
            print(f"   Generated: {outputs[0].outputs[0].text[:50]}...")
            print("   ✅ Generation successful!")
            
            successful_configs.append(config)
            
            # Clean up
            del llm
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            print(f"   Error type: {type(e).__name__}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if successful_configs:
        print("✅ Working configurations:")
        for config in successful_configs:
            print(f"   - {config['name']}: {config['cache_mode']} cache, "
                  f"{config['tensor_parallel_size']} GPU(s)")
        
        print(f"\n🎯 RECOMMENDATION:")
        recommended = successful_configs[0]
        print(f"Use: {recommended['name']}")
        print(f"Command: python llm_inference.py \\")
        print(f"  --dataset your_data.csv \\")
        print(f"  --model {recommended['model']} \\")
        print(f"  --cache_mode {recommended['cache_mode']} \\")
        if recommended['cache_mode'] == 'kvtuner':
            print(f"  --kvtuner_scheme pertoken \\")
        if recommended['tensor_parallel_size'] > 1:
            print(f"  --gpu_ids {recommended['gpu_ids']} \\")
        print(f"  --gpu_memory_utilization {recommended['gpu_memory_utilization']} \\")
        print(f"  --max_new_tokens 200")
    else:
        print("❌ No configurations worked successfully.")
        print("\nTroubleshooting suggestions:")
        print("1. Check GPU availability: nvidia-smi")
        print("2. Verify model path exists")
        print("3. Check vLLM installation")
        print("4. Try with a smaller model")
        print("5. Reduce GPU memory utilization")

def main():
    """Main function"""
    try:
        test_gpu_configuration()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
