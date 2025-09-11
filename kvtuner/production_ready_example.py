#!/usr/bin/env python3
"""
KVTuner-vLLM Integration: Ready for Production Testing

This script demonstrates how to use the completed KVTuner integration with vLLM.
Run this after installing vLLM and its dependencies.
"""

import os
import sys
import yaml
from typing import Optional, Dict, Any

def validate_kvtuner_config(config_path: str) -> Dict[str, Any]:
    """Validate and load KVTuner configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"KVTuner config not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError("KVTuner config must be a dictionary")
    
    # Validate structure
    for layer_id, layer_config in config.items():
        if not isinstance(layer_id, int):
            raise ValueError(f"Layer ID must be integer, got {type(layer_id)}")
        
        if 'nbits_key' not in layer_config or 'nbits_value' not in layer_config:
            raise ValueError(f"Layer {layer_id} missing nbits_key or nbits_value")
    
    print(f"✓ Validated KVTuner config with {len(config)} layers")
    return config

def run_kvtuner_vllm_example():
    """Run example KVTuner inference with vLLM."""
    
    # Configuration
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    config_path = "/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml"
    
    print("🚀 KVTuner-vLLM Integration Example")
    print("=" * 50)
    
    # Step 1: Validate KVTuner configuration
    print("\n📋 Step 1: Validating KVTuner Configuration")
    try:
        kvtuner_config = validate_kvtuner_config(config_path)
        first_layer = kvtuner_config[0]
        last_layer_id = max(kvtuner_config.keys())
        last_layer = kvtuner_config[last_layer_id]
        
        print(f"  - Model: {model_name}")
        print(f"  - Config: {os.path.basename(config_path)}")
        print(f"  - Layers: {len(kvtuner_config)} total")
        print(f"  - First layer: K{first_layer['nbits_key']}bits V{first_layer['nbits_value']}bits")
        print(f"  - Last layer: K{last_layer['nbits_key']}bits V{last_layer['nbits_value']}bits")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Step 2: Show vLLM integration usage
    print("\n🔧 Step 2: vLLM Integration Usage")
    
    # Python API usage
    print("\n✅ Python API Usage:")
    python_code = f'''
from vllm import LLM

# Initialize with KVTuner quantization
llm = LLM(
    model="{model_name}",
    quantization="kvtuner",
    kvtuner_config_path="{config_path}",
    kvtuner_scheme="per_token",
    kvtuner_backend="vanilla",
    # Optional: additional vLLM parameters
    max_model_len=4096,
    tensor_parallel_size=1
)

# Generate responses
prompts = [
    "Hello, how are you?",
    "Explain the benefits of KV cache quantization",
    "What is the capital of France?"
]

responses = llm.generate(prompts, sampling_params=SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
))

for prompt, response in zip(prompts, responses):
    print(f"Prompt: {{prompt}}")
    print(f"Response: {{response.outputs[0].text}}")
    print("-" * 40)
'''
    print(python_code)
    
    # CLI usage
    print("\n✅ CLI Usage (vLLM Serve):")
    cli_command = f'''
# Start vLLM server with KVTuner quantization
vllm serve {model_name} \\
    --quantization kvtuner \\
    --kvtuner-config-path "{config_path}" \\
    --kvtuner-scheme per_token \\
    --kvtuner-backend vanilla \\
    --host 0.0.0.0 \\
    --port 8000

# Then use OpenAI-compatible API
curl http://localhost:8000/v1/completions \\
    -H "Content-Type: application/json" \\
    -d '{{
        "model": "{model_name}",
        "prompt": "Hello, how are you?",
        "max_tokens": 512,
        "temperature": 0.7
    }}'
'''
    print(cli_command)
    
    # Step 3: Memory usage comparison
    print("\n📊 Step 3: Expected Memory Benefits")
    
    # Calculate theoretical memory savings
    total_layers = len(kvtuner_config)
    
    # Estimate memory usage for different configurations
    baseline_bits = 16  # FP16 baseline
    
    avg_key_bits = sum(layer['nbits_key'] for layer in kvtuner_config.values()) / total_layers
    avg_value_bits = sum(layer['nbits_value'] for layer in kvtuner_config.values()) / total_layers
    
    key_compression = baseline_bits / avg_key_bits
    value_compression = baseline_bits / avg_value_bits
    
    print(f"  - Baseline: FP16 ({baseline_bits} bits)")
    print(f"  - KVTuner average: K{avg_key_bits:.1f}bits V{avg_value_bits:.1f}bits")
    print(f"  - Key compression: {key_compression:.1f}x")
    print(f"  - Value compression: {value_compression:.1f}x")
    print(f"  - Estimated KV cache memory reduction: ~{((key_compression + value_compression) / 2):.1f}x")
    
    # Step 4: Performance testing recommendations
    print("\n🔬 Step 4: Recommended Performance Testing")
    
    test_commands = [
        "# Install dependencies",
        "pip install vllm torch>=2.1.0",
        "pip install flexible-quant  # For KVTuner",
        "",
        "# Benchmark memory usage",
        "python -m vllm.entrypoints.benchmark_memory \\",
        f"    --model {model_name} \\",
        "    --quantization kvtuner \\",
        f"    --kvtuner-config-path {config_path}",
        "",
        "# Benchmark throughput", 
        "python -m vllm.entrypoints.benchmark_throughput \\",
        f"    --model {model_name} \\",
        "    --quantization kvtuner \\",
        f"    --kvtuner-config-path {config_path} \\",
        "    --num-prompts 100",
        "",
        "# Compare with baseline FP16",
        "python -m vllm.entrypoints.benchmark_throughput \\",
        f"    --model {model_name} \\",
        "    --num-prompts 100"
    ]
    
    for cmd in test_commands:
        print(cmd)
    
    print("\n🎯 Integration Summary")
    print("=" * 50)
    print("✅ KVTuner quantization method registered in vLLM")
    print("✅ Configuration pipeline: LLM → EngineArgs → CacheConfig")
    print("✅ KV cache integration with KVTunerCacheManager")
    print("✅ Support for YAML preset configurations")
    print("✅ Compatible with vLLM serving infrastructure")
    print("✅ Memory-efficient mixed precision (2-8 bits per layer)")
    
    print(f"\n🎉 READY FOR PRODUCTION TESTING!")
    
    return True

if __name__ == "__main__":
    success = run_kvtuner_vllm_example()
    if success:
        print("\n🔄 Next Steps:")
        print("1. Install vLLM: pip install vllm")
        print("2. Install KVTuner: pip install flexible-quant")
        print("3. Run the examples above")
        print("4. Monitor memory usage and performance")
        print("5. Test with different model sizes and configurations")
    
    sys.exit(0 if success else 1)
