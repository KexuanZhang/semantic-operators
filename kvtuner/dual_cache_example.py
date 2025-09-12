#!/usr/bin/env python3
"""
Dual Cache Example Script

This script demonstrates how to use the updated llm_inference.py script
with both KVTuner quantized cache and basic vLLM cache modes.
"""

import os
import subprocess
import sys

def run_inference_example():
    """Run inference examples with both cache modes"""
    
    # Path to the inference script
    script_path = os.path.join(os.path.dirname(__file__), "llm_inference.py")
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Test dataset not found at {dataset_path}")
        print("Please create a test dataset or provide a valid dataset path")
        return
    
    # Example model (adjust based on your setup)
    model_name = "microsoft/DialoGPT-small"  # Small model for testing
    
    print("=" * 60)
    print("Dual Cache Mode Inference Examples")
    print("=" * 60)
    
    # Example 1: Basic vLLM cache mode
    print("\n1. Running inference with BASIC cache mode...")
    print("-" * 40)
    
    basic_cmd = [
        sys.executable, script_path,
        "--dataset", dataset_path,
        "--model", model_name,
        "--cache_mode", "basic",
        "--max_rows", "5",  # Limit for testing
        "--gpu_memory_utilization", "0.6",
        "--output_prefix", "basic_example"
    ]
    
    print("Command:", " ".join(basic_cmd))
    print("This would run basic vLLM cache inference...")
    
    # Example 2: KVTuner quantized cache mode
    print("\n2. Running inference with KVTUNER cache mode...")
    print("-" * 40)
    
    kvtuner_cmd = [
        sys.executable, script_path,
        "--dataset", dataset_path,
        "--model", "meta-llama/Llama-2-7b-chat-hf",  # Model with KVTuner support
        "--cache_mode", "kvtuner",
        "--kvtuner_scheme", "pertoken",
        "--max_rows", "5",  # Limit for testing
        "--gpu_memory_utilization", "0.6",
        "--output_prefix", "kvtuner_example"
    ]
    
    print("Command:", " ".join(kvtuner_cmd))
    print("This would run KVTuner quantized cache inference...")
    
    print("\n" + "=" * 60)
    print("Example commands shown above. To actually run:")
    print("1. Ensure you're in the target environment with vLLM installed")
    print("2. Copy these commands to your terminal")
    print("3. Adjust model names and paths as needed")
    print("=" * 60)

def print_usage_examples():
    """Print various usage examples"""
    print("Usage Examples:")
    print("=" * 50)
    
    examples = [
        {
            "title": "Basic vLLM Cache",
            "command": "python llm_inference.py --dataset data.csv --model microsoft/DialoGPT-small --cache_mode basic"
        },
        {
            "title": "KVTuner Quantized Cache (PerToken)",
            "command": "python llm_inference.py --dataset data.csv --model meta-llama/Llama-2-7b-chat-hf --cache_mode kvtuner --kvtuner_scheme pertoken"
        },
        {
            "title": "KVTuner Quantized Cache (KIVI)",
            "command": "python llm_inference.py --dataset data.csv --model meta-llama/Llama-2-7b-chat-hf --cache_mode kvtuner --kvtuner_scheme kivi"
        },
        {
            "title": "Custom GPU Memory (60%)",
            "command": "python llm_inference.py --dataset data.csv --model microsoft/DialoGPT-small --cache_mode basic --gpu_memory_utilization 0.6"
        },
        {
            "title": "Multi-GPU Setup",
            "command": "python llm_inference.py --dataset data.csv --model meta-llama/Llama-2-7b-chat-hf --cache_mode kvtuner --gpu_ids '0,1'"
        },
        {
            "title": "Limited Dataset Testing",
            "command": "python llm_inference.py --dataset data.csv --model microsoft/DialoGPT-small --cache_mode basic --max_rows 10"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   {example['command']}")
    
    print(f"\nCache Mode Details:")
    print("- basic: Uses vLLM's default cache implementation")
    print("- kvtuner: Uses KVTuner's mixed precision quantized cache (2-8 bits per layer)")
    print("           Provides ~4.6x memory reduction with minimal quality loss")
    print("           Requires KVTuner calibration presets for the specific model")
    
    print(f"\nAutomatic Fallback:")
    print("- If KVTuner config is not found, automatically falls back to basic cache")
    print("- Ensures inference always works regardless of KVTuner availability")

def show_help():
    """Show help information"""
    print("Dual Cache Example Script")
    print("=" * 30)
    print()
    print("This script demonstrates the dual cache functionality of llm_inference.py")
    print()
    print("Options:")
    print("  python dual_cache_example.py           - Show example commands")
    print("  python dual_cache_example.py --examples - Show detailed usage examples")
    print("  python dual_cache_example.py --help    - Show this help message")
    print()
    print("The updated llm_inference.py script supports:")
    print("- KVTuner quantized cache (--cache_mode kvtuner)")
    print("- Basic vLLM cache (--cache_mode basic)")
    print("- Configurable GPU memory utilization")
    print("- Automatic fallback for missing KVTuner configs")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--examples":
            print_usage_examples()
        elif sys.argv[1] == "--help":
            show_help()
        else:
            print("Unknown option. Use --examples or --help")
    else:
        run_inference_example()
