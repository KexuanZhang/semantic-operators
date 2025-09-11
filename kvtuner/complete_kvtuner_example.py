#!/usr/bin/env python3
"""
Complete KVTuner + vLLM Integration Example

This script demonstrates a complete workflow for using KVTuner quantization
with vLLM, including model loading, configuration, inference, and benchmarking.

Usage:
    python complete_kvtuner_example.py [options]

Features:
    - Model loading with KVTuner quantization
    - Memory usage monitoring
    - Performance benchmarking
    - Quality evaluation
    - Multiple backend support

Author: GitHub Copilot Assistant
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

# Add KVTuner to path
KVTUNER_PATH = "/home/data/so2/KVTuner"
if KVTUNER_PATH not in sys.path:
    sys.path.insert(0, KVTUNER_PATH)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete KVTuner + vLLM Integration Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and serving arguments
    parser.add_argument(
        "--model", 
        type=str, 
        default="/home/data/so2/models/Qwen2.5-3B-Instruct",
        help="Path to the model"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="GPU memory utilization"
    )
    
    # KVTuner configuration
    parser.add_argument(
        "--kvtuner-config", 
        type=str, 
        default="/home/data/so2/KVTuner/calibration_presets/Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml",
        help="KVTuner configuration file"
    )
    parser.add_argument(
        "--kvtuner-scheme", 
        type=str, 
        default="per_token",
        choices=["per_token", "per_channel"],
        help="KVTuner quantization scheme"
    )
    parser.add_argument(
        "--kvtuner-backend", 
        type=str, 
        default="vanilla",
        choices=["vanilla", "quanto", "hqq"],
        help="KVTuner backend"
    )
    
    # Comparison modes
    parser.add_argument(
        "--compare-baseline", 
        action="store_true",
        help="Compare with baseline (no quantization)"
    )
    parser.add_argument(
        "--compare-fp8", 
        action="store_true",
        help="Compare with FP8 quantization"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--num-prompts", 
        type=int, 
        default=5,
        help="Number of prompts for evaluation"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    
    # Output and logging
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./kvtuner_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


class MemoryMonitor:
    """Monitor GPU memory usage."""
    
    def __init__(self):
        self.measurements = []
    
    def record(self, label: str) -> Dict[str, float]:
        """Record current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            
            measurement = {
                "label": label,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "timestamp": time.time()
            }
            
            self.measurements.append(measurement)
            return measurement
        else:
            return {"label": label, "allocated_gb": 0, "reserved_gb": 0}
    
    def get_peak_memory(self) -> float:
        """Get peak allocated memory."""
        if not self.measurements:
            return 0.0
        return max(m["allocated_gb"] for m in self.measurements)
    
    def print_summary(self):
        """Print memory usage summary."""
        print("\n" + "="*50)
        print("MEMORY USAGE SUMMARY")
        print("="*50)
        
        for measurement in self.measurements:
            print(f"{measurement['label']:.<30} {measurement['allocated_gb']:.2f}GB")
        
        peak = self.get_peak_memory()
        print(f"{'Peak Memory':.<30} {peak:.2f}GB")


class PerformanceBenchmark:
    """Benchmark performance metrics."""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(
        self, 
        llm, 
        prompts: List[str], 
        sampling_params,
        label: str = "benchmark"
    ) -> Dict[str, Any]:
        """Run performance benchmark."""
        print(f"\nRunning {label} benchmark...")
        
        # Warmup
        llm.generate(prompts[:1], sampling_params)
        
        # Actual benchmark
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        total_input_tokens = sum(len(prompt.split()) * 1.3 for prompt in prompts)  # Approximate
        total_output_tokens = sum(len(output.outputs[0].text.split()) * 1.3 for output in outputs)
        total_tokens = total_input_tokens + total_output_tokens
        
        throughput = total_tokens / total_time if total_time > 0 else 0
        latency = total_time / len(prompts)
        
        result = {
            "label": label,
            "total_time": total_time,
            "throughput_tokens_per_sec": throughput,
            "latency_per_request": latency,
            "num_requests": len(prompts),
            "total_tokens": total_tokens,
            "outputs": [output.outputs[0].text for output in outputs]
        }
        
        self.results.append(result)
        return result
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        for result in self.results:
            print(f"\n{result['label'].upper()}:")
            print(f"  Total Time: {result['total_time']:.2f}s")
            print(f"  Throughput: {result['throughput_tokens_per_sec']:.2f} tokens/sec")
            print(f"  Latency: {result['latency_per_request']:.2f}s per request")
            print(f"  Total Tokens: {result['total_tokens']:.0f}")


def get_test_prompts(num_prompts: int = 5) -> List[str]:
    """Get test prompts for evaluation."""
    base_prompts = [
        "Explain the concept of artificial intelligence and its applications.",
        "Write a short story about a robot learning to understand human emotions.",
        "What are the main benefits and challenges of renewable energy?",
        "Describe the process of photosynthesis in plants and its importance.",
        "How does machine learning differ from traditional programming approaches?",
        "Explain the role of quantum computing in future technology.",
        "What are the ethical considerations in developing AI systems?",
        "Describe the impact of climate change on global ecosystems.",
        "How do neural networks work and what are their applications?",
        "Explain the concept of blockchain technology and its uses."
    ]
    
    return base_prompts[:num_prompts]


def create_llm_instance(
    model_path: str,
    quantization: Optional[str] = None,
    kvtuner_config: Optional[str] = None,
    kvtuner_scheme: str = "per_token",
    kvtuner_backend: str = "vanilla",
    **kwargs
):
    """Create LLM instance with specified configuration."""
    try:
        from vllm import LLM
        
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
            "enforce_eager": kwargs.get("enforce_eager", False),
        }
        
        if quantization:
            llm_kwargs["quantization"] = quantization
            
            if quantization == "kvtuner":
                llm_kwargs.update({
                    "kvtuner_config_path": kvtuner_config,
                    "kvtuner_scheme": kvtuner_scheme,
                    "kvtuner_backend": kvtuner_backend,
                })
        
        print(f"Creating LLM with configuration:")
        for key, value in llm_kwargs.items():
            print(f"  {key}: {value}")
        
        return LLM(**llm_kwargs)
        
    except ImportError:
        print("Error: vLLM not available. Using mock implementation.")
        return MockLLM()


class MockLLM:
    """Mock LLM for testing when vLLM is not available."""
    
    def generate(self, prompts, sampling_params):
        """Mock generate method."""
        time.sleep(0.1 * len(prompts))  # Simulate processing time
        
        class MockOutput:
            def __init__(self, text):
                self.text = text
        
        class MockResult:
            def __init__(self, prompt):
                self.outputs = [MockOutput(f"Mock response to: {prompt[:50]}...")]
        
        return [MockResult(prompt) for prompt in prompts]


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize monitoring
    memory_monitor = MemoryMonitor()
    benchmark = PerformanceBenchmark()
    
    print("=" * 60)
    print("KVTuner + vLLM Complete Integration Example")
    print("=" * 60)
    
    # Check configurations
    if args.kvtuner_config and not os.path.exists(args.kvtuner_config):
        print(f"Warning: KVTuner config not found: {args.kvtuner_config}")
    
    # Get test prompts
    prompts = get_test_prompts(args.num_prompts)
    
    # Create sampling parameters
    try:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.9,
            max_tokens=args.max_tokens
        )
    except ImportError:
        sampling_params = None  # Mock
    
    # Results storage
    all_results = {}
    
    memory_monitor.record("Initial")
    
    # 1. KVTuner Quantization
    print(f"\n1. Testing KVTuner Quantization")
    print(f"   Config: {args.kvtuner_config}")
    print(f"   Scheme: {args.kvtuner_scheme}")
    print(f"   Backend: {args.kvtuner_backend}")
    
    try:
        kvtuner_llm = create_llm_instance(
            args.model,
            quantization="kvtuner",
            kvtuner_config=args.kvtuner_config,
            kvtuner_scheme=args.kvtuner_scheme,
            kvtuner_backend=args.kvtuner_backend,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        
        memory_monitor.record("After KVTuner LLM")
        
        kvtuner_result = benchmark.run_benchmark(
            kvtuner_llm, prompts, sampling_params, "KVTuner"
        )
        all_results["kvtuner"] = kvtuner_result
        
        memory_monitor.record("After KVTuner Inference")
        
        del kvtuner_llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"KVTuner test failed: {e}")
        all_results["kvtuner"] = {"error": str(e)}
    
    # 2. Baseline Comparison (if requested)
    if args.compare_baseline:
        print(f"\n2. Testing Baseline (No Quantization)")
        
        try:
            baseline_llm = create_llm_instance(
                args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization
            )
            
            memory_monitor.record("After Baseline LLM")
            
            baseline_result = benchmark.run_benchmark(
                baseline_llm, prompts, sampling_params, "Baseline"
            )
            all_results["baseline"] = baseline_result
            
            memory_monitor.record("After Baseline Inference")
            
            del baseline_llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Baseline test failed: {e}")
            all_results["baseline"] = {"error": str(e)}
    
    # 3. FP8 Comparison (if requested)
    if args.compare_fp8:
        print(f"\n3. Testing FP8 Quantization")
        
        try:
            fp8_llm = create_llm_instance(
                args.model,
                quantization="fp8",
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization
            )
            
            memory_monitor.record("After FP8 LLM")
            
            fp8_result = benchmark.run_benchmark(
                fp8_llm, prompts, sampling_params, "FP8"
            )
            all_results["fp8"] = fp8_result
            
            memory_monitor.record("After FP8 Inference")
            
            del fp8_llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"FP8 test failed: {e}")
            all_results["fp8"] = {"error": str(e)}
    
    # Print summaries
    memory_monitor.print_summary()
    benchmark.print_summary()
    
    # Print comparison
    if len(all_results) > 1:
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        
        configs = list(all_results.keys())
        
        print(f"{'Config':<15} {'Throughput':<12} {'Memory':<10} {'Latency':<10}")
        print("-" * 50)
        
        for config in configs:
            result = all_results[config]
            if "error" not in result:
                throughput = result['throughput_tokens_per_sec']
                latency = result['latency_per_request']
                # Memory would need to be tracked per config
                print(f"{config:<15} {throughput:<12.2f} {'N/A':<10} {latency:<10.2f}")
            else:
                print(f"{config:<15} {'ERROR':<12} {'N/A':<10} {'N/A':<10}")
    
    # Save results
    results_file = output_dir / "kvtuner_integration_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "results": all_results,
            "memory_measurements": memory_monitor.measurements,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Example outputs
    print("\n" + "="*50)
    print("SAMPLE OUTPUTS")
    print("="*50)
    
    for config, result in all_results.items():
        if "error" not in result and "outputs" in result:
            print(f"\n{config.upper()} Sample Output:")
            print("-" * 30)
            print(f"Prompt: {prompts[0]}")
            print(f"Response: {result['outputs'][0][:200]}...")
    
    print("\n🎉 KVTuner + vLLM integration example completed!")


if __name__ == "__main__":
    main()
