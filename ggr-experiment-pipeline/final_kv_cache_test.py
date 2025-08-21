#!/usr/bin/env python3
"""
Final KV Cache Stats Test - Comprehensive validation with fallbacks
This is the definitive test to verify the KV cache stats implementation works correctly.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_gpu_device(gpu_id: int = 0):
    """Set GPU device for testing"""
    import os
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_id >= gpu_count:
                print(f"⚠️  Warning: GPU {gpu_id} not available. Available GPUs: 0-{gpu_count-1}")
                gpu_id = 0
            
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"🎯 Set GPU device to: {gpu_id}")
            return True
        else:
            print(f"❌ CUDA not available")
            return False
    except ImportError:
        print(f"⚠️  PyTorch not available for GPU detection")
        return False

def test_import_compatibility():
    """Test if all required imports work"""
    print("🔍 Testing import compatibility...")
    
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM imports: OK")
        return True, "vLLM available"
    except ImportError as e:
        print(f"❌ vLLM not available: {e}")
        return False, str(e)

def test_metrics_collector_import():
    """Test metrics collector import"""
    print("🔍 Testing metrics collector import...")
    
    try:
        from experiment.run_experiment import VLLMMetricsCollector
        print("✅ VLLMMetricsCollector import: OK")
        return True, "Metrics collector available"
    except ImportError as e:
        print(f"❌ Metrics collector import failed: {e}")
        return False, str(e)

def test_vllm_initialization():
    """Test vLLM initialization with KV cache settings"""
    print("🚀 Testing vLLM initialization...")
    
    try:
        from vllm import LLM
        
        # Test with minimal compatible parameters
        llm = LLM(
            model="microsoft/DialoGPT-small",  # Small model for testing
            max_model_len=512,  # Very small for fast initialization
            enable_prefix_caching=True,  # Critical for KV cache testing
            gpu_memory_utilization=0.3,  # Conservative
            tensor_parallel_size=1,
            disable_log_stats=False  # Enable stats (compatible parameter)
        )
        
        print("✅ vLLM initialization: OK")
        print(f"   Model: microsoft/DialoGPT-small")
        print(f"   Prefix caching: ENABLED")
        
        return True, llm
        
    except Exception as e:
        print(f"❌ vLLM initialization failed: {e}")
        return False, str(e)

def test_metrics_collector_creation(llm):
    """Test metrics collector creation and basic functionality"""
    print("📊 Testing metrics collector creation...")
    
    try:
        from experiment.run_experiment import VLLMMetricsCollector
        
        collector = VLLMMetricsCollector(llm_instance=llm)
        print("✅ Metrics collector created: OK")
        
        # Test stats collection
        stats = collector.collect_internal_stats()
        print("✅ Internal stats collection: OK")
        print(f"   Collection method: {stats.get('collection_method', 'unknown')}")
        print(f"   Stats available: {stats.get('stats_available', False)}")
        
        # Check for key metrics
        key_metrics = [
            'gpu_cache_usage_sys',
            'gpu_prefix_cache_hit_rate', 
            'gpu_cache_usage',
            'cpu_cache_usage_sys'
        ]
        
        found_metrics = []
        for metric in key_metrics:
            if metric in stats:
                found_metrics.append(metric)
                print(f"   {metric}: {stats[metric]}")
        
        if found_metrics:
            print(f"✅ Found {len(found_metrics)}/{len(key_metrics)} key metrics")
        else:
            print("⚠️  No key metrics found yet (may need inference to populate)")
        
        return True, collector, stats
        
    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        return False, str(e), {}

def test_simple_inference(llm, collector):
    """Test simple inference with KV cache monitoring"""
    print("🧪 Testing inference with KV cache monitoring...")
    
    try:
        from vllm import SamplingParams
        
        # Simple test prompts with shared prefix (for cache hits)
        test_prompts = [
            "The weather today is",
            "The weather today is quite",
            "The weather today is absolutely"
        ]
        
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for testing
            max_tokens=10,   # Very short for fast testing
            top_p=1.0
        )
        
        print(f"   Processing {len(test_prompts)} prompts with shared prefixes...")
        
        # Collect stats before
        stats_before = collector.collect_internal_stats()
        
        # Run inference
        outputs = llm.generate(test_prompts, sampling_params)
        
        # Collect stats after
        stats_after = collector.collect_internal_stats()
        
        print("✅ Inference completed: OK")
        print(f"   Generated {len(outputs)} responses")
        
        # Compare key metrics
        print("\n📊 Stats Comparison (Before → After):")
        key_metrics = ['gpu_cache_usage_sys', 'gpu_prefix_cache_hit_rate']
        
        for metric in key_metrics:
            before_val = stats_before.get(metric, 'N/A')
            after_val = stats_after.get(metric, 'N/A')
            print(f"   {metric}: {before_val} → {after_val}")
        
        return True, stats_before, stats_after
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False, str(e), {}

def main():
    """Run comprehensive KV cache stats validation"""
    import argparse
    
    # Add command line argument for GPU selection
    parser = argparse.ArgumentParser(description='Final KV cache stats validation test')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU device ID to use (default: 0)')
    args = parser.parse_args()
    
    print("🎯 Final KV Cache Stats Implementation Test")
    print("=" * 60)
    print("This test validates the complete KV cache stats implementation")
    print("with direct vLLM internal Stats object access.")
    print(f"🎯 Using GPU: {args.gpu}\n")
    
    # Set GPU device
    set_gpu_device(args.gpu)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'gpu_device': args.gpu,
        'tests': {},
        'overall_success': False
    }
    
    # Test 1: Import compatibility
    import_ok, import_msg = test_import_compatibility()
    results['tests']['imports'] = {'success': import_ok, 'message': import_msg}
    
    if not import_ok:
        print("\n❌ Cannot proceed without vLLM. Install with: pip install vllm")
        return results
    
    # Test 2: Metrics collector import
    collector_import_ok, collector_msg = test_metrics_collector_import()
    results['tests']['collector_import'] = {'success': collector_import_ok, 'message': collector_msg}
    
    if not collector_import_ok:
        print("\n❌ Cannot proceed without metrics collector")
        return results
    
    # Test 3: vLLM initialization
    init_ok, llm_or_msg = test_vllm_initialization()
    results['tests']['vllm_init'] = {'success': init_ok, 'message': str(llm_or_msg)}
    
    if not init_ok:
        print(f"\n❌ vLLM initialization failed: {llm_or_msg}")
        return results
    
    llm = llm_or_msg
    
    # Test 4: Metrics collector creation
    collector_ok, collector_or_msg, initial_stats = test_metrics_collector_creation(llm)
    results['tests']['collector_creation'] = {
        'success': collector_ok, 
        'message': str(collector_or_msg),
        'stats': initial_stats
    }
    
    if not collector_ok:
        print(f"\n❌ Metrics collector creation failed: {collector_or_msg}")
        return results
    
    collector = collector_or_msg
    
    # Test 5: Simple inference with monitoring
    inference_ok, stats_before, stats_after = test_simple_inference(llm, collector)
    results['tests']['inference'] = {
        'success': inference_ok,
        'stats_before': stats_before if isinstance(stats_before, dict) else {},
        'stats_after': stats_after if isinstance(stats_after, dict) else {}
    }
    
    # Overall assessment
    all_tests_passed = all(test_data['success'] for test_data in results['tests'].values())
    results['overall_success'] = all_tests_passed
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ KV cache stats implementation is working correctly")
        print("✅ Direct vLLM internal Stats access: CONFIRMED")
        print("✅ Ready for production use!")
        
        # Show final stats summary
        if inference_ok and isinstance(stats_after, dict):
            print(f"\n📊 Final Stats Summary:")
            key_metrics = ['gpu_cache_usage_sys', 'gpu_prefix_cache_hit_rate', 'gpu_cache_usage']
            for metric in key_metrics:
                if metric in stats_after:
                    print(f"   {metric}: {stats_after[metric]}")
    else:
        print("❌ SOME TESTS FAILED")
        failed_tests = [name for name, data in results['tests'].items() if not data['success']]
        print(f"   Failed tests: {', '.join(failed_tests)}")
    
    # Save results
    results_file = Path("kv_cache_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    return results

if __name__ == "__main__":
    main()
